import os
import sys

import numpy as np

import random
from scipy import io
from scipy.spatial.transform import Rotation

import trimesh
import triangle as tr

MESH_EXTENSIONS = [
    '.obj',
    '.off',
]

def is_mesh_file(filename:str):
    """
    @description: Check if a given filename is a mesh file.
    @param filename: The name of the file to check.
    @Returns: True if the filename ends with a mesh file extension, False otherwise.
    """
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

# # ================================================= #
# # ******** load mesh ******** #
# # ================================================= #
def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    """
    @description: Randomly rotate the mesh along different axes.
    @param mesh: The mesh to be rotated.
    @Returns: The rotated mesh.
    """
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh.vertices = rotation.apply(mesh.vertices)
    return mesh

def random_scale(mesh: trimesh.Trimesh):
    """
    @description: Randomly scale the mesh.
    @param mesh: The mesh to be scaled.
    @Returns: The scaled mesh.
    """
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh

def mesh_normalize(mesh: trimesh.Trimesh):
    """
    @description: Normalize the mesh vertices to be within [0, 1].
    @param mesh: The mesh to be normalized.
    @Returns: The normalized mesh.
    """
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices
    return mesh

def load_mesh(path:str, normalize=False, augments=[]):
    """
    @description: Load a mesh from a file and optionally apply normalization and augmentations.
    @param path: Path to the mesh file.
    @param normalize: Whether to normalize the mesh vertices.
    @param augments: List of augmentation methods to apply ('orient' for orientation, 'scale' for scaling).
    @Returns: The processed mesh.
    """
    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)

    if normalize:
        mesh = mesh_normalize(mesh)

    return mesh

# # ================================================= #
# # ******** geometry utils ******** #
# # ================================================= #
def face_areas(verts, faces):
    """
    @description: Calculate the areas of faces in a mesh.
    @param verts: The vertices of the mesh.
    @param faces: The faces of the mesh.
    @Returns: A numpy array containing the area of each face.
    """
    areas = []
    for face in faces:
        t = np.cross(verts[face[1]] - verts[face[0]], 
                     verts[face[2]] - verts[face[0]])
        areas.append(np.linalg.norm(t) / 2)
    return np.array(areas)

def vector_angle(A, B):
    """
    @description: Calculate the angle between two vectors.
    @param A: The first vector.
    @param B: The second vector.
    @Returns: The angle between vectors A and B in radians.
    """
    return np.arccos(np.dot(A, B) / np.linalg.norm(A) / np.linalg.norm(B))

def triangle_angles(triangle):
    """
    @description: Calculate the angles of a triangle.
    @param triangle: The vertices of the triangle.
    @Returns: A numpy array containing the angles of the triangle in radians.
    """  
    a = vector_angle(triangle[1] - triangle[0], triangle[2] - triangle[0])
    b = vector_angle(triangle[2] - triangle[1], triangle[0] - triangle[1])
    c = np.pi - a - b
    return np.array([a, b, c])

def min_triangle_angles(triangle):
    """
    @description: Calculate the minimum angle of a triangle.
    @param triangle: The vertices of the triangle.
    @Returns: The minimum angle of the triangle in radians.
    """
    return triangle_angles(triangle).min()

# # ================================================= #
# # ******** face features ******** #
# # ================================================= #
def face_features_from_hu(mesh: trimesh.Trimesh, request=[]):
    """
    @description: Extract 13-dimensional features for each face of the mesh. From https://github.com/lzhengning/SubdivNet.
    @param mesh: The mesh from which to extract features.
    @param request: List of feature types to extract ('area', 'normal', 'center', 'face_angles', 'curvs').
    @Returns: A numpy array of extracted features.
    """
    faces = mesh.faces
    vertices = mesh.vertices

    face_center = vertices[faces.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[faces[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[faces[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[faces[:, 2]] * face_normals).sum(axis=1),
    ])

    features = []
    if 'area' in request:
        features.append(mesh.area_faces)
    if 'normal' in request:
        features.append(face_normals.T)
    if 'center' in request:
        features.append(face_center.T)
    if 'face_angles' in request:
        features.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        features.append(np.sort(face_curvs, axis=0))

    features = np.vstack(features)

    return features

# # ================================================= #
# # ******** load from mat/save to mat ******** #
# # ================================================= #
def load_face_areas_from_mat(path:str):
    """
    @description: Load face areas from a MAT file.
    @param path: The path of the MAT file to load from.
    @Returns: A numpy array containing the face areas.
    """
    areas = None
    if os.path.exists(path):
        areas = io.loadmat(path)['areas']
    return np.transpose(areas)

def load_face_features_from_mat(path:str):
    """
    @description: Load face features from a MAT file.
    @param path: The path of the MAT file to load from.
    @Returns: A numpy array containing the face features.
    """
    feature = None
    if os.path.exists(path):
        feature = io.loadmat(path)['feature']
    return feature

def load_face_segs_from_mat(path:str):
    """
    @description: Load face segs from a MAT file.
    @param path: The path of the MAT file to load from.
    @Returns: A numpy array containing the face segs.
    """
    seg = None
    if os.path.exists(path):
        seg = io.loadmat(path)['seg']
    return seg

def save_face_segs_to_mat(path:str, seg:np.ndarray):
    """
    @description: Save face segmentations to a MAT file.
    @param path: The path of the MAT file to save to.
    @param seg: The numpy array containing the face segs to save.
    @Returns: None
    """
    io.savemat(path, {'seg':seg})

# # ================================================= #
# # ******** mesh segmentation ******** #
# # ================================================= #
def compute_seg_accuracy(pred_label:np.ndarray, gt:np.ndarray, areas:np.ndarray):
    """
    @description: Compute the segmentation accuracy.
    @param pred_label: The predicted labels for each face.
    @param gt: The ground truth labels for each face.
    @param areas: The area of each face.
    @Returns: The segmentation accuracy as a float.
    """
    accuracy = areas[pred_label == gt].sum() / areas.sum()
    return accuracy

# # ================================================= #
# # ******** dataset ******** #
# # ================================================= #
def get_dataset_info(dataset_name:str):
    """
    @description: Get information about a specific dataset.
    @param dataset_name: The name of the dataset to retrieve information for.
    @Returns: A dictionary containing information about the dataset.
    """
    dataset_name = dataset_name.upper()
    dataset_info = {}

    if dataset_name == 'PSB':
        dataset_info = {
            'name': 'PSB',
            'description': '',
            'category_names': [
                'Vase', 'Teddy', 'Table', 'Plier', 'Octopus', 'Mech',
                'Human', 'Hand', 'Glasses', 'Fourleg', 'Fish', 'Cup', 'Chair', 'Bust',
                'Bird', 'Bearing', 'Armadillo', 'Ant', 'Airplane'
            ],
            'category_nums': 20,
            'category_class_nums': [
                5, 5, 2, 3, 2, 5, 8, 6, 3, 6, 3, 2, 4, 8, 5, 5, 11, 5, 5
            ]
        }

    elif dataset_name == 'COSEG':
        dataset_info = {
            'name': 'COSEG',
            'description': '共11类模型, 带有Large的和TeleAliens是大COSEG数据集(3), 其余都是小COSEG数据集(8)',
            'category_names': [
                'Candelabra', 'Chairs', 'ChairsLarge', 'Fourleg', 'Goblets', 'Guitars',
                'Irons', 'Lamps', 'TeleAliens', 'Vases', 'VasesLarge'
            ],
            'category_class_nums': [
                4, 3, 3, 5, 3, 3, 3, 3, 4, 4
            ],
            'large_coseg_category_names': [
                'ChairsLarge', 'TeleAliens', 'VasesLarge'
            ],
            'small_coseg_category_names': [
                'Candelabra', 'Chairs', 'Fourleg', 'Goblets', 'Guitars', 'Irons', 'Lamps', 'Vases'
            ]
        }

    elif dataset_name == 'HUMANBODY':
        dataset_info = {
            'name': 'Human Body',
            'description': '全是人体模型',
            'category_class_nums': 8
    }
            
    elif dataset_name == 'SHAPENETCORE':
        dataset_info = {
            'name': 'ShapeNetCore',
            'description': '共16类模型',
            'category_names': [
                'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp',
                'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'
            ],
            'category_nums': [500, 76, 55, 500, 500, 69, 500, 392, 500, 445, 202, 184, 275, 66, 152, 500],
            'category_class_nums': [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        }

    else:
        raise ValueError("Unknown dataset name.")

    return dataset_info

if __name__ == '__main__':
    # test load_face_areas_from_mat
    areas = load_face_areas_from_mat('./data/Airplane_1_Areas.mat')
    print(areas)
    print(type(areas))

    # test load_face_areas_from_mat
    feature = load_face_features_from_mat('./data/Airplane_1_747.mat')
    print(feature.shape)
    print(type(feature))

    # test load_face_segs_from_mat
    seg = load_face_segs_from_mat('./data/Airplane_1_Seg.mat')
    print(seg.shape)
    print(type(seg))

    labels = np.ones(seg.shape)
    print(labels)
    print(labels.shape)

    # test save_face_segs_to_mat
    accuracy = compute_seg_accuracy(labels, seg, areas)
    print(accuracy)

    # test save_face_segs_to_mat
    save_face_segs_to_mat('./data/Airplane_1_Seg_My.mat', labels)

    # test get_dataset_info
    psb = get_dataset_info('psb')
    print(psb)

    mesh = load_mesh('./data/Airplane_1.off', normalize=True, augments=['orient', 'scale'])

    # test face_areas
    areas = face_areas(mesh.vertices, mesh.faces)
    print(areas)
    print(areas.shape)

    # test face_features
    features = face_features_from_hu(mesh, request=['area', 'normal', 'center', 'face_angles', 'curvs'])
    print(features)
    print(features.shape)