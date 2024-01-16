import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(Path(__file__).resolve().parent)
sys.path.insert(0, BASE_DIR)

from scipy import io
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from matplotlib.tri import Triangulation

from tools.utils import Utils

# surface_color = Utils.r2h((255, 230, 205))
# edge_color = r2h((90, 90, 90))
# edge_colors = (r2h((15, 167, 175)), r2h((230, 81, 81)), r2h((142, 105, 252)), r2h((248, 235, 57)),
#                r2h((51, 159, 255)), r2h((225, 117, 231)), r2h((97, 243, 185)), r2h((161, 183, 196)))

MESH_EXTENSIONS = [
    '.obj',
    '.off',
]

class TriangleMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)
        self.normals = None

class MeshUtils:
    @staticmethod
    def is_mesh_file(filename:str):
        """
        @description: 判断给定的文件是否是mesh文件
        @param: filename -- 文件名
        @Returns: 是否是mesh文件
        """
        return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

    @staticmethod
    def get_mesh_file_extension(filename:str):
        """
        @description: 获取mesh文件的后缀名
        @param: filename -- 文件名
        @Returns: mesh文件的后缀名
        """
        return os.path.splitext(filename)[1]

class MeshIO():
    @staticmethod
    def read_mesh(filename:str):
        """
        @description: 读取mesh文件
        @param: filename -- 文件名
        @Returns: 读取的mesh
        """
        triangle_mesh = None

        if not os.path.exists(filename):
            print('文件不存在: {}'.format(filename))
            return triangle_mesh

        if(MeshUtils.get_mesh_file_extension(filename) == '.obj'):
            o3d_mesh = o3d.io.read_triangle_mesh(filename)
            vertices = o3d_mesh.vertices
            faces = o3d_mesh.triangles
            triangle_mesh = TriangleMesh(vertices, faces)
            if o3d_mesh.has_vertex_normals():
                triangle_mesh.normals = np.asarray(o3d_mesh.vertex_normals)
        elif(MeshUtils.get_mesh_file_extension(filename) == '.off'):
            tri_mesh = trimesh.load(filename)
            vertices = tri_mesh.vertices
            faces = tri_mesh.faces
            triangle_mesh = TriangleMesh(vertices, faces)

        return triangle_mesh
    
    @staticmethod
    def write_mesh(mesh:TriangleMesh, filename:str):
        """
        @description: 写入mesh文件
        @param: mesh -- mesh数据, filename -- 写入的文件名
        @Returns: None, 直接生成指定的mesh文件
        """
        vertices = mesh.vertices
        faces = mesh.faces

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3dVector(faces)

        Utils.mkdir(filename)

        o3d.io.write_triangle_mesh(filename, o3d_mesh, write_ascii=True, print_progress=True)

        pass
    
    @staticmethod
    def write_mesh_with_face_colors(mesh:TriangleMesh, face_labels, filename):
        pass

class MeshPlot:
    @staticmethod
    def plot_mesh_with_face_values(mesh:TriangleMesh, face_values=None):
        vertices = mesh.vertices
        faces = mesh.faces

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vtx = vertices[faces]
        pass

    @staticmethod
    def plot_mesh_with_face_labels(mesh:TriangleMesh, face_labels=None):
        vertices = mesh.vertices
        faces = mesh.faces

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pass

class MeshProcessing:
    @staticmethod
    def compute_vertex_normals(mesh:TriangleMesh):
        pass

    @staticmethod
    def covert_mesh_format(mesh:TriangleMesh, format='obj'):
        pass

    @staticmethod
    def convert_to_manifold(filename):
        pass

    @staticmethod
    def compute_geodesic_distance_matrix(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_vertex_agd(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_face_gaussian_curvature(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_vertex_gaussian_curvature(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_face_sdf(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_sihks(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_wks(mesh:TriangleMesh):
        pass
    
    @staticmethod
    def compute_dihedral_angles_matrix(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_face_areas(mesh:TriangleMesh):
        pass

    @staticmethod
    def compute_vertex_areas(mesh:TriangleMesh):
        pass
    
    @staticmethod
    def load_face_areas_from_mat(filename:str):
        """
        @description: 从mat文件中读取mesh每个面片的面积
        @param: filename -- mat文件名
        @Returns: mesh每个面片的面积组成的ndarray向量(1*N), N为面片数
        """
        areas = None
        if os.path.exists(filename):
            areas = io.loadmat(filename)['areas']
        return areas
    
    @staticmethod
    def load_face_feature_from_mat(filename:str):
        """
        @description: 从mat文件中读取mesh每个面片的特征
        @param: filename -- mat文件名
        @Returns: mesh每个面片的特征组成的ndarray向量(N*F), N为面片数, F为特征维度
        """
        feature = None
        if os.path.exists(filename):
            feature = io.loadmat(filename)['feature']
        return feature

class MeshSegmentation:
    @staticmethod
    def load_seg_file_from_mat(filename:str):
        """
        @description: 从mat文件中读取mesh的分割标签
        @param: filename -- mat文件名
        @Returns: mesh的分割标签组成的ndarray向量(N*1), N为面片数
        """
        seg = None
        if os.path.exists(filename):
            seg = io.loadmat(filename)['seg']
        return seg

    @staticmethod
    def compute_segmentation_accuracy(pred_label:np.ndarray, gt:np.ndarray, areas:np.ndarray):
        """
        @description: 计算分割结果的准确率
        @param: pred_label -- 分割结果(1*N), gt -- ground truth(1*N), areas -- 每个面片的面积(1*N), N为面片数
        @Returns: 分割结果的准确率
        """
        accuracy = np.sum((pred_label == gt) * areas) / np.sum(areas)
        return accuracy

    @staticmethod
    def compute_segmentation_iou(pred, gt):
        pass

    @staticmethod
    def save_seg_file_to_mat(filename:str, seg:np.ndarray):
        """
        @description: 将分割结果保存到mat文件中
        @param: filename -- mat文件名, seg -- 分割结果(1*N), N为面片数
        @Returns: None
        """
        io.savemat(filename, {'seg':seg})

    @staticmethod
    def get_dataset_info(dataset_name:str):
        """
        @description: 获取数据集的相关信息
        @param: dataset_name -- 数据集名称
        @Returns: 数据集的相关信息
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

    @staticmethod
    def graph_cut(mesh:TriangleMesh, smoothing_lambda, prob_matrix, cluster_to_segment=False):
        pass

    @staticmethod
    def compute_segmentation_score(original_mesh_filename, seg_mesh_filename, gt_mesh_filename):
        pass

    @staticmethod
    def over_segmentation(mesh:TriangleMesh, num_clusters):
        pass

    @staticmethod
    def seg_using_sdf(mesh:TriangleMesh, num_clusters, smoothing_lambda):
        pass

class MeshGeneration:
    pass

if __name__ == '__main__':
    mesh = MeshIO.read_mesh('E:/1_wuteng/科研/Code/MyGraphicsTools/Tests/data/1.off')
    vertices = mesh.vertices
    faces = mesh.faces
    print(vertices.shape)
    print(faces.shape)

    areas = MeshProcessing.load_face_areas_from_mat('E:/1_wuteng/科研/Code/MyGraphicsTools/Tests/data/Airplane_1_Areas.mat')
    print(areas.shape)
    print(type(areas))

    feature = MeshProcessing.load_face_feature_from_mat('E:/1_wuteng/科研/Code/MyGraphicsTools/Tests/data/Airplane_1_747.mat')
    print(feature.shape)
    print(type(feature))

    seg = MeshSegmentation.load_seg_file_from_mat('E:/1_wuteng/科研/Code/MyGraphicsTools/Tests/data/Airplane_1_Seg.mat')
    print(seg.shape)
    print(type(seg))

    labels = np.ones(seg.shape) + 1
    print(labels.shape)

    accuracy = MeshSegmentation.compute_segmentation_accuracy(np.transpose(labels), np.transpose(seg), areas)
    print(accuracy)

    MeshSegmentation.save_seg_file_to_mat('E:/1_wuteng/科研/Code/MyGraphicsTools/Tests/data/Airplane_1_Seg_My.mat', labels)

    psb = MeshSegmentation.get_dataset_info('psb')
    print(psb)
    