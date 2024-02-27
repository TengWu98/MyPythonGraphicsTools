import os
import sys

from scipy import io
import numpy as np

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
# # ******** load from mat/save to mat ******** #
# # ================================================= #
def load_face_areas_from_mat(filename:str):
    """
    @description: Load face areas from a MAT file.
    @param filename: The name of the MAT file to load from.
    @Returns: A numpy array containing the face areas.
    """
    areas = None
    if os.path.exists(filename):
        areas = io.loadmat(filename)['areas']
    return np.transpose(areas)

def load_face_features_from_mat(filename:str):
    """
    @description: Load face features from a MAT file.
    @param filename: The name of the MAT file to load from.
    @Returns: A numpy array containing the face features.
    """
    feature = None
    if os.path.exists(filename):
        feature = io.loadmat(filename)['feature']
    return feature

def load_face_segs_from_mat(filename:str):
    """
    @description: Load face segs from a MAT file.
    @param filename: The name of the MAT file to load from.
    @Returns: A numpy array containing the face segs.
    """
    seg = None
    if os.path.exists(filename):
        seg = io.loadmat(filename)['seg']
    return seg

def save_face_segs_to_mat(filename:str, seg:np.ndarray):
    """
    @description: Save face segmentations to a MAT file.
    @param filename: The name of the MAT file to save to.
    @param seg: The numpy array containing the face segs to save.
    @Returns: None
    """
    io.savemat(filename, {'seg':seg})

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
    accuracy = np.sum((pred_label == gt) * areas) / np.sum(areas)
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