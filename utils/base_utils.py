import os
import sys
import uuid

from matplotlib import colors

def mkdir(path):
    """
    @description: Create a new directory if it does not exist.
    @param: path - The path of the directory to be created.
    @Returns: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_file_extension(filename:str):
    """
    @description: Extract the file extension from a given filename.
    @param filename: The name of the file to extract the extension from.
    @Returns: The file extension of the filename.
    """
    return os.path.splitext(filename)[1]

def r2h(x):
    """
    @description: Convert RGB color values to a hexadecimal color code.
    @param: x - A tuple or list containing three elements representing the RGB values.
    @Returns: A string representing the hexadecimal color code.
    """
    return colors.rgb2hex(tuple(map(lambda y: y / 255., x)))


def get_uuid():
    """
    @description: Generate a random UUID (Universally Unique Identifier) in hexadecimal format.
    @param: None
    @Returns: A string representing the hexadecimal UUID.
    """
    return uuid.uuid4().hex

if __name__ == "__main__":
    # test get_uuid
    print(get_uuid())

    # test r2h
    print(r2h([255, 0, 0]))

    # test mkdir
    mkdir("test")