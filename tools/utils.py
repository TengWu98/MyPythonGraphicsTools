import os
import sys
import uuid

from matplotlib import colors

class Utils:
    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    @staticmethod
    def r2h(x):
        return colors.rgb2hex(tuple(map(lambda y: y / 255., x)))
    
    @staticmethod
    def get_uuid():
        return uuid.uuid4().hex