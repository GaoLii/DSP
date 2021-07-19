import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.cityscapes_loader16 import cityscapesLoader16
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "cityscapes16": cityscapesLoader16,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '../../dataset/Cityscapes/'
    if name == 'cityscapes16':
        return '../../dataset/Cityscapes/'
    if name == 'gta':
        return '../../dataset/GTA5/'
    if name == 'synthia':
        return '../../dataset/RAND_CITYSCAPES'
