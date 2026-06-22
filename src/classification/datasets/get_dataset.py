from src.classification.datasets.get_mnist import MNISTDataset
from src.classification.datasets.get_imagenet import IMAGENETDataset
from src.utils.path_tools import get_root
from torch.utils.data import Dataset


def get_dataset(model_name: str):
    if model_name == 'lenet':
        pass
    elif model_name in ['alexnet']:
        pass
    else:
        raise ValueError(f'Wrong model: {model_name}')
    return