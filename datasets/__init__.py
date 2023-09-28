from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .stl import STL10
from .usps import USPS
from .office31 import office31
from .utils import *
from .randaugment import RandAugment
from .mnist_m import MNIST_M
from .syn32 import SYN32

__all__ = ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10','USPS','office31','MNIST_M', 'SYN32')
