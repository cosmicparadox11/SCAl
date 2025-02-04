from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .stl import STL10
from .usps import USPS
from .office31 import office31,office31_Full
from .office_home import OfficeHome,OfficeHome_Full
from .utils import *
from .randaugment import RandAugment
from .mnist_m import MNIST_M
from .syn32 import SYN32
from .visda import VisDA_source ,VisDA_target
from .office_caltech import OfficeCaltech,OfficeCaltech_Full
from .domain_net import    DomainNet ,DomainNet_Full
from .domain_net_s import    DomainNetS ,DomainNetS_Full
__all__ = ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10','USPS','office31','OfficeHome','MNIST_M', 'SYN32','VisDA','OfficeCaltech','DomainNet','DomainNetS' ,'DomainNetS_Full')
