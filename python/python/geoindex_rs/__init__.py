from ._rust import kdtree, rtree
from ._rust import ___version
from .enums import DistanceMetric, RTreeMethod

__version__: str = ___version()

__all__ = [
    "kdtree",
    "rtree", 
    "DistanceMetric",
    "RTreeMethod",
    "__version__",
]
