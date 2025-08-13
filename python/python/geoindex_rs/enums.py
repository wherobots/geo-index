from enum import Enum, auto


class StrEnum(str, Enum):
    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, auto)):
            raise TypeError(
                f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
            )
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self):
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name.lower()


class RTreeMethod(StrEnum):
    Hilbert = auto()
    """Use hilbert curves for sorting the RTree
    """

    STR = auto()
    """Use the Sort-Tile-Recursive algorithm for sorting the RTree
    """


class DistanceMetric(StrEnum):
    Euclidean = auto()
    """Euclidean distance metric for planar coordinates.
    
    This is the standard straight-line distance calculation suitable for
    planar coordinate systems. When working with longitude/latitude coordinates,
    the unit of distance will be degrees.
    """

    Haversine = auto()
    """Haversine distance metric for geographic coordinates.
    
    This calculates the great-circle distance between two points on a sphere.
    It's more accurate for geographic distances than Euclidean distance.
    The input coordinates should be in longitude/latitude (degrees), and
    the output distance is in meters.
    """

    Spheroid = auto()
    """Spheroid distance metric for high-precision geographic coordinates.
    
    This calculates the shortest distance between two points on the surface
    of a spheroid (ellipsoid), providing a more accurate Earth model than
    a simple sphere. The input coordinates should be in longitude/latitude
    (degrees), and the output distance is in meters.
    """
