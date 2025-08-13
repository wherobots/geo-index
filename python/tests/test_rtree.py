import numpy as np
from arro3.core import list_flatten
from geoindex_rs import rtree as rt


def create_index():
    builder = rt.RTreeBuilder(5)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    return builder.finish()


def test_search():
    tree = create_index()
    result = rt.search(tree, 0.5, 0.5, 1.5, 1.5)
    assert len(result) == 2
    assert result[0].as_py() == 0
    assert result[1].as_py() == 1


def test_rtree():
    builder = rt.RTreeBuilder(5)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()

    boxes = rt.boxes_at_level(tree, 0)
    values = list_flatten(boxes)
    np_arr = np.asarray(values).reshape(-1, 4)
    assert np.all(min_x == np_arr[:, 0])
    assert np.all(min_y == np_arr[:, 1])
    assert np.all(max_x == np_arr[:, 2])
    assert np.all(max_y == np_arr[:, 3])


def test_partitions():
    builder = rt.RTreeBuilder(5, 2)
    min_x = np.arange(5)
    min_y = np.arange(5)
    max_x = np.arange(5, 10)
    max_y = np.arange(5, 10)
    builder.add(min_x, min_y, max_x, max_y)
    tree = builder.finish()

    partitions = rt.partitions(tree)
    indices = partitions["indices"]
    partition_id = partitions["partition_id"]

    assert np.all(np.asarray(indices) == np.arange(5))
    assert len(np.unique(np.asarray(partition_id))) == 3


def test_neighbors():
    """Test basic neighbors functionality"""
    tree = create_index()
    
    # Test neighbors (backward compatibility)
    result = rt.neighbors(tree, 0.0, 0.0, max_results=3)
    assert len(result) <= 3
    result_array = np.asarray(result)
    # Should return indices in order of distance from (0,0)
    assert result_array[0] == 0  # First item should be closest


def test_neighbors_with_distance():
    """Test neighbors with different distance metrics"""
    tree = create_index()
    
    # Test with Euclidean distance
    result_euclidean = rt.neighbors_with_distance(
        tree, 0.0, 0.0, rt.PyDistanceMetric.Euclidean, max_results=3
    )
    assert len(result_euclidean) <= 3
    
    # Test with Haversine distance (should work for geographic coordinates)
    result_haversine = rt.neighbors_with_distance(
        tree, 0.0, 0.0, rt.PyDistanceMetric.Haversine, max_results=3
    )
    assert len(result_haversine) <= 3
    
    # Test with Spheroid distance
    result_spheroid = rt.neighbors_with_distance(
        tree, 0.0, 0.0, rt.PyDistanceMetric.Spheroid, max_results=3
    )
    assert len(result_spheroid) <= 3
    
    # Results should be similar for small distances (but order might vary)
    euclidean_array = np.asarray(result_euclidean)
    haversine_array = np.asarray(result_haversine)
    spheroid_array = np.asarray(result_spheroid)
    
    # All should return the same number of results
    assert len(euclidean_array) == len(haversine_array) == len(spheroid_array)


def test_distance_metric_enum():
    """Test that distance metric enum values work correctly"""
    # Test enum values
    assert rt.PyDistanceMetric.Euclidean == rt.PyDistanceMetric.Euclidean
    assert rt.PyDistanceMetric.Haversine == rt.PyDistanceMetric.Haversine
    assert rt.PyDistanceMetric.Spheroid == rt.PyDistanceMetric.Spheroid
    
    # Test that different enums are not equal
    assert rt.PyDistanceMetric.Euclidean != rt.PyDistanceMetric.Haversine
