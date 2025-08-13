use std::sync::Arc;

use arrow_array::UInt32Array;
use arrow_buffer::ScalarBuffer;
use geo_index::rtree::distance::{DistanceMetric, EuclideanDistance, HaversineDistance, SpheroidDistance};
use geo_index::rtree::RTreeIndex;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::rtree::input::PyRTreeRef;

#[derive(Debug, Clone)]
#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq)]
pub enum PyDistanceMetric {
    #[pyo3(name = "euclidean")]
    Euclidean,
    #[pyo3(name = "haversine")]
    Haversine,
    #[pyo3(name = "spheroid")]
    Spheroid,
}

impl PyDistanceMetric {
    fn to_metric_f32(&self) -> Box<dyn DistanceMetric<f32>> {
        match self {
            PyDistanceMetric::Euclidean => Box::new(EuclideanDistance),
            PyDistanceMetric::Haversine => Box::new(HaversineDistance::default()),
            PyDistanceMetric::Spheroid => Box::new(SpheroidDistance::default()),
        }
    }

    fn to_metric_f64(&self) -> Box<dyn DistanceMetric<f64>> {
        match self {
            PyDistanceMetric::Euclidean => Box::new(EuclideanDistance),
            PyDistanceMetric::Haversine => Box::new(HaversineDistance::default()),
            PyDistanceMetric::Spheroid => Box::new(SpheroidDistance::default()),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (index, x, y, *, max_results = None, max_distance = None))]
pub fn neighbors(
    py: Python,
    index: PyRTreeRef,
    x: f64,
    y: f64,
    max_results: Option<usize>,
    max_distance: Option<f64>,
) -> PyResult<PyObject> {
    let results = match index {
        PyRTreeRef::Float32(tree) => tree.neighbors(
            x as f32,
            y as f32,
            max_results,
            max_distance.map(|x| x as f32),
        ),
        PyRTreeRef::Float64(tree) => tree.neighbors(x, y, max_results, max_distance),
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    Ok(PyArray::from_array_ref(Arc::new(results))
        .to_arro3(py)?
        .unbind())
}

#[pyfunction]
#[pyo3(signature = (index, x, y, distance_metric, *, max_results = None, max_distance = None))]
pub fn neighbors_with_distance(
    py: Python,
    index: PyRTreeRef,
    x: f64,
    y: f64,
    distance_metric: PyDistanceMetric,
    max_results: Option<usize>,
    max_distance: Option<f64>,
) -> PyResult<PyObject> {
    let results = match index {
        PyRTreeRef::Float32(tree) => {
            let metric = distance_metric.to_metric_f32();
            tree.neighbors_with_distance(
                x as f32,
                y as f32,
                max_results,
                max_distance.map(|x| x as f32),
                metric.as_ref(),
            )
        },
        PyRTreeRef::Float64(tree) => {
            let metric = distance_metric.to_metric_f64();
            tree.neighbors_with_distance(
                x,
                y,
                max_results,
                max_distance,
                metric.as_ref(),
            )
        },
    };
    let results = UInt32Array::new(ScalarBuffer::from(results), None);
    Ok(PyArray::from_array_ref(Arc::new(results))
        .to_arro3(py)?
        .unbind())
}
