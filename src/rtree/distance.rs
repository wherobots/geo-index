//! Distance metrics for spatial queries.
//!
//! This module provides different distance calculation methods for spatial queries,
//! including Euclidean, Haversine, and Spheroid distance calculations.

use crate::r#type::IndexableNum;
use geo::algorithm::{Distance, Euclidean, Geodesic, Haversine};
use geo::{Geometry, Point};

/// A trait for calculating distances between geometries and points.
pub trait DistanceMetric<N: IndexableNum> {
    /// Calculate the distance between two points (x1, y1) and (x2, y2).
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N;

    /// Calculate the distance from a point to a bounding box.
    /// This is used for spatial index optimization during tree traversal.
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N;

    /// Calculate the distance from a point to a geometry.
    fn distance_to_geometry(&self, x: N, y: N, geometry: &Geometry<f64>) -> N;

    /// Calculate the distance between two geometries.
    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N;

    /// Return the maximum distance value for this metric.
    fn max_distance(&self) -> N {
        N::max_value()
    }
}

/// Euclidean distance metric.
///
/// This is the standard straight-line distance calculation suitable for
/// planar coordinate systems. When working with longitude/latitude coordinates,
/// the unit of distance will be degrees.
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl<N: IndexableNum> DistanceMetric<N> for EuclideanDistance {
    #[inline]
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
        let p1 = Point::new(x1.to_f64().unwrap_or(0.0), y1.to_f64().unwrap_or(0.0));
        let p2 = Point::new(x2.to_f64().unwrap_or(0.0), y2.to_f64().unwrap_or(0.0));
        N::from_f64(Euclidean.distance(p1, p2)).unwrap_or(N::max_value())
    }

    #[inline]
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
        let dx = axis_dist(x, min_x, max_x);
        let dy = axis_dist(y, min_y, max_y);
        (dx * dx + dy * dy).sqrt().unwrap_or(N::max_value())
    }

    #[inline]
    fn distance_to_geometry(&self, x: N, y: N, geometry: &Geometry<f64>) -> N {
        let point = Point::new(x.to_f64().unwrap_or(0.0), y.to_f64().unwrap_or(0.0));
        let point_geom = Geometry::Point(point);
        N::from_f64(Euclidean.distance(&point_geom, geometry)).unwrap_or(N::max_value())
    }

    #[inline]
    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
        N::from_f64(Euclidean.distance(geom1, geom2)).unwrap_or(N::max_value())
    }
}

/// Haversine distance metric.
///
/// This calculates the great-circle distance between two points on a sphere.
/// It's more accurate for geographic distances than Euclidean distance.
/// The input coordinates should be in longitude/latitude (degrees), and
/// the output distance is in meters.
#[derive(Debug, Clone, Copy)]
pub struct HaversineDistance {
    /// Earth's radius in meters
    pub earth_radius: f64,
}

impl Default for HaversineDistance {
    fn default() -> Self {
        Self {
            earth_radius: 6378137.0, // WGS84 equatorial radius in meters
        }
    }
}

impl HaversineDistance {
    /// Create a new Haversine distance metric with custom Earth radius.
    pub fn with_radius(earth_radius: f64) -> Self {
        Self { earth_radius }
    }
}

impl<N: IndexableNum> DistanceMetric<N> for HaversineDistance {
    fn distance(&self, lon1: N, lat1: N, lon2: N, lat2: N) -> N {
        let p1 = Point::new(lon1.to_f64().unwrap_or(0.0), lat1.to_f64().unwrap_or(0.0));
        let p2 = Point::new(lon2.to_f64().unwrap_or(0.0), lat2.to_f64().unwrap_or(0.0));
        N::from_f64(Haversine.distance(p1, p2)).unwrap_or(N::max_value())
    }

    fn distance_to_bbox(
        &self,
        lon: N,
        lat: N,
        min_lon: N,
        min_lat: N,
        max_lon: N,
        max_lat: N,
    ) -> N {
        // For geographic distance to bbox, find the closest point on the bbox
        let lon_f = lon.to_f64().unwrap_or(0.0);
        let lat_f = lat.to_f64().unwrap_or(0.0);
        let min_lon_f = min_lon.to_f64().unwrap_or(0.0);
        let min_lat_f = min_lat.to_f64().unwrap_or(0.0);
        let max_lon_f = max_lon.to_f64().unwrap_or(0.0);
        let max_lat_f = max_lat.to_f64().unwrap_or(0.0);

        let closest_lon = lon_f.clamp(min_lon_f, max_lon_f);
        let closest_lat = lat_f.clamp(min_lat_f, max_lat_f);

        let point = Point::new(lon_f, lat_f);
        let closest_point = Point::new(closest_lon, closest_lat);
        N::from_f64(Haversine.distance(point, closest_point)).unwrap_or(N::max_value())
    }

    fn distance_to_geometry(&self, lon: N, lat: N, geometry: &Geometry<f64>) -> N {
        let point = Point::new(lon.to_f64().unwrap_or(0.0), lat.to_f64().unwrap_or(0.0));
        // For Haversine, use point-to-centroid distance as approximation
        match geometry {
            Geometry::Point(p) => {
                N::from_f64(Haversine.distance(point, *p)).unwrap_or(N::max_value())
            }
            _ => {
                // For non-point geometries, use centroid
                use geo::algorithm::Centroid;
                if let Some(centroid) = geometry.centroid() {
                    N::from_f64(Haversine.distance(point, centroid)).unwrap_or(N::max_value())
                } else {
                    N::max_value()
                }
            }
        }
    }

    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
        // For Haversine, use centroid-to-centroid distance as approximation
        use geo::algorithm::Centroid;
        let centroid1 = geom1.centroid().unwrap_or(Point::new(0.0, 0.0));
        let centroid2 = geom2.centroid().unwrap_or(Point::new(0.0, 0.0));
        N::from_f64(Haversine.distance(centroid1, centroid2)).unwrap_or(N::max_value())
    }
}

/// Spheroid distance metric (using Geodesic/Vincenty's formula).
///
/// This calculates the shortest distance between two points on the surface
/// of a spheroid (ellipsoid), providing a more accurate Earth model than
/// a simple sphere. The input coordinates should be in longitude/latitude
/// (degrees), and the output distance is in meters.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpheroidDistance;

impl SpheroidDistance {
    /// Create a new Spheroid distance metric for GRS80 ellipsoid.
    pub fn grs80() -> Self {
        Self
    }
}

impl<N: IndexableNum> DistanceMetric<N> for SpheroidDistance {
    fn distance(&self, lon1: N, lat1: N, lon2: N, lat2: N) -> N {
        let p1 = Point::new(lon1.to_f64().unwrap_or(0.0), lat1.to_f64().unwrap_or(0.0));
        let p2 = Point::new(lon2.to_f64().unwrap_or(0.0), lat2.to_f64().unwrap_or(0.0));
        N::from_f64(Geodesic.distance(p1, p2)).unwrap_or(N::max_value())
    }

    fn distance_to_bbox(
        &self,
        lon: N,
        lat: N,
        min_lon: N,
        min_lat: N,
        max_lon: N,
        max_lat: N,
    ) -> N {
        // Similar to haversine, approximate using closest point on bbox
        let lon_f = lon.to_f64().unwrap_or(0.0);
        let lat_f = lat.to_f64().unwrap_or(0.0);
        let min_lon_f = min_lon.to_f64().unwrap_or(0.0);
        let min_lat_f = min_lat.to_f64().unwrap_or(0.0);
        let max_lon_f = max_lon.to_f64().unwrap_or(0.0);
        let max_lat_f = max_lat.to_f64().unwrap_or(0.0);

        let closest_lon = lon_f.clamp(min_lon_f, max_lon_f);
        let closest_lat = lat_f.clamp(min_lat_f, max_lat_f);

        let point = Point::new(lon_f, lat_f);
        let closest_point = Point::new(closest_lon, closest_lat);
        N::from_f64(Geodesic.distance(point, closest_point)).unwrap_or(N::max_value())
    }

    fn distance_to_geometry(&self, lon: N, lat: N, geometry: &Geometry<f64>) -> N {
        let point = Point::new(lon.to_f64().unwrap_or(0.0), lat.to_f64().unwrap_or(0.0));
        // For Geodesic, use point-to-centroid distance as approximation
        match geometry {
            Geometry::Point(p) => {
                N::from_f64(Geodesic.distance(point, *p)).unwrap_or(N::max_value())
            }
            _ => {
                // For non-point geometries, use centroid
                use geo::algorithm::Centroid;
                if let Some(centroid) = geometry.centroid() {
                    N::from_f64(Geodesic.distance(point, centroid)).unwrap_or(N::max_value())
                } else {
                    N::max_value()
                }
            }
        }
    }

    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> N {
        // For Geodesic, use centroid-to-centroid distance as approximation
        use geo::algorithm::Centroid;
        let centroid1 = geom1.centroid().unwrap_or(Point::new(0.0, 0.0));
        let centroid2 = geom2.centroid().unwrap_or(Point::new(0.0, 0.0));
        N::from_f64(Geodesic.distance(centroid1, centroid2)).unwrap_or(N::max_value())
    }
}

/// A trait for calculating distances using indices rather than direct geometry references.
/// This allows for more flexible implementations including:
/// - On-demand WKB decoding
/// - Caching of decoded geometries
/// - Custom storage backends
/// - Lazy evaluation strategies
pub trait IndexedDistanceMetric<N: IndexableNum> {
    /// Calculate the distance between a query geometry and an indexed item.
    ///
    /// # Arguments
    /// * `query_index` - Index of the query geometry (-1 for external query)
    /// * `item_index` - Index of the item being compared
    /// * `query_geometry` - The query geometry (if query_index is -1)
    /// * `item_bbox` - The bounding box of the item (for optimization)
    fn indexed_distance(
        &self,
        query_index: i32,
        item_index: usize,
        query_geometry: Option<&Geometry<f64>>,
        item_bbox: (N, N, N, N),
    ) -> N;

    /// Calculate the distance from a point to a bounding box (for tree traversal).
    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N;

    /// Return the maximum distance value for this metric.
    fn max_distance(&self) -> N {
        N::max_value()
    }
}

/// Adapter that uses a geometry array with Euclidean distance.
/// This provides backward compatibility with the existing API.
pub struct GeometryArrayAdapter<'a> {
    geometries: &'a [Geometry<f64>],
    distance_metric: Box<dyn DistanceMetric<f64> + 'a>,
}

impl<'a> GeometryArrayAdapter<'a> {
    /// Create a new adapter from a slice of geometries with a specific distance metric.
    pub fn new(
        geometries: &'a [Geometry<f64>],
        distance_metric: Box<dyn DistanceMetric<f64> + 'a>,
    ) -> Self {
        Self {
            geometries,
            distance_metric,
        }
    }

    /// Create a new adapter with Euclidean distance (for backward compatibility).
    pub fn euclidean(geometries: &'a [Geometry<f64>]) -> Self {
        Self {
            geometries,
            distance_metric: Box::new(EuclideanDistance),
        }
    }
}

impl<'a, N: IndexableNum> IndexedDistanceMetric<N> for GeometryArrayAdapter<'a> {
    fn indexed_distance(
        &self,
        _query_index: i32,
        item_index: usize,
        query_geometry: Option<&Geometry<f64>>,
        _item_bbox: (N, N, N, N),
    ) -> N {
        if let Some(query) = query_geometry {
            if item_index < self.geometries.len() {
                let distance: f64 = self
                    .distance_metric
                    .geometry_to_geometry_distance(query, &self.geometries[item_index]);
                N::from_f64(distance).unwrap_or(N::max_value())
            } else {
                N::max_value()
            }
        } else {
            N::max_value()
        }
    }

    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
        // Convert N to f64 for the underlying distance metric
        let x_f64 = x.to_f64().unwrap_or(0.0);
        let y_f64 = y.to_f64().unwrap_or(0.0);
        let min_x_f64 = min_x.to_f64().unwrap_or(0.0);
        let min_y_f64 = min_y.to_f64().unwrap_or(0.0);
        let max_x_f64 = max_x.to_f64().unwrap_or(0.0);
        let max_y_f64 = max_y.to_f64().unwrap_or(0.0);

        let distance: f64 = self
            .distance_metric
            .distance_to_bbox(x_f64, y_f64, min_x_f64, min_y_f64, max_x_f64, max_y_f64);

        N::from_f64(distance).unwrap_or(N::max_value())
    }
}

/// 1D distance from a value to a range.
#[inline]
fn axis_dist<N: IndexableNum>(k: N, min: N, max: N) -> N {
    if k < min {
        min - k
    } else if k <= max {
        N::zero()
    } else {
        k - max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::LineString;
    use geo_types::coord;

    #[test]
    fn test_euclidean_distance() {
        let metric = EuclideanDistance;
        let distance = metric.distance(0.0f64, 0.0f64, 3.0f64, 4.0f64);
        assert!((distance - 5.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_distance() {
        let metric = HaversineDistance::default();
        // Distance between New York and London (approximately)
        let distance = metric.distance(-74.0f64, 40.7f64, -0.1f64, 51.5f64);
        // Should be approximately 5585 km
        assert!((distance - 5585000.0f64).abs() < 50000.0f64);
    }

    #[test]
    fn test_spheroid_distance() {
        let metric = SpheroidDistance;
        // Distance between New York and London (approximately)
        let distance = metric.distance(-74.0f64, 40.7f64, -0.1f64, 51.5f64);
        // Should be approximately 5585 km (slightly different from Haversine)
        assert!((distance - 5585000.0f64).abs() < 50000.0f64);
    }

    #[test]
    fn test_distance_to_geometry_point() {
        let metric = EuclideanDistance;
        let point_geom = Geometry::Point(Point::new(3.0, 4.0));
        let distance: f64 = metric.distance_to_geometry(0.0f64, 0.0f64, &point_geom);
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_geometry_linestring() {
        let metric = EuclideanDistance;
        let line_geom = Geometry::LineString(LineString::new(vec![
            coord! { x: 0.0, y: 5.0 },
            coord! { x: 10.0, y: 5.0 },
        ]));
        let distance: f64 = metric.distance_to_geometry(0.0f64, 0.0f64, &line_geom);
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometry_to_geometry_distance() {
        let metric = EuclideanDistance;
        let point1 = Geometry::Point(Point::new(0.0, 0.0));
        let point2 = Geometry::Point(Point::new(3.0, 4.0));
        let distance: f64 = metric.geometry_to_geometry_distance(&point1, &point2);
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_geometry_distance() {
        let metric = HaversineDistance::default();
        let ny_point = Geometry::Point(Point::new(-74.0, 40.7)); // New York
        let london_point = Geometry::Point(Point::new(-0.1, 51.5)); // London
        let distance: f64 = metric.geometry_to_geometry_distance(&ny_point, &london_point);
        // Should be approximately 5585 km
        assert!((distance - 5585000.0).abs() < 50000.0);
    }

    #[test]
    fn test_geometry_array_adapter() {
        let geometries = vec![
            Geometry::Point(Point::new(0.0, 0.0)),
            Geometry::Point(Point::new(3.0, 4.0)),
            Geometry::LineString(LineString::new(vec![
                geo_types::coord! { x: 0.0, y: 5.0 },
                geo_types::coord! { x: 10.0, y: 5.0 },
            ])),
        ];

        let adapter = GeometryArrayAdapter::euclidean(&geometries);
        let query = Geometry::Point(Point::new(1.0, 1.0));

        // Test distance to first point
        let dist: f64 = adapter.indexed_distance(-1, 0, Some(&query), (0.0, 0.0, 0.0, 0.0));
        assert!((dist - 1.414).abs() < 0.01);

        // Test distance to second point
        let dist: f64 = adapter.indexed_distance(-1, 1, Some(&query), (3.0, 4.0, 3.0, 4.0));
        assert!((dist - 3.605).abs() < 0.01);
    }
}
