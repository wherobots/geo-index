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

    /// Calculate the distance from a query geometry to an indexed item.
    ///
    /// This method allows accessing items via their index, enabling:
    /// - On-demand WKB decoding
    /// - Geometry caching strategies
    /// - Custom storage backends
    /// - Lazy evaluation
    ///
    /// # Arguments
    /// * `item_index` - Index of the item being compared
    /// * `query_geometry` - The query geometry
    /// * `item_bbox` - The bounding box of the item (for optimization)
    ///
    /// # Default Implementation
    /// The default implementation returns `N::max_value()`, which means this method
    /// is optional and only needs to be implemented if you want to use `neighbors_geometry`.
    fn distance_to_item(
        &self,
        _item_index: usize,
        _query_geometry: &Geometry<f64>,
        _item_bbox: (N, N, N, N),
    ) -> N {
        N::max_value()
    }

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
}

/// Adapter that uses a geometry array with a configurable distance metric.
/// This provides backward compatibility with the existing API.
pub struct GeometryArrayAdapter<'a> {
    geometries: &'a [Geometry<f64>],
    base_metric: Box<dyn DistanceMetricBase + 'a>,
}

/// Internal trait for computing geometry-to-geometry distances.
/// This is used by the GeometryArrayAdapter to delegate actual distance calculations.
pub trait DistanceMetricBase {
    /// Calculate the distance between two geometries.
    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> f64;
}

impl DistanceMetricBase for EuclideanDistance {
    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> f64 {
        Euclidean.distance(geom1, geom2)
    }
}

impl DistanceMetricBase for HaversineDistance {
    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> f64 {
        // For Haversine, use centroid-to-centroid distance as approximation
        use geo::algorithm::Centroid;
        let centroid1 = geom1.centroid().unwrap_or(Point::new(0.0, 0.0));
        let centroid2 = geom2.centroid().unwrap_or(Point::new(0.0, 0.0));
        Haversine.distance(centroid1, centroid2)
    }
}

impl DistanceMetricBase for SpheroidDistance {
    fn geometry_to_geometry_distance(&self, geom1: &Geometry<f64>, geom2: &Geometry<f64>) -> f64 {
        // For Geodesic, use centroid-to-centroid distance as approximation
        use geo::algorithm::Centroid;
        let centroid1 = geom1.centroid().unwrap_or(Point::new(0.0, 0.0));
        let centroid2 = geom2.centroid().unwrap_or(Point::new(0.0, 0.0));
        Geodesic.distance(centroid1, centroid2)
    }
}

impl<'a> GeometryArrayAdapter<'a> {
    /// Create a new adapter from a slice of geometries with a specific distance metric.
    pub fn new<M: DistanceMetricBase + 'a>(
        geometries: &'a [Geometry<f64>],
        base_metric: M,
    ) -> Self {
        Self {
            geometries,
            base_metric: Box::new(base_metric),
        }
    }

    /// Create a new adapter with Euclidean distance (for backward compatibility).
    pub fn euclidean(geometries: &'a [Geometry<f64>]) -> Self {
        Self {
            geometries,
            base_metric: Box::new(EuclideanDistance),
        }
    }
}

impl<'a, N: IndexableNum> DistanceMetric<N> for GeometryArrayAdapter<'a> {
    fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
        // Not used for geometry queries, delegate to Euclidean for point-to-point
        EuclideanDistance.distance(x1, y1, x2, y2)
    }

    fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
        // Delegate to Euclidean for bbox distance
        EuclideanDistance.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
    }

    fn distance_to_item(
        &self,
        item_index: usize,
        query_geometry: &Geometry<f64>,
        _item_bbox: (N, N, N, N),
    ) -> N {
        if item_index < self.geometries.len() {
            let distance: f64 = self
                .base_metric
                .geometry_to_geometry_distance(query_geometry, &self.geometries[item_index]);
            N::from_f64(distance).unwrap_or(N::max_value())
        } else {
            N::max_value()
        }
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
        let dist: f64 = adapter.distance_to_item(0, &query, (0.0, 0.0, 0.0, 0.0));
        assert!((dist - 1.414).abs() < 0.01);

        // Test distance to second point
        let dist: f64 = adapter.distance_to_item(1, &query, (3.0, 4.0, 3.0, 4.0));
        assert!((dist - 3.605).abs() < 0.01);
    }

    #[test]
    fn test_wkb_decoding_distance_metric() {
        use geozero::{wkb, GeozeroGeometry};

        /// Custom distance metric that stores WKB-encoded geometries and decodes them on-demand
        struct WkbDistanceMetric<'a> {
            wkb_data: &'a [Vec<u8>], // Array of WKB-encoded geometries
        }

        impl<'a> WkbDistanceMetric<'a> {
            fn new(wkb_data: &'a [Vec<u8>]) -> Self {
                Self { wkb_data }
            }

            /// Decode WKB data on-demand to get geometry
            fn decode_geometry(&self, index: usize) -> Option<Geometry<f64>> {
                if index < self.wkb_data.len() {
                    use geozero::geo_types::GeoWriter;

                    let mut geo_writer = GeoWriter::new();
                    // Pass the byte slice directly to Wkb
                    if wkb::Wkb(self.wkb_data[index].as_slice())
                        .process_geom(&mut geo_writer)
                        .is_ok()
                    {
                        geo_writer.take_geometry()
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }

        impl<'a, N: IndexableNum> DistanceMetric<N> for WkbDistanceMetric<'a> {
            fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                EuclideanDistance.distance(x1, y1, x2, y2)
            }

            fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
                EuclideanDistance.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
            }

            fn distance_to_item(
                &self,
                item_index: usize,
                query_geometry: &Geometry<f64>,
                _item_bbox: (N, N, N, N),
            ) -> N {
                // On-demand WKB decoding
                if let Some(item_geom) = self.decode_geometry(item_index) {
                    let distance = Euclidean.distance(query_geometry, &item_geom);
                    N::from_f64(distance).unwrap_or(N::max_value())
                } else {
                    N::max_value()
                }
            }
        }

        // Create some test WKB data (encoded points)
        let point1 = Geometry::Point(Point::new(0.0, 0.0));
        let point2 = Geometry::Point(Point::new(3.0, 4.0));
        let point3 = Geometry::Point(Point::new(6.0, 8.0));

        // Encode geometries to WKB using geozero
        use geozero::ToWkb;
        let wkb1 = point1.to_wkb(geozero::CoordDimensions::default()).unwrap();
        let wkb2 = point2.to_wkb(geozero::CoordDimensions::default()).unwrap();
        let wkb3 = point3.to_wkb(geozero::CoordDimensions::default()).unwrap();
        let wkb_data = vec![wkb1, wkb2, wkb3];

        // Create the WKB-based distance metric
        let wkb_metric = WkbDistanceMetric::new(&wkb_data);
        let query = Geometry::Point(Point::new(1.0, 1.0));

        // Test distance calculation with on-demand WKB decoding
        let dist: f64 = wkb_metric.distance_to_item(0, &query, (0.0, 0.0, 0.0, 0.0));
        assert!((dist - 1.414).abs() < 0.01); // Distance from (1,1) to (0,0)

        let dist: f64 = wkb_metric.distance_to_item(1, &query, (3.0, 4.0, 3.0, 4.0));
        assert!((dist - 3.605).abs() < 0.01); // Distance from (1,1) to (3,4)

        let dist: f64 = wkb_metric.distance_to_item(2, &query, (6.0, 8.0, 6.0, 8.0));
        assert!((dist - 8.602).abs() < 0.01); // Distance from (1,1) to (6,8)
    }

    #[test]
    fn test_cached_geometry_distance_metric() {
        use std::cell::RefCell;
        use std::collections::HashMap;

        /// Custom distance metric with geometry caching to avoid repeated calculations
        struct CachedDistanceMetric<'a> {
            geometries: &'a [Geometry<f64>],
            cache: RefCell<HashMap<usize, Geometry<f64>>>, // Cache for decoded geometries
            cache_hits: RefCell<usize>,                     // Track cache performance
            cache_misses: RefCell<usize>,
        }

        impl<'a> CachedDistanceMetric<'a> {
            fn new(geometries: &'a [Geometry<f64>]) -> Self {
                Self {
                    geometries,
                    cache: RefCell::new(HashMap::new()),
                    cache_hits: RefCell::new(0),
                    cache_misses: RefCell::new(0),
                }
            }

            /// Get geometry with caching - simulates expensive decode operation
            fn get_cached_geometry(&self, index: usize) -> Option<Geometry<f64>> {
                if index >= self.geometries.len() {
                    return None;
                }

                // Check cache first
                if let Some(cached_geom) = self.cache.borrow().get(&index) {
                    *self.cache_hits.borrow_mut() += 1;
                    return Some(cached_geom.clone());
                }

                // Cache miss - "expensive" operation simulation
                *self.cache_misses.borrow_mut() += 1;
                let geometry = self.geometries[index].clone();

                // Store in cache
                self.cache.borrow_mut().insert(index, geometry.clone());
                Some(geometry)
            }

            fn get_cache_stats(&self) -> (usize, usize) {
                (*self.cache_hits.borrow(), *self.cache_misses.borrow())
            }
        }

        impl<'a, N: IndexableNum> DistanceMetric<N> for CachedDistanceMetric<'a> {
            fn distance(&self, x1: N, y1: N, x2: N, y2: N) -> N {
                EuclideanDistance.distance(x1, y1, x2, y2)
            }

            fn distance_to_bbox(&self, x: N, y: N, min_x: N, min_y: N, max_x: N, max_y: N) -> N {
                EuclideanDistance.distance_to_bbox(x, y, min_x, min_y, max_x, max_y)
            }

            fn distance_to_item(
                &self,
                item_index: usize,
                query_geometry: &Geometry<f64>,
                _item_bbox: (N, N, N, N),
            ) -> N {
                // Use cached geometry access
                if let Some(item_geom) = self.get_cached_geometry(item_index) {
                    let distance = Euclidean.distance(query_geometry, &item_geom);
                    N::from_f64(distance).unwrap_or(N::max_value())
                } else {
                    N::max_value()
                }
            }
        }

        // Create test data
        let geometries = vec![
            Geometry::Point(Point::new(0.0, 0.0)),
            Geometry::Point(Point::new(3.0, 4.0)),
            Geometry::Point(Point::new(6.0, 8.0)),
        ];

        let cached_metric = CachedDistanceMetric::new(&geometries);
        let query = Geometry::Point(Point::new(1.0, 1.0));

        // First access - should be cache misses
        let dist1: f64 = cached_metric.distance_to_item(0, &query, (0.0, 0.0, 0.0, 0.0));
        let dist2: f64 = cached_metric.distance_to_item(1, &query, (3.0, 4.0, 3.0, 4.0));
        let dist3: f64 = cached_metric.distance_to_item(2, &query, (6.0, 8.0, 6.0, 8.0));

        assert!((dist1 - 1.414).abs() < 0.01);
        assert!((dist2 - 3.605).abs() < 0.01);
        assert!((dist3 - 8.602).abs() < 0.01);

        let (hits_after_first, misses_after_first) = cached_metric.get_cache_stats();
        assert_eq!(hits_after_first, 0); // No hits yet
        assert_eq!(misses_after_first, 3); // 3 misses

        // Second access to same geometries - should be cache hits
        let dist1_cached: f64 = cached_metric.distance_to_item(0, &query, (0.0, 0.0, 0.0, 0.0));
        let dist2_cached: f64 = cached_metric.distance_to_item(1, &query, (3.0, 4.0, 3.0, 4.0));

        assert!((dist1_cached - 1.414).abs() < 0.01);
        assert!((dist2_cached - 3.605).abs() < 0.01);

        let (hits_after_second, misses_after_second) = cached_metric.get_cache_stats();
        assert_eq!(hits_after_second, 2); // 2 cache hits
        assert_eq!(misses_after_second, 3); // Still 3 misses total
    }
}
