"""
Fixtures for testing watershed_utils.get_mean_values function.
"""

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import pyproj


@pytest.fixture
def mean_value_data():
    """
    Create synthetic data where expected results are trivially obvious.

    Strategy: Design catchments that cover exact fractions of grid cells
    with simple, uniform data patterns so expected means are easy to deduce.

    Returns:
        tuple: (dataset, catchments, catchment_id_column)
    """
    # Define coordinate system (WGS84)
    crs = pyproj.CRS.from_epsg(4326)
    crs_wkt = crs.to_wkt()

    # Create 2x2 grid with 1-degree resolution for simple calculations
    lon_coords = np.array([101.0, 102.0])  # 2 longitude points
    lat_coords = np.array([46.0, 47.0])  # 2 latitude points
    lon_res = 1.0
    lat_res = 1.0

    # Input data
    temperature_data = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )
    precipitation_data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    # Create xarray Dataset
    dataset = xr.Dataset(
        {
            "temperature": (["lat", "lon"], temperature_data),
            "precipitation": (["lat", "lon"], precipitation_data),
        },
        coords={
            "lon": lon_coords,
            "lat": lat_coords,
        },
        attrs={"crs_wkt": crs_wkt},
    )

    # Define catchments by explicit coverage fractions
    # Format: (name, [(lat_idx, lon_idx, fraction), ...])
    catchment_specs = [
        (
            "half_bottom",
            [
                (0, 0, 0.5),  # 50% of cell (lat=46, lon=101)
                (0, 1, 0.5),  # 50% of cell (lat=46, lon=102)
            ],
        ),
        (
            "single_cell",
            [
                (1, 0, 1.0),  # 100% of cell (lat=47, lon=101)
            ],
        ),
        (
            "quarter_all",
            [
                (0, 0, 0.25),  # 25% of each of the 4 cells
                (0, 1, 0.25),
                (1, 0, 0.25),
                (1, 1, 0.25),
            ],
        ),
    ]

    catchments_data = []
    for name, coverage_spec in catchment_specs:
        # Generate polygon from coverage spec
        geometry = create_catchment_from_coverage(
            coverage_spec, lon_coords, lat_coords, lon_res, lat_res
        )

        # Calculate expected values
        total_weight = sum(frac for _, _, frac in coverage_spec)
        expected_temp = (
            sum(
                temperature_data[lat_idx, lon_idx] * frac
                for lat_idx, lon_idx, frac in coverage_spec
            )
            / total_weight
        )

        expected_precip = (
            sum(
                precipitation_data[lat_idx, lon_idx] * frac
                for lat_idx, lon_idx, frac in coverage_spec
            )
            / total_weight
        )

        catchments_data.append(
            {
                "basin_id": name,
                "expected_temp": expected_temp,
                "expected_precip": expected_precip,
                "geometry": geometry,
            }
        )

    catchments = gpd.GeoDataFrame(catchments_data, crs=crs)
    return dataset, catchments, "basin_id"


def create_catchment_from_coverage(coverage_spec, lons, lats, lon_res, lat_res):
    """
    Create polygons that cover specified fractions of grid cells.

    Args:
        coverage_spec: List of (lat_idx, lon_idx, fraction) tuples
        lons, lats: Coordinate arrays
        lon_res, lat_res: Grid resolution

    Returns:
        Polygon geometry
    """

    # For simple rectangular coverage within each cell
    cell_boxes = []
    for lat_idx, lon_idx, fraction in coverage_spec:
        # Cell boundaries
        lon_min = lons[lon_idx] - lon_res / 2
        lon_max = lons[lon_idx] + lon_res / 2
        lat_min = lats[lat_idx] - lat_res / 2
        lat_max = lats[lat_idx] + lat_res / 2

        # Create sub-box covering the fraction
        # (assuming coverage from bottom of cell)
        lat_coverage = lat_min + (lat_max - lat_min) * fraction

        cell_boxes.append(box(lon_min, lat_min, lon_max, lat_coverage))

    # Union all boxes to create catchment
    return unary_union(cell_boxes)


@pytest.fixture
def mismatched_crs_data():
    """
    Create dataset and catchments with mismatched CRS for error testing.
    """
    # Dataset in WGS84
    dataset_crs = pyproj.CRS.from_epsg(4326)
    dataset = xr.Dataset(
        {
            "temperature": (["latitude", "longitude"], np.array([[20.0]])),
        },
        coords={
            "longitude": np.array([-101.0]),
            "latitude": np.array([47.0]),
        },
        attrs={"crs_wkt": dataset_crs.to_wkt()},
    )

    # Catchments in different CRS
    catchments_crs = pyproj.CRS.from_epsg(32614)  # UTM Zone 14N
    catchment_polygon = Polygon(
        [(300000, 5200000), (400000, 5200000), (400000, 5300000), (300000, 5300000)]
    )

    catchments = gpd.GeoDataFrame(
        {"catchment_id": ["test_basin"], "geometry": [catchment_polygon]},
        crs=catchments_crs,
    )

    return dataset, catchments, "catchment_id"
