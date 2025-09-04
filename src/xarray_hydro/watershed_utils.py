from copy import deepcopy

import numpy as np
import geopandas as gpd
import xarray as xr
import polygongrid as pg
import pyproj


def _calculate_res(coords_arr_np: np.ndarray) -> float | None:
    if coords_arr_np is not None and len(coords_arr_np) >= 2:
        coords_min = coords_arr_np.min().astype(float)
        coords_max = coords_arr_np.max().astype(float)
        res = (coords_max - coords_min) / (len(coords_arr_np) - 1)
        return float(res)
    else:
        return None


def get_grid_from_dataset(
    dataset: xr.Dataset | xr.DataArray,
    x_coords: str = "longitude",
    y_coords: str = "latitude",
) -> gpd.GeoDataFrame:
    """Take an xarray data structure as an input and return a vector grid as a geodataframe."""

    # Get all grid input values from the xarray dataset
    res_x = _calculate_res(dataset[x_coords].values)
    res_y = _calculate_res(dataset[y_coords].values)

    x_min = dataset[x_coords].min() - res_x / 2
    x_max = dataset[x_coords].max() + res_x / 2
    y_min = dataset[y_coords].min() - res_y / 2
    y_max = dataset[y_coords].max() + res_y / 2

    bounds = (x_min, x_max, y_min, y_max)
    dim_size = (len(dataset[x_coords]), len(dataset[y_coords]))

    # Create the geojson. Any inconsistencies between inputs should raise an error
    # Might be possible to implement this in house. polygongrid package seems unmaintained.
    my_grid = pg.PolygonGrid(
        bounds, step_size=(res_x, res_y), dim_size=dim_size, properties="grid"
    )
    my_grid.build_grid()
    my_grid.build_geojson()

    # GeoJSON to geopandas
    ds_crs = pyproj.CRS.from_wkt(dataset.attrs["crs_wkt"])
    df_grid = gpd.GeoDataFrame.from_features(my_grid.geojson["features"], crs=ds_crs)
    # drop uneeded columns
    return df_grid.drop(["cell_id", "row", "column"], axis=1)


def get_intersected_areas(intersect: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return areas of the intersection between the catchments and the raster grid.
    If input CRS is not projected, the areas are computed with a custom Equal Earth projection:
    - same ellipsoid as the input CRS,
    - latitude of origin centered on the region of interest
    """
    if intersect.crs.is_geographic:
        # Get data from input
        ellipsoid = intersect.crs.ellipsoid
        min_lon, _, max_lon, _ = intersect.total_bounds
        mean_lon = (min_lon + max_lon) / 2
        # Create custom CRS
        prime_meridian = pyproj.crs.datum.CustomPrimeMeridian(longitude=mean_lon)
        eq_conversion = pyproj.crs.CoordinateOperation.from_string("+proj=eqearth")
        custom_datum = pyproj.crs.datum.CustomDatum(
            ellipsoid=ellipsoid, prime_meridian=prime_meridian
        )
        crs_eqearth = pyproj.crs.ProjectedCRS(
            name=f"Equal Earth projection on {ellipsoid.name} ellipsoid "
            "and custom prime meridian",
            conversion=eq_conversion,
            geodetic_crs=pyproj.crs.GeographicCRS(datum=custom_datum),
        )
        assert crs_eqearth.is_projected
        # Apply projection
        intersect_reproj = intersect.to_crs(crs_eqearth)
    else:
        intersect_reproj = intersect

    # Compute the area
    intersect["intersected_area"] = intersect_reproj.area
    return intersect


def get_representative_points(intersect: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the representative points of each areas in the input dataframe."""
    intersect_copy = intersect.copy()
    intersect_copy["geometry"] = intersect.representative_point()
    return intersect_copy


def get_mean_values(
    dataset: xr.Dataset | xr.DataArray,
    catchments: gpd.GeoDataFrame,
    catchment_id: str,
    x_coords: str = "longitude",
    y_coords: str = "latitude",
) -> xr.Dataset:
    """Return the mean value of each dataset variable and each catchment.
    CRS of the dataset is taken from the attribute 'crs_wkt'.
    """
    try:
        dataset_crs = pyproj.CRS.from_wkt(dataset.attrs["crs_wkt"])
    except Exception:
        raise ValueError("'dataset' must have a parsable 'crs_wkt' attribute.")
    if catchments.crs is None:
        raise ValueError("'catchments' must have a crs.")
    if catchments.crs != dataset_crs:
        raise ValueError("'catchments' and 'dataset' crs must match.")

    # Get the vector grid
    grid = get_grid_from_dataset(dataset, x_coords, y_coords)
    # Intersect the catchments with the grid
    catchments_grid_intersections = grid.overlay(catchments, how="intersection")
    # Get the surface areas of each sub-catchments
    intersected_areas = get_intersected_areas(catchments_grid_intersections)
    # Get representative points for each sub-catchments
    intersected_nodes = get_representative_points(intersected_areas)
    print(intersected_areas)
    print(intersected_nodes)
    return None
