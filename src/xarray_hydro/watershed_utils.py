"""
Copyright [2025] The authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from copy import deepcopy

# Necessary for weighted mean
import numpy as np
import geopandas as gpd
import xarray as xr
import pyproj
import shapely


def _calculate_res(coords_arr_np: np.ndarray) -> float | None:
    if coords_arr_np is not None and len(coords_arr_np) >= 2:
        coords_min = coords_arr_np.min().astype(float)
        coords_max = coords_arr_np.max().astype(float)
        res = (coords_max - coords_min) / (len(coords_arr_np) - 1)
        return float(res)
    else:
        return None


def polygon_grid_from_dataset(
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

    cols = list(np.arange(x_min, x_max, res_x))
    rows = list(np.arange(y_min, y_max, res_y))

    rings=[shapely.linearrings([[x,y],[x,y+res_y],[x+res_x,y+res_y],[x+res_x,y]]) for x in cols for y in rows]
    polygons=shapely.polygons([rings])
    ds_crs = pyproj.CRS.from_wkt(dataset.attrs["crs_wkt"])
    df_grid = gpd.GeoDataFrame({'geometry':polygons[0]},crs=ds_crs)
    return df_grid


def watershed_areas(watershed: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return areas of the intersection between the catchments and the raster grid.
    If input CRS is not projected, the areas are computed with a custom Equal Earth projection:
    - same ellipsoid as the input CRS,
    - latitude of origin centered on the region of interest
    """
    if watershed.crs.is_geographic:
        # Get data from input
        ellipsoid = watershed.crs.ellipsoid
        min_lon, _, max_lon, _ = watershed.total_bounds
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
        watershed_reproj = watershed.to_crs(crs_eqearth)
    else:
        watershed_reproj = watershed

    # Compute the area
    watershed["area"] = watershed_reproj.area
    return watershed


def get_representative_points(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the representative points of each polygon in the input dataframe.
    The attributes of polygons are preserved."""
    r_points = polygons.copy()
    r_points["geometry"] = polygons.representative_point()
    return r_points


def extract_value_points(
    dataset: xr.Dataset | xr.DataArray,
    points: gpd.GeoDataFrame,
    id: str,
) -> xr.Dataset:
    """ Extract values of dataset variables at each representative points."""
    target_lon = xr.DataArray(points.geometry.x, coords={id: points[id]}, dims=id)
    target_lat = xr.DataArray(points.geometry.y, coords={id: points[id]}, dims=id)
    extracted = dataset.sel(longitude=target_lon, latitude=target_lat, method="nearest")
    return extracted


def _weighted_mean(
    dataset: xr.Dataset | xr.DataArray,
    representative_points: gpd.GeoDataFrame,
    catchment_id: str,
    x_coords: str,
    y_coords: str,
) -> xr.Dataset:
    """Compute the weighted mean of each variables in 'dataset'. """
    # extract values of dataset variables at each representative points
    extracted = extract_value_points(dataset,representative_points,catchment_id)

    # Calculate total catchment area. 
    representative_points.index = representative_points[catchment_id]
    representative_points.drop(catchment_id, axis=1, inplace=True)
    ds_representative_points = representative_points.to_xarray()
    total_catchment_area = ds_representative_points.groupby(catchment_id).sum()[
        "area"
    ]

    # Apply area-weighted mean calculation to all data variables
    area_weighted = extracted * ds_representative_points["area"]
    sum_by_catchment = area_weighted.groupby(
        catchment_id, restore_coord_dims=True
    ).sum()
    val_mean = sum_by_catchment / total_catchment_area

    # Preserve CRS and data type
    val_mean.attrs["crs_wkt"] = dataset.attrs["crs_wkt"]
    for var_name in val_mean.data_vars:
        input_dtype = dataset[var_name].dtype
        val_mean[var_name] = val_mean[var_name].astype(input_dtype)
    return val_mean


def mean_values(
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
    # TODO: check if catchment_id is present in catchments

    # Get the vector grid
    grid = polygon_grid_from_dataset(dataset, x_coords, y_coords)
    # Intersect the catchments with the grid
    catchments_grid_intersections = grid.overlay(catchments, how="intersection")
    # Get the surface areas of each sub-catchments
    intersected_areas = watershed_areas(catchments_grid_intersections)
    # Get representative points for each sub-catchments
    representative_points = get_representative_points(intersected_areas)
    # Finally, calculate the weighted mean
    ds_mean = _weighted_mean(
        dataset,
        representative_points,
        catchment_id=catchment_id,
        x_coords=x_coords,
        y_coords=y_coords,
    )
    return ds_mean