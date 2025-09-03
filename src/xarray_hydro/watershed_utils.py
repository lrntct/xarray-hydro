import numpy as np
import geopandas as gpd
import xarray as xr
import polygongrid as pg


def _calculate_res(coords_arr_np: np.ndarray) -> float | None:
    if coords_arr_np is not None and len(coords_arr_np) >= 2:
        # Ensure consistent dtype for subtraction, then convert to float
        res = np.abs(
            coords_arr_np.max().astype(float) - coords_arr_np.min().astype(float)
        ) / (len(coords_arr_np) - 1)
        return float(res)


def get_grid_from_dataset(
    dataset: xr.Dataset | xr.DataArray,
    x_coords: str = "longitude",
    y_coords: str = "latitude",
) -> gpd.GeoDataFrame:
    """take an xarray data structure as an input and return a vector grid as a geodataframe."""

    res_x = _calculate_res(dataset[x_coords].values)
    res_y = _calculate_res(dataset[y_coords].values)

    x_min = dataset[x_coords].min() - res_x / 2
    x_max = dataset[x_coords].max() + res_x / 2
    y_min = dataset[y_coords].min() - res_y / 2
    y_max = dataset[y_coords].max() + res_y / 2

    bounds = (x_min, x_max, y_min, y_max)
    dim_size = (len(dataset[x_coords]), len(dataset[y_coords]))

    my_grid = pg.PolygonGrid(
        bounds, step_size=(res_x, res_y), dim_size=dim_size, properties="grid"
    )
    my_grid.build_grid()
    my_grid.build_geojson()

    grid = gpd.GeoDataFrame.from_features(my_grid.geojson["features"])
    return grid
