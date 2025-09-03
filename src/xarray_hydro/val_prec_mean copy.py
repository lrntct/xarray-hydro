from time import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
import xvec  # noqa: F401

from xarray_hydro.watershed_utils import get_grid_from_dataset


def main():
    t0 = time()

    # project path
    path = Path(__file__).resolve().parent.parent.parent
    print(path)

    ## Importar el shape de cuencas
    watershed_path = path / Path("input") / Path("watershed_area.shp")
    watershed = gpd.read_file(watershed_path)
    watershed.crs = "epsg:4326"
    watershed = watershed[["nodeID", "area", "geometry"]]

    ############################
    ## Generar shape de poligonos de los pixeles que abarcan los datos de ERA5 descargados
    # Pendiente buscar otra alternativa para generar el grid a partir del raster
    df_era5 = xr.open_dataset(
        path
        / Path("input")
        / Path("era5_total_precipitation_2020_hourly_118W-86W_14N-34N_ensemble.nc")
    )
    grid = get_grid_from_dataset(df_era5)
    grid.crs = "epsg:4326"
    print(grid)

    assert False

    ############################
    ## Intersectar las cuencas y el grid.
    intersect = grid.overlay(watershed, how="intersection")
    intersect.to_file(
        path / Path("output") / Path("intersect.shp")
    )  # opcional guardarlo

    ## Generar puntos representativos en los poligonos resultantes de la intersección entre cuencas y el grid.
    intersect_points = intersect.representative_point()
    d = {"No": np.arange(0, len(intersect)), "geometry": intersect_points}
    intersect_points = gpd.GeoDataFrame(d, crs="EPSG:4326")

    ## Copiar el nodeID de las áreas intersectadas a los puntos representativos
    points_id = intersect.sjoin(intersect_points, how="right", predicate="contains")
    points_id = points_id[["nodeID", "geometry"]]
    points_id.crs = "epsg:4326"
    points_id.to_file(
        path / Path("output") / Path("points_id.shp")
    )  # opcional guardarlo
    points_id_coor = gpd.GeoSeries(points_id["geometry"])

    ## Calcular áreas de las subcuencas intersectadas con el grid
    intersect_area = intersect.to_crs(6369)
    intersect_area["area_inter"] = intersect.area
    intersect_area = intersect_area[["nodeID", "area_inter"]]
    intersect_area = intersect_area.set_index("nodeID")
    intersect_area = intersect_area.to_xarray()

    ############################
    ## Cálculo de valores medios por año
    for i in range(2020, 2025):
        var = xr.open_dataset(
            path
            / Path("input")
            / Path(f"era5_total_precipitation_{i}_hourly_118W-86W_14N-34N_ensemble.nc")
        )

        ## Convertir la precipitacion de metros a milimetros
        var = var * 1000
        var = var["tp"].astype("float32")

        ## Extraer los datos del raster en los puntos representativos
        extracted = var.xvec.extract_points(
            points_id_coor, x_coords="longitude", y_coords="latitude"
        )  # , index=True)
        extracted = extracted.to_dataset()
        extracted["geometry"] = points_id["nodeID"].to_numpy()
        extracted = extracted.rename(geometry="nodeID")

        ## Cálculo de valores medios
        area_var = extracted["tp"] * intersect_area["area_inter"]
        var_sum = area_var.groupby("nodeID", restore_coord_dims=True).sum()
        area_ = intersect_area.groupby("nodeID").sum()
        val_mean = var_sum / area_["area_inter"]
        val_mean = val_mean.to_dataset(name="tp")
        val_mean["time"] = var["time"]
        val_mean["number"] = var["number"]
        val_mean = val_mean.astype("float32")
        print(val_mean)

        ## Guardad resultados en formato netcdf o zarr
        """
        if i==2020:
            compressor = BloscCodec(cname='zlib', clevel=5, shuffle='shuffle')  #zarr 3
            val_mean.to_zarr(path + "output/prec_mean.zarr", encoding={'tp': {'compressors': compressor,'chunks': (2920,10,1)}}, mode="w")
        else:    
            val_mean.to_zarr(path + "output/prec_mean.zarr", append_dim="time")
        """
        val_mean.to_netcdf(path / Path("output") / Path(f"prec_mean_{i}.nc"))
        break

    print(f"Elapsed time: {time() - t0:.3f} seconds")


if __name__ == "__main__":
    main()
