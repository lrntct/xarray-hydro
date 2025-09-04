from pathlib import Path
from collections.abc import Iterable

import numpy as np
import geopandas as gpd
import xarray as xr
import pyproj

from xarray_hydro.watershed_utils import get_mean_values, get_mean_values_xvect

import matplotlib.pyplot as plt


def calculate_mean(watershed_path: Path, era5_files: Iterable, mean_zarr_path: Path):
    # Project CRS
    crs = pyproj.CRS.from_epsg(4326)

    ## Importar el shape de cuencas
    watershed = gpd.read_file(watershed_path)
    watershed.crs = crs
    watershed = watershed[["nodeID", "area", "geometry"]]
    # print(f"{watershed=}")

    ## Importar datos ERA5
    ds_era5 = xr.open_mfdataset(
        era5_files,
        concat_dim="time",
        combine="nested",
        join="outer",  # 2024 tiene solamente 39 celdas de latitude (14.0 a 33.0), no 41 como las demás
        compat="no_conflicts",
        parallel=True,
    )
    # Fix dtype
    ds_era5["tp"] = ds_era5["tp"].astype(np.float32)
    ds_era5["latitude"] = ds_era5["latitude"].astype(np.float32)
    # Set CRS
    ds_era5.attrs["crs_wkt"] = crs.to_wkt()
    # Rechunk
    ds_era5 = ds_era5.chunk({"longitude": -1, "latitude": -1, "number": -1})
    # Transform to mm
    ds_era5["tp"] = ds_era5["tp"] * 1000
    # print(f"{ds_era5=}")
    # Get mean values
    mean_values = get_mean_values(
        ds_era5,
        watershed,
        catchment_id="nodeID",
        x_coords="longitude",
        y_coords="latitude",
    )
    # Rechunk
    mean_values = mean_values.chunk({"time": -1, "nodeID": 100})
    # ds_era5["tp"] = ds_era5["tp"].astype(np.float32)

    # print(f"{mean_values=}")

    # zarr encoding
    default_encoding = {
        "compressors": {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 7,
            },
        }
    }
    encoding = {}
    for var in mean_values:
        encoding[var] = default_encoding
    # Save to zarr
    mean_values.to_zarr(
        store=mean_zarr_path, mode="w", zarr_format=3, encoding=encoding
    )

    # xvect native implementation
    # mean_values_xvect = get_mean_values_xvect(
    #     ds_era5, watershed, catchment_id="nodeID", x_coords="longitude", y_coords="latitude"
    # )
    # print(f"{mean_values_xvect=}")


def plot_differences(ref_ds, calc_ds, var_name, output_path):
    """
    Create plots showing differences between datasets.
    """

    print(f"\nCreating plots for variable: {var_name}")

    # Align datasets
    ref_aligned, calc_aligned = xr.align(ref_ds, calc_ds, join="inner")

    if var_name not in ref_aligned.data_vars or var_name not in calc_aligned.data_vars:
        print(f"Variable {var_name} not found in both datasets")
        return

    ref_var = ref_aligned[var_name]
    calc_var = calc_aligned[var_name]

    # Calculate differences
    abs_diff = np.abs(ref_var - calc_var)
    rel_diff = np.abs((ref_var - calc_var) / (ref_var + 1e-10))

    if "nodeID" in ref_var.dims and "time" in ref_var.dims:
        # Time series plots for worst nodeIDs
        mean_abs_diff_by_node = abs_diff.mean(dim="time")

        # Handle ensemble dimension if present
        if "number" in mean_abs_diff_by_node.dims:
            mean_abs_diff_by_node = mean_abs_diff_by_node.mean(dim="number")

        # Compute values to handle dask arrays
        mean_abs_diff_values = (
            mean_abs_diff_by_node.compute().values
            if hasattr(mean_abs_diff_by_node, "compute")
            else mean_abs_diff_by_node.values
        )
        worst_nodes = np.argsort(mean_abs_diff_values)[-5:]  # Top 5 worst
        best_nodes = np.argsort(mean_abs_diff_values)[
            :5
        ]  # Top 5 best (smallest differences)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f"Dataset Comparison for {var_name}", fontsize=16)

        # Plot 1: Histogram of absolute differences
        axes[0, 0].hist(
            abs_diff.values.flatten(), bins=50, alpha=0.7, edgecolor="black"
        )
        axes[0, 0].set_xlabel("Absolute Difference")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Absolute Differences")
        axes[0, 0].set_yscale("log")

        # Plot 2: Histogram of relative differences
        axes[0, 1].hist(
            rel_diff.values.flatten(), bins=50, alpha=0.7, edgecolor="black"
        )
        axes[0, 1].set_xlabel("Relative Difference")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Relative Differences")
        axes[0, 1].set_yscale("log")

        # Plot 3: Mean absolute difference by nodeID
        mean_abs_diff_by_node.plot(ax=axes[0, 2])
        axes[0, 2].set_title("Mean Absolute Difference by nodeID")
        axes[0, 2].set_ylabel("Mean Absolute Difference")

        # Plot 4: Time series for worst nodeID
        worst_node_idx = worst_nodes[-1]
        worst_node_id = ref_var.nodeID.values[worst_node_idx]
        ref_var_worst = ref_var.isel(nodeID=worst_node_idx)
        calc_var_worst = calc_var.isel(nodeID=worst_node_idx)

        # Handle ensemble dimension for plotting
        if "number" in ref_var_worst.dims:
            ref_var_worst = ref_var_worst.mean(dim="number")
            calc_var_worst = calc_var_worst.mean(dim="number")

        ref_var_worst.plot(ax=axes[1, 0], label="Reference", alpha=0.7)
        calc_var_worst.plot(ax=axes[1, 0], label="Calculated", alpha=0.7)
        axes[1, 0].set_title(f"Time Series - Worst nodeID: {worst_node_id}")
        axes[1, 0].legend()

        # Plot 5: Scatter plot ref vs calc for worst nodeID
        ref_vals = ref_var_worst.values
        calc_vals = calc_var_worst.values
        axes[1, 1].scatter(ref_vals, calc_vals, alpha=0.6)
        min_val = min(ref_vals.min(), calc_vals.min())
        max_val = max(ref_vals.max(), calc_vals.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 line")
        axes[1, 1].set_xlabel("Reference Values")
        axes[1, 1].set_ylabel("Calculated Values")
        axes[1, 1].set_title(f"Ref vs Calc - Worst nodeID: {worst_node_id}")
        axes[1, 1].legend()

        # Plot 6: Box plot of differences for worst nodeIDs
        top_5_worst = worst_nodes  # We only have 5 worst nodes
        diff_data = []
        node_labels = []
        for node_idx in top_5_worst:
            node_id = ref_var.nodeID.values[node_idx]
            node_diffs_da = abs_diff.isel(nodeID=node_idx)
            # Handle ensemble dimension
            if "number" in node_diffs_da.dims:
                node_diffs_da = node_diffs_da.mean(dim="number")
            node_diffs = node_diffs_da.values
            diff_data.append(node_diffs)
            node_labels.append(f"ID_{node_id}")

        axes[1, 2].boxplot(diff_data, tick_labels=node_labels)
        axes[1, 2].set_xlabel("nodeID")
        axes[1, 2].set_ylabel("Absolute Difference")
        axes[1, 2].set_title("Abs Diff Distribution - Top 5 Worst nodeIDs")
        axes[1, 2].tick_params(axis="x", rotation=45)

        # Plot 7: Time series for best nodeID
        best_node_idx = best_nodes[0]  # Best (smallest difference)
        best_node_id = ref_var.nodeID.values[best_node_idx]
        ref_var_best = ref_var.isel(nodeID=best_node_idx)
        calc_var_best = calc_var.isel(nodeID=best_node_idx)

        # Handle ensemble dimension for plotting
        if "number" in ref_var_best.dims:
            ref_var_best = ref_var_best.mean(dim="number")
            calc_var_best = calc_var_best.mean(dim="number")

        ref_var_best.plot(ax=axes[2, 0], label="Reference", alpha=0.7)
        calc_var_best.plot(ax=axes[2, 0], label="Calculated", alpha=0.7)
        axes[2, 0].set_title(f"Time Series - Best nodeID: {best_node_id}")
        axes[2, 0].legend()

        # Plot 8: Scatter plot ref vs calc for best nodeID
        ref_vals_best = ref_var_best.values
        calc_vals_best = calc_var_best.values
        axes[2, 1].scatter(ref_vals_best, calc_vals_best, alpha=0.6, color="green")
        min_val_best = min(ref_vals_best.min(), calc_vals_best.min())
        max_val_best = max(ref_vals_best.max(), calc_vals_best.max())
        axes[2, 1].plot(
            [min_val_best, max_val_best],
            [min_val_best, max_val_best],
            "r--",
            label="1:1 line",
        )
        axes[2, 1].set_xlabel("Reference Values")
        axes[2, 1].set_ylabel("Calculated Values")
        axes[2, 1].set_title(f"Ref vs Calc - Best nodeID: {best_node_id}")
        axes[2, 1].legend()

        # Plot 9: Box plot of differences for best nodeIDs
        top_5_best = best_nodes  # We only have 5 best nodes
        diff_data_best = []
        node_labels_best = []
        for node_idx in top_5_best:
            node_id = ref_var.nodeID.values[node_idx]
            node_diffs_da = abs_diff.isel(nodeID=node_idx)
            # Handle ensemble dimension
            if "number" in node_diffs_da.dims:
                node_diffs_da = node_diffs_da.mean(dim="number")
            node_diffs = node_diffs_da.values
            diff_data_best.append(node_diffs)
            node_labels_best.append(f"ID_{node_id}")

        axes[2, 2].boxplot(diff_data_best, tick_labels=node_labels_best)
        axes[2, 2].set_xlabel("nodeID")
        axes[2, 2].set_ylabel("Absolute Difference")
        axes[2, 2].set_title("Abs Diff Distribution - Top 5 Best nodeIDs")
        axes[2, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_file = output_path / f"comparison_plots_{var_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to: {plot_file}")


def main():
    base_path = Path(__file__).resolve().parent
    data_path = base_path / Path("data")
    input_path = data_path / Path("input")
    output_path = data_path / Path("output")

    watershed_path = input_path / Path("watershed_area.shp")

    era5_pattern = "era5_total_precipitation_*_hourly_118W-86W_14N-34N_ensemble.nc"
    era5_files = input_path.glob(era5_pattern)

    mean_zarr_path = output_path / Path(
        "era5_total_precipitation_hourly_118W-86W_14N-34N_ensemble_mean_by_catchment.zarr"
    )
    ref_mean_files = output_path.glob("prec_mean_*.nc")

    ## Calculate means
    # calculate_mean(watershed_path, era5_files, mean_zarr_path)

    # reference means
    ref_means = xr.open_mfdataset(ref_mean_files)
    print("Reference means:")
    print(ref_means)

    # Calculated mean
    calc_means = xr.open_dataset(mean_zarr_path)
    print("\nCalculated means:")
    print(calc_means)

    # Compare datasets
    # compare_datasets(ref_means, calc_means)

    # Create plots
    common_vars = set(ref_means.data_vars) & set(calc_means.data_vars)
    for var in common_vars:
        plot_differences(ref_means, calc_means, var, output_path)



if __name__ == "__main__":
    main()
