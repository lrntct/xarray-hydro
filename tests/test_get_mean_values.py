"""
Copyright 2025 The authors

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

import numpy as np

import pytest

from xarray_hydro.watershed_utils import mean_values


def test_mean_values_success(mean_value_data):
    """Test wether the computation of mean values is accurate."""
    test_data, catchments, catchment_id = mean_value_data

    # Calculate results and convert to pandas DataFrame
    ds_results = mean_values(
        dataset=test_data,
        catchments=catchments,
        x_coords="lon",
        y_coords="lat",
        catchment_id="basin_id",
    )
    df_results = ds_results.to_pandas().sort_index()

    # Set catchment id as index and sort (get_mean_values does not work with id as index)
    catchments_idx_sorted = catchments.set_index(catchment_id, drop=True).sort_index()

    assert np.all(
        np.isclose(df_results["temperature"], catchments_idx_sorted["expected_temp"])
    )
    assert np.all(
        np.isclose(
            df_results["precipitation"], catchments_idx_sorted["expected_precip"]
        )
    )


def test_crs_mismatch(mismatched_crs_data):
    test_data, catchments, catchment_id = mismatched_crs_data
    with pytest.raises(ValueError, match="crs must match"):
        mean_values(
            dataset=test_data,
            catchments=catchments,
            x_coords="lon",
            y_coords="lat",
            catchment_id="basin_id",
        )
