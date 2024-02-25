"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Sunday, February 25th 2024, 12:40:43 pm
Author: unoaccaso

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""

from .common import SaveBackend, ReadBackend

import polars
import zarr

import pathlib
import concurrent.futures
import os


class ZarrStore(SaveBackend, ReadBackend):

    @classmethod
    def save_data(cls, serie, path: str):
        save_path = os.path.join(path, f"{serie.segment_name}/{serie.detector_id}/")
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

        # Get all attributes of the class as a dictionary
        class_items = dir(serie.__class__)
        properties = {}

        for item in class_items:
            if isinstance(getattr(serie.__class__, item), property):
                properties[item] = getattr(serie, item)

        # Remove the data that will be saved with Zarr
        array_data = {}
        for property_name in serie.array_items:
            array_data[property_name] = properties.pop(property_name)

        # Create a DataFrame with the remaining metadata
        metadata_df = polars.DataFrame(properties)

        # Save the DataFrame in HDF5 format
        metadata_file_name = f"[metadata]_{serie.segment_name}_{serie.detector_id}_{serie.reference_time_gps}.h5"
        metadata_df.write_parquet(os.path.join(save_path, metadata_file_name))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for array_name, array_value in array_data.items():
                executor.submit(
                    cls._write_zarr_file,
                    save_path,
                    f"[{array_name}]_{serie.segment_name}_{serie.detector_id}_{serie.reference_time_gps}.zarr",
                    array_value,
                )

    # Write the data to Zarr files
    @classmethod
    def _write_zarr_file(cls, file_path: str, file_name: str, data):
        zarr.open(
            os.path.join(file_path, file_name),
            mode="w",
            shape=data.shape,
            dtype=data.dtype,
        )[:] = data

    def open_data(cls, path: str | list[str]): ...
