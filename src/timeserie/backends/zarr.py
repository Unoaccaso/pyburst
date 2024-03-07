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

from .common import ReadBackend

from timeserie.convert import from_array

import polars
import zarr


import pathlib
import concurrent.futures
import os
import warnings


class ZarrStore(ReadBackend):

    @classmethod
    def open_data(cls, file_path: str | list[str]):
        if isinstance(file_path, str):
            file_path = [file_path]

        futures = []
        # ! TODO: FINIRE IL CODICE PER LA LETTURA DEI FILES, IL PARALLELO DEVE ESSERE SUGLI EVENTI
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for path in file_path:
                future = executor.submit(cls._read_zarr_file, path)
                futures.append(future)

            for future in futures:
                future.result()

    @classmethod
    def save_data(cls, file_path: str): ...

    @classmethod
    def _read_zarr_file(cls, file_path: str):
        path = pathlib.Path(file_path)
        if not path.exists():
            warnings.warn(f"\n{path.absolute()} does not exist, skipping")
            return None
        value_file = list(path.glob("**/values_*.zarr"))

        if not value_file:

            warnings.warn(f"\nno values file found in path, skipping")
            return None
        elif len(value_file) > 1:

            warnings.warn(f"\nmultiple values file found in path, skipping!")
            return None

        metadata_file = list(path.glob("**/meta*.h5"))
        if not metadata_file:
            warnings.warn(f"\nno metadata file found in path, skipping")
            return None
        elif len(metadata_file) > 1:
            warnings.warn(f"\nmultiple metadata file found in path, skipping!")
            return None

        values = zarr.open(value_file[0], mode="r")[:]
        meta = polars.read_parquet(metadata_file[0])
        time_file = list(path.glob("**/gps_times*.zarr"))

        if time_file and len(time_file) == 1:
            gps_times = zarr.open(time_file[0], mode="r")[:]
        else:
            gps_times = None

        return from_array(values=values, gps_times=gps_times)
