# Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>
#
# Created Date: Thursday, February 22nd 2024, 10:07:32 am
# Author: unoaccaso
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, version 3. This program is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https: //www.gnu.org/licenses/>.


from .common import BackendBase, PathBase

from timeserie.convert import from_array

import polars
import zarr


import pathlib
import concurrent.futures
import warnings
from dataclasses import dataclass


# file pattern: SEGMENT-NAME_DETECTOR-ID_T0_SAMPLING-RATE_DURATION
NAME_PATTERN = "*_*_*_*_*"


@dataclass
class ZarrPath(PathBase):

    def _check_path_firm(self, path_str: pathlib.Path):

        path = pathlib.Path(path_str)

        # check path existance
        if not path.exists():
            raise FileExistsError(f"{path.absolute()} does not exist")

        # check if contains zarr
        data_files = list(path.glob(NAME_PATTERN + ".zarr"))
        if len(data_files) == 0:
            raise FileNotFoundError(
                f"no zarr file found in {path}\nCheck name pattern: SEGMENT-NAME_DETECTOR-ID_T0_SAMPLING-RATE_DURATION.zarr"
            )
        elif len(data_files) > 1:
            raise ValueError(f"Multiple metadata files found!")

        # check if contains meta
        meta_files = list(path.glob("meta_" + NAME_PATTERN + "h5"))
        if len(meta_files) == 0:
            raise FileNotFoundError(
                f"no metadata file found in {path}\nCheck name pattern: meta_SEGMENT-NAME_DETECTOR-ID_T0_SAMPLING-RATE_DURATION.h5"
            )
        elif len(meta_files) > 1:
            raise ValueError(f"Multiple metadata files found!")


class ZarrStore(BackendBase):

    @classmethod
    def open_data(cls, file_paths: str | list):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        zarr_paths = []
        for file_path in file_paths:
            zarr_paths.append(ZarrPath(file_path))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(cls._read_zarr_file, zarr_paths)
            timeseries = list(results)

        return timeseries

    @classmethod
    def _read_zarr_file(cls, file_path: ZarrPath):
        if not isinstance(file_path, ZarrPath):
            raise ValueError(f"Path must be a ZarrPath object")

        meta_path = list(file_path.path.glob("meta_" + NAME_PATTERN + "h5"))[0]
        metadata = polars.read_parquet(meta_path)
        print(metadata)
