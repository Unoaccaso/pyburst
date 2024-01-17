""""""
"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Monday, January 15th 2024, 11:22:27 am
Author: Riccardo Felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""

# system
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../"
sys.path.append(PATH_TO_MASTER)

# custom settings from config file
import configparser

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

# standard modules
from dataclasses import dataclass
from typing import Union
from enum import Enum

# cuda
import cupy
from cupy import cuda

# tables
import polars


class FloatPrecision(Enum):
    FLOAT16 = cupy.float16
    FLOAT32 = cupy.float32
    FLOAT64 = cupy.float64


class IntPrecision(Enum):
    INT16 = cupy.int16
    INT32 = cupy.int32
    INT64 = cupy.int64


class ComplexPrecision(Enum):
    COMPLEX64 = cupy.complex64
    COMPLEX128 = cupy.complex128


FLOAT_PRECISION = FloatPrecision[config["numeric.precision"]["FloatPrecision"]].value
INT_PRECISION = IntPrecision[config["numeric.precision"]["IntPrecision"]].value
COMPLEX_PRECISION = ComplexPrecision[
    config["numeric.precision"]["ComplexPrecision"]
].value


@dataclass
class gpu_info:
    name: str
    clock_MHz: float
    tot_mem_GB: float
    shared_mem_kB: float
    max_threads_per_block: int
    max_block_dims: list[Union[int, int, int]]
    multiprocessor_count: int


def get_sys_info():
    gpu_list = []
    num_gpus = cuda.runtime.getDeviceCount()

    for i in range(num_gpus):
        gpu = cuda.runtime.getDeviceProperties(i)
        name = gpu["name"].decode("utf-8")
        clock_MHz = gpu["clockRate"] / 1000
        tot_mem_GB = gpu["totalGlobalMem"] / (1024) ** 3
        shared_mem_kB = gpu["sharedMemPerBlock"] / (1024)
        max_threads_per_block = gpu["maxThreadsPerBlock"]
        max_block_dims = gpu["maxThreadsDim"]
        multiprocessor_count = gpu["multiProcessorCount"]

        gpu_list.append(
            gpu_info(
                name,
                clock_MHz,
                tot_mem_GB,
                shared_mem_kB,
                max_threads_per_block,
                max_block_dims,
                multiprocessor_count,
            )
        )

    system_info = polars.DataFrame(gpu_list)

    return system_info
