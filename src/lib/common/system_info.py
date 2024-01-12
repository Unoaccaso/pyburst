"""
Copyright (C) 2024 riccardo felicetti <https://github.com/Unoaccaso>

Created Date: Friday, January 12th 2024, 11:40:08 am
Author: riccardo felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""

# standard modules
from dataclasses import dataclass
from typing import Union

# cuda
from cupy import cuda

# tables
import polars


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
