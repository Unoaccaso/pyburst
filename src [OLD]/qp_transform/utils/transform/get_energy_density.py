"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:49:23 am
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
"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:49:23 am
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

# system libs for package managing
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# settings
from utils.commons import CONFIG, BLOCK_SHAPE

# custom
from utils.transform.kernels import compute_energy_density

# cpu
import numpy

# GPU
import cupy
import cupy.typing


def get_energy_density(
    tau_phi_plane: cupy.typing.NDArray,
    phi_axis: cupy.typing.NDArray,
    tau_axis: cupy.typing.NDArray,
    kernel_dim: numpy.int32 = 5,
) -> cupy.typing.NDArray:
    # preallocating the fft-phi plane.
    N_Qs = numpy.int32(CONFIG["computation.parameters"]["N_q"])
    assert N_Qs == tau_phi_plane.shape[0]
    energy_density_GPU = cupy.zeros(
        (
            N_Qs,
            phi_axis.shape[0] - 2 * kernel_dim,
            tau_axis.shape[0] - 2 * kernel_dim,
        ),
        dtype=numpy.float32,
    )
    height = numpy.int32(energy_density_GPU.shape[0])
    width = numpy.int32(energy_density_GPU.shape[1] + 2 * kernel_dim)
    depth = numpy.int32(energy_density_GPU.shape[2] + 2 * kernel_dim)

    # instatiating cuda variables
    grid_shape = (
        energy_density_GPU.shape[0] // BLOCK_SHAPE[0] + 1,  # X
        energy_density_GPU.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
        energy_density_GPU.shape[2] // BLOCK_SHAPE[2] + 1,  # Z
    )
    block_shape = BLOCK_SHAPE

    compute_energy_density[grid_shape, block_shape](
        tau_phi_plane,
        kernel_dim,
        height,
        width,
        depth,
        energy_density_GPU,
    )

    return energy_density_GPU
