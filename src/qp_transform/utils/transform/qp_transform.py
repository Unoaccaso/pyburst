"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:20:45 am
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

# cuda
import cupy
import cupy.typing
import cupyx.scipy.fft as cufft

# cpu and typing
import numpy

# locals
from utils.commons import BLOCK_SHAPE, CONFIG
from .kernels import compute_Q_fft_phi_tensor, compute_fft_phi_plane


def qp_transform(
    signal_fft_GPU: cupy.typing.NDArray,
    fft_freqs_GPU: cupy.typing.NDArray,
    phi_axis_GPU: cupy.typing.NDArray,
    Q_values_GPU: cupy.typing.NDArray | cupy.float32,
    p_value: numpy.float32,
    sampling_rate: numpy.int32,
) -> cupy.typing.NDArray:
    if isinstance(Q_values_GPU, cupy.ndarray):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        # preallocating the fft-phi plane.
        Q_fft_phi_tensor = cupy.zeros(
            (Q_values_GPU.shape[0], phi_axis_GPU.shape[0], signal_fft_GPU.shape[0]),
            dtype=numpy.complex64,
        )
        height = numpy.int32(Q_fft_phi_tensor.shape[0])
        width = numpy.int32(Q_fft_phi_tensor.shape[1])
        depth = numpy.int32(Q_fft_phi_tensor.shape[2])

        # instatiating cuda variables
        grid_shape = (
            Q_fft_phi_tensor.shape[0] // BLOCK_SHAPE[0] + 1,  # X
            Q_fft_phi_tensor.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
            Q_fft_phi_tensor.shape[2] // BLOCK_SHAPE[2] + 1,  # Z
        )
        block_shape = BLOCK_SHAPE

        # here the qp transform is calculated
        compute_Q_fft_phi_tensor[grid_shape, block_shape](
            phi_axis_GPU,
            signal_fft_GPU,
            fft_freqs_GPU,
            Q_values_GPU,
            p_value,
            sampling_rate,
            height,
            width,
            depth,
            Q_fft_phi_tensor,
        )

        # here the inverse fft is computed and the phi-tau plane is returned
        normalized_Q_tau_phi_tensor = cufft.ifft(Q_fft_phi_tensor).astype(
            numpy.complex64
        )

        return normalized_Q_tau_phi_tensor

    elif isinstance(Q_values_GPU, numpy.float32):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        # preallocating the fft-phi plane.
        fft_phi_plane = cupy.zeros(
            (phi_axis_GPU.shape[0], signal_fft_GPU.shape[0]),
            dtype=numpy.complex64,
        )
        height = numpy.int32(fft_phi_plane.shape[0])
        width = numpy.int32(fft_phi_plane.shape[1])

        # instatiating cuda variables
        grid_shape = (
            fft_phi_plane.shape[0] // BLOCK_SHAPE[0] + 1,  # X
            fft_phi_plane.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
        )
        block_shape = BLOCK_SHAPE

        compute_fft_phi_plane[grid_shape, block_shape](
            phi_axis_GPU,
            signal_fft_GPU,
            fft_freqs_GPU,
            Q_values_GPU,
            p_value,
            sampling_rate,
            height,
            width,
            fft_phi_plane,
        )

        normalized_tau_phi_plane = cufft.ifft(fft_phi_plane).astype(numpy.complex64)

        return normalized_tau_phi_plane

    else:
        raise Exception(
            "Q_values_GPU must be an istance of cupy.ndarray, or numpy.float32"
        )
