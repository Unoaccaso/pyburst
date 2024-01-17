"""
Copyright (C) 2024 riccardo felicetti <https://github.com/Unoaccaso>

Created Date: Friday, January 12th 2024, 2:13:14 pm
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
# system libs for package managing
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../"
sys.path.append(PATH_TO_MASTER)

# cuda section
from cupyx import jit
import cupy

# cpu section
import numpy

# fft computation
import cupyx.scipy.fft as cufft

# type hinting
from typing import Union

# errors and wanings
import warnings

# custom settings
import configparser

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
CONFIG = configparser.ConfigParser()
CONFIG.read(PATH_TO_SETTINGS)
from utils.commons import FLOAT_PRECISION, INT_PRECISION, COMPLEX_PRECISION

BLOCK_SHAPE = (
    INT_PRECISION(CONFIG["cuda"]["BlockSizeX"]),
    INT_PRECISION(CONFIG["cuda"]["BlockSizeY"]),
)

qp_transform_kernel = cupy.RawKernel(
    r"""

#include <cupy/complex.cuh>

// those are HARD CODED this is not an elegant solution
#define TILE_DIM_X (int)32
#define TILE_DIM_Y (int)16
#define PI (float)(3.14159265)

extern "C"
__global__ void qp_transform(
    const float *phi_values, 
    const complex<float> *fft_values,
    const float *fft_frequencies,
    const int sampling_rate, 
    const float Q,
    const float p, 
    const int n_rows, 
    const int n_cols,
    complex<float> *fft_phi_norm_plane) {

    const unsigned int fft_freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int phi_idx = blockIdx.y * blockDim.y + threadIdx.y;

    
    // saving the arrays on shared memory to speed - up the calculation
    __shared__ complex<float> _shared_fft_values[TILE_DIM_X];
    __shared__ float _shared_fft_frequencies[TILE_DIM_X];
    __shared__ float _shared_phi_values[TILE_DIM_Y];

    if ((fft_freq_idx == 0) && (phi_idx == 0)){
        // printf("(%u, %u)\n", blockDim.x, blockDim.y);
    }
    
    if ((threadIdx.y == 0) && (fft_freq_idx < n_cols)){
        _shared_fft_values[threadIdx.x] = fft_values[fft_freq_idx];
        _shared_fft_frequencies[threadIdx.x] = fft_frequencies[fft_freq_idx];
    }
    if ((phi_idx < n_rows) && (threadIdx.x == 0)){
        _shared_phi_values[threadIdx.y] = phi_values[phi_idx];
    }
    __syncthreads();
    

    if ((phi_idx < n_rows) && (fft_freq_idx < n_cols)){

        
        const float phi = _shared_phi_values[threadIdx.y];
        const float fft_frequency = _shared_fft_frequencies[threadIdx.x];
        const complex<float> fft_value = _shared_fft_values[threadIdx.x];

        // wavelet parameters
        const complex<float> q_tilde = Q / sqrt(complex<float>(1.0, 2.0 * Q * p));
        const float norm = pow((2.0 * PI / (Q * Q)), (1.0 / 4.0)) * sqrt(1.0 / (2.0 * PI * phi));
        const complex<float> qp_norm =  norm * q_tilde;
        const float exponent = (0.5 * (fft_frequency - phi)) / phi;
        const complex<float> qp_exponent = exponent * q_tilde;

        // const complex<float> qp_exponent_2 = qp_exponent * qp_exponent;

        // the actual wavelet
        const complex<float> wavelet = qp_norm * exp(- qp_exponent * qp_exponent);

        fft_phi_norm_plane[phi_idx * n_cols + fft_freq_idx] = fft_value * wavelet * sqrtf(sampling_rate);
    }
}
""",
    "qp_transform",
)


# WARNING: AT THE TIME OF WRITIN, JIT.RAWKERNEL IS EXPERIMENTAL!!!
@jit.rawkernel()
def fft_phi_plane_kernel(
    phi_values,
    fft_values,
    fft_frequencies,
    sampling_rate,
    P: FLOAT_PRECISION,
    Q: FLOAT_PRECISION,
    n_rows,
    n_cols,
    fft_phi_plane,
):
    # TODO: MAGARI INSERISCI LA FORMULA IN LATEX
    # supposing that the max gridsize is big enough to cover the whole out plane
    fft_freq_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y

    # the grid shape can (and will) in general be larger than the out plane
    if (phi_idx < n_rows) and (fft_freq_idx < n_cols):
        phi = phi_values[phi_idx]
        fft_frequency = fft_frequencies[fft_freq_idx]
        fft_value = fft_values[fft_freq_idx]

        # wavelet parameters
        q_tilde = Q / cupy.sqrt(1.0 + 2.0j * Q * P)
        norm = (
            ((2.0 * cupy.pi / (Q**2)) ** (1.0 / 4.0))
            * cupy.sqrt(1.0 / (2.0 * cupy.pi * phi))
            * q_tilde
        )
        exponent = ((0.5 * q_tilde * (fft_frequency - phi)) / phi) ** 2

        # the actual wavelet
        wavelet = norm * cupy.exp(-exponent)

        # scalar product and normalization for fft
        fft_normalization_factor = cupy.sqrt(sampling_rate)
        fft_phi_plane[phi_idx, fft_freq_idx] = (
            fft_value * wavelet * fft_normalization_factor
        )


def get_tau_phi_plane_cuda(
    signal_fft,
    fft_frequencies,
    alpha,
    P,
    Q,
    phi_range,
    data_segment_duration: float,
    data_sampling_rate: INT_PRECISION,
):
    phi_tiles = get_phi_tiling(
        phi_range,
        alpha,
        Q,
        P,
        data_segment_duration,
        data_sampling_rate,
    )

    # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
    # TODO: la qp transform su un dataset di dimensione grande a piacere.

    fft_phi_plane = cupy.zeros(
        (phi_tiles.shape[0], signal_fft.shape[0]),
        dtype=COMPLEX_PRECISION,
    )

    phi_tiles_GPU = cupy.array(phi_tiles, dtype=FLOAT_PRECISION)
    signal_fft_GPU = cupy.array(signal_fft, dtype=COMPLEX_PRECISION)
    fft_frequencies_GPU = cupy.array(fft_frequencies, dtype=FLOAT_PRECISION)
    grid_shape = (
        fft_phi_plane.shape[1] // BLOCK_SHAPE[1] + 1,  # X
        fft_phi_plane.shape[0] // BLOCK_SHAPE[0] + 1,  # y
    )

    qp_transform_kernel(
        grid_shape,
        BLOCK_SHAPE,
        (
            phi_tiles_GPU,
            signal_fft_GPU,
            fft_frequencies_GPU,
            data_sampling_rate,
            Q,
            P,
            INT_PRECISION(fft_phi_plane.shape[0]),
            INT_PRECISION(fft_phi_plane.shape[1]),
            fft_phi_plane,
        ),
    )
    norm_tau_phi_plane_GPU = cufft.ifft(fft_phi_plane)

    return norm_tau_phi_plane_GPU, phi_tiles_GPU


def get_tau_phi_plane_cp(
    signal_fft,
    fft_frequencies,
    alpha,
    P,
    Q,
    phi_range,
    data_segment_duration: float,
    data_sampling_rate: INT_PRECISION,
):
    phi_tiles = get_phi_tiling(
        phi_range,
        alpha,
        Q,
        P,
        data_segment_duration,
        data_sampling_rate,
    )

    # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
    # TODO: la qp transform su un dataset di dimensione grande a piacere.

    fft_phi_plane = cupy.zeros(
        (phi_tiles.shape[0], signal_fft.shape[0]),
        dtype=COMPLEX_PRECISION,
    )

    phi_tiles_GPU = cupy.array(phi_tiles, dtype=FLOAT_PRECISION)
    signal_fft_GPU = cupy.array(signal_fft, dtype=COMPLEX_PRECISION)
    fft_frequencies_GPU = cupy.array(fft_frequencies, dtype=FLOAT_PRECISION)
    grid_shape = (
        fft_phi_plane.shape[1] // BLOCK_SHAPE[1] + 1,  # X
        fft_phi_plane.shape[0] // BLOCK_SHAPE[0] + 1,  # y
    )
    # here the qp transform is calculated
    fft_phi_plane_kernel[grid_shape, BLOCK_SHAPE](
        phi_tiles_GPU,
        signal_fft_GPU,
        fft_frequencies_GPU,
        data_sampling_rate,
        P,
        Q,
        INT_PRECISION(fft_phi_plane.shape[0]),
        INT_PRECISION(fft_phi_plane.shape[1]),
        fft_phi_plane,
    )
    norm_tau_phi_plane_GPU = cufft.ifft(fft_phi_plane)

    return norm_tau_phi_plane_GPU, phi_tiles_GPU


def get_phi_tiling(
    phi_range: list[Union[float, float]],
    alpha,
    Q,
    P,
    data_segment_duration: float,
    data_sampling_rate: int = INT_PRECISION(
        CONFIG["signal.parameters"]["SamplingRate"]
    ),
    thr_sigmas: int = 4,
):
    # perform sanity check on input to ensure safety limits:
    # Input range should be inside [0, Nyquist_freq] with thr_sigmas
    # of upper and lower space.
    lowest_acceptable_phi = (1 / data_segment_duration) * (
        1 - (thr_sigmas * numpy.sqrt(1 + ((2 * P * Q) ** 2))) / Q
    )
    highest_acceptable_phi = (data_sampling_rate / 2) * (
        1 + (thr_sigmas * numpy.sqrt(1 + ((2 * P * Q) ** 2))) / Q
    )

    if phi_range[0] < lowest_acceptable_phi:
        min_phi = lowest_acceptable_phi
        warnings.warn(
            f"lower bound of frequency range set to {lowest_acceptable_phi} Hz",
            RuntimeWarning,
        )
    else:
        min_phi = phi_range[0]

    if phi_range[1] > highest_acceptable_phi:
        max_phi = highest_acceptable_phi
        warnings.warn(
            f"upper bound of frequency range set to {highest_acceptable_phi} Hz",
            RuntimeWarning,
        )
    else:
        max_phi = phi_range[1]

    n_tiles = numpy.ceil(
        numpy.log(max_phi / min_phi)
        / numpy.log(1.0 + (alpha * numpy.sqrt(1 + ((2 * P * Q) ** 2))) / Q),
    ).astype(INT_PRECISION)

    phi_tiles = numpy.geomspace(
        min_phi,
        max_phi,
        n_tiles,
        dtype=FLOAT_PRECISION,
    )

    return phi_tiles
