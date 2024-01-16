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
    int(CONFIG["cuda"]["BlockSizeX"]),
    int(CONFIG["cuda"]["BlockSizeY"]),
)
ALPHA = FLOAT_PRECISION(CONFIG["computation.parameters"]["Alpha"])


# WARNING: AT THE TIME OF WRITIN, JIT.RAWKERNEL IS EXPERIMENTAL!!!
@jit.rawkernel()
def fft_phi_plane_kernel(
    phi_values,
    fft_values,
    P: FLOAT_PRECISION,
    Q: FLOAT_PRECISION,
    fft_frequencies,
    fft_phi_plane,
):
    """fft_phi_plane

    [extended_summary]

    Args:
        phi ([type]): [description]
        fft_values ([type]): [description]
        P ([type]): [description]
        Q ([type]): [description]
        fft_frequencies ([type]): [description]
        fft_phi_plane ([type]): [description]

    """
    # TODO: MAGARI INSERISCI LA FORMULA IN LATEX
    # supposing that the max gridsize is big enough to cover the whole out plane
    fft_freq_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y

    # the grid shape can (and will) in general be larger than the out plane
    if (phi_idx < fft_phi_plane.shape[0]) and (fft_freq_idx < fft_phi_plane.shape[1]):
        # build the wavelet
        phi = phi_values[phi_idx]
        fft_frequency = fft_frequencies[fft_freq_idx]
        fft_value = fft_values[fft_freq_idx]
        q_tilde = Q / cupy.sqrt(1.0 + 2.0j * Q * P)
        norm = (
            ((2.0 * cupy.pi / (Q**2)) ** (1.0 / 4.0))
            * cupy.sqrt(1.0 / (2.0 * cupy.pi * phi))
            * q_tilde
        )
        exponent = (
            (0.5 * q_tilde * (fft_frequency - phi))
            / phi
        ) ** 2

        wavelet = norm * cupy.exp(-exponent)

        # TODO: QUI VA AGGIUNTA LA COPIA SU SHARED MEMORY DEL BLOCCO DI OUTPLANE
        fft_phi_plane[phi_idx, fft_freq_idx] = fft_value * wavelet


def get_tau_phi_plane(
    signal_fft,
    fft_frequencies,
    P,
    Q,
    phi_range,
    data_segment_duration: float,
    data_sampling_rate: int,
):
    phi_tiles = get_phi_tiling(
        phi_range,
        Q,
        P,
        data_segment_duration,
        data_sampling_rate,
    )

    fft_phi_plane = cupy.zeros(
        (phi_tiles.shape[0], signal_fft.shape[0]),
        dtype=COMPLEX_PRECISION,
    )
    phi_tiles_GPU = cupy.array(phi_tiles)
    signal_fft_GPU = cupy.array(signal_fft)
    fft_frequencies_GPU = cupy.array(fft_frequencies)

    grid_shape = (
        fft_phi_plane.shape[0] // BLOCK_SHAPE[0] + 1,
        fft_phi_plane.shape[1] // BLOCK_SHAPE[1] + 1,
    )
    # here the qp transform is calculated
    fft_phi_plane_kernel[grid_shape, BLOCK_SHAPE](
        phi_tiles_GPU, signal_fft_GPU, P, Q, fft_frequencies_GPU, fft_phi_plane
    )
    tau_phi_plane = cufft.ifft(fft_phi_plane)
    norm_tau_phi_plane_GPU = tau_phi_plane * cupy.sqrt(data_sampling_rate)

    return norm_tau_phi_plane_GPU, phi_tiles_GPU


def get_phi_tiling(
    phi_range: list[Union[float, float]],
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
        / numpy.log(1.0 + (ALPHA * numpy.sqrt(1 + ((2 * P * Q) ** 2))) / Q),
    ).astype(INT_PRECISION)

    phi_tiles = numpy.geomspace(min_phi, max_phi, n_tiles)

    return phi_tiles
