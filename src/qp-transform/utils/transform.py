""""""
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


# WARNING: AT THE TIME OF WRITING, JIT.RAWKERNEL IS EXPERIMENTAL!!!
@jit.rawkernel()
def fft_phi_plane_kernel(
    phi_values,
    fft_values,
    fft_frequencies,
    sampling_rate,
    P: FLOAT_PRECISION,
    Q: FLOAT_PRECISION,
    n_rows: INT_PRECISION,
    n_cols: INT_PRECISION,
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


def get_tau_phi_plane(
    signal_fft,
    fft_frequencies,
    P: FLOAT_PRECISION,
    Q: FLOAT_PRECISION,
    phi_range: list[Union[float, float]],
    data_segment_duration: FLOAT_PRECISION,
    data_sampling_rate: INT_PRECISION,
    alpha: FLOAT_PRECISION = CONFIG["signal.parameters"]["alpha"],
):
    #
    #
    # computing the tiling of phi
    phi_tiles = get_phi_tiles(
        phi_range,
        alpha,
        Q,
        P,
        data_segment_duration,
        data_sampling_rate,
    )

    # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
    # TODO: la qp transform su un dataset di dimensione grande a piacere.

    # preallocating the fft-phi plane.
    fft_phi_plane = cupy.zeros(
        (phi_tiles.shape[0], signal_fft.shape[0]),
        dtype=COMPLEX_PRECISION,
    )

    # moving data to GPU for computation
    phi_tiles_GPU = cupy.array(phi_tiles, dtype=FLOAT_PRECISION)
    signal_fft_GPU = cupy.array(signal_fft, dtype=COMPLEX_PRECISION)
    fft_frequencies_GPU = cupy.array(fft_frequencies, dtype=FLOAT_PRECISION)

    # instatiating cuda variables
    grid_shape = (
        fft_phi_plane.shape[1] // BLOCK_SHAPE[1] + 1,  # X
        fft_phi_plane.shape[0] // BLOCK_SHAPE[0] + 1,  # y
    )
    block_shape = BLOCK_SHAPE

    # here the qp transform is calculated
    fft_phi_plane_kernel[grid_shape, block_shape](
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

    # here the inverse fft is computed and the phi-tau plane is returned
    norm_tau_phi_plane_GPU = cufft.ifft(fft_phi_plane)

    return norm_tau_phi_plane_GPU, phi_tiles


def get_phi_tiles(
    phi_range: list[Union[float, float]],
    alpha: FLOAT_PRECISION,
    Q: FLOAT_PRECISION,
    P: FLOAT_PRECISION,
    data_segment_duration: float,
    data_sampling_rate: int = INT_PRECISION(
        CONFIG["signal.parameters"]["SamplingRate"]
    ),
    thr_sigmas: int = 4,
):
    """Frequencies for the transform

    Build an array containing all the frequencies over wich the
    transform will be computed.

    Example
    -------
    [To be done]::

        $ an example.py

    Another section

    Notes
    -----
        Prova

    Arguments
    ---------
        phi_range : list[Union[float, float]]
            Range of frequencies. This will be adjusted if needed to match the
            condition: :math:`\phi \in [0, f_N]`, where :math:`f_N` is the
            Nyquist frequency.
        alpha : float
            Used to compute the optimal distance between frequencies,
            to maximize the statistics.
        Q : float
            Q parameter of the Qp-Transform.
        P : float
            P parameter of the Qp-Transform
        data_segment_duration : float
            Time duration of the data segment over wich the transform will
            take place. Used to compute the minimum acceptable frequency.
        data_sampling_rate : :obj:`int`, optional
            Sampling rate of the data to be transformed. Defaults to the
            value in the configuration file.
        thr_sigmas : :obj:`int`, optional
            The minimum number of sigmas of the wavelets on the border of
            the plane, that have to be included in the analysis. Defaults to 4.

    Returns
    -------
        numpy.NDArray
            1D array of frequencies.
    """
    # perform sanity check on input to ensure safety limits:
    # Input range should be inside [0, Nyquist_freq] with thr_sigmas
    # of upper and lower space.
    lowest_acceptable_phi = (1 / data_segment_duration) * (
        1 - (thr_sigmas * numpy.sqrt(1 + ((2 * P * Q) ** 2))) / Q
    )
    highest_acceptable_phi = (data_sampling_rate / 2) * (
        1 + (thr_sigmas * numpy.sqrt(1 + ((2 * P * Q) ** 2))) / Q
    )
    # Adjusting phi range
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

    # building an array of phi values
    phi_tiles = numpy.geomspace(
        min_phi,
        max_phi,
        n_tiles,
        dtype=FLOAT_PRECISION,
    )

    return phi_tiles
