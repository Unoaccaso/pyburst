"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Saturday, January 27th 2024, 10:53:52 pm
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


import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# settings
from utils.commons import CONFIG
import cupy
import cupyx.scipy.special
from scipy import special
import cupy.fft as cufft
import numpy

import matplotlib.pyplot as plt

from .find_boudaries import find_boundaries


def filter(
    signal_fft,
    fft_freqs,
    phi_axis,
    time_axis,
    Q,
    p,
    energy_plane,
    energy_threshold,
):
    Q_tilde = (Q / numpy.sqrt(1 + 1.0j * 2 * p * Q)).astype(numpy.complex64)

    contour_GPU = find_boundaries(
        energy_plane,
        energy_threshold,
    )

    contour_coords = cupy.argwhere(contour_GPU == 1)

    if len(contour_coords) > 0:
        ordered_contour_coords = contour_coords[contour_coords[:, 1].argsort()]

        start_frequency_idx = ordered_contour_coords[::2, 0]
        start_frequency = phi_axis[start_frequency_idx]
        end_frequency_idx = ordered_contour_coords[1::2, 0]
        end_frequency = phi_axis[end_frequency_idx]
        time_idx = ordered_contour_coords[::2, 1]
        unique_time_idx = cupy.unique(ordered_contour_coords[:, 1])

        if p == 0:
            Q_tilde = Q
            alpha = (fft_freqs / start_frequency[:, None]).astype(numpy.float32)
            beta = (fft_freqs / end_frequency[:, None]).astype(numpy.float32)
            windows_gpu = (
                cupyx.scipy.special.erf(Q_tilde / 2 * (alpha - 1), dtype=numpy.float32)
                - cupyx.scipy.special.erf(Q_tilde / 2 * (beta - 1), dtype=numpy.float32)
            ) / cupy.real(special.erf(Q_tilde / 2)).astype(numpy.complex64)
        else:
            Q_tilde = (Q / numpy.sqrt(1 + 1.0j * 2 * p * Q)).astype(numpy.complex64)
            alpha = (fft_freqs / start_frequency[:, None]).get().astype(numpy.float32)
            beta = (fft_freqs / end_frequency[:, None]).get().astype(numpy.float32)
            windows = (
                special.erf(Q_tilde / 2 * (alpha - 1), dtype=numpy.complex64)
                - special.erf(Q_tilde / 2 * (beta - 1), dtype=numpy.complex64)
            ) / numpy.real(special.erf(Q_tilde / 2)).astype(numpy.complex64)
            windows_gpu = cupy.array(windows)

        # unisco le windows corrispondenti a tempi uguali
        summed_windows = cupy.einsum(
            "ik, kj -> ij", (time_idx == unique_time_idx[:, None]), windows_gpu
        )

        filtered_ffts = signal_fft * summed_windows

        partial_filtered_signal = cupy.real(cufft.ifft(filtered_ffts))
        filtered_signal = cupy.zeros_like(time_axis)
        time_axis_idx = cupy.arange(0, len(time_axis))
        filtered_signal[unique_time_idx] = partial_filtered_signal[
            unique_time_idx[:, None] == time_axis_idx
        ]

        return filtered_signal

    else:
        return cupy.zeros_like(time_axis)
