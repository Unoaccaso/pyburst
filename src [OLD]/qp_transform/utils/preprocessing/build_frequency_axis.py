"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:36:53 am
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

# cpu
import numpy

# type hinting
from typing import Union

# errors and wanings
import warnings


def build_frequency_axis(
    phi_range: list[Union[numpy.float32, numpy.float32]],
    Q_values: list[Union[numpy.float32, numpy.float32]] | numpy.float32,
    p_values: list[Union[numpy.float32, numpy.float32]] | numpy.float32,
    time_series_duration: numpy.float32,
    data_sampling_rate: int = numpy.int32(
        CONFIG["signal.preprocessing"]["NewSamplingRate"]
    ),
    alpha: numpy.float32 = numpy.float32(CONFIG["computation.parameters"]["Alpha"]),
    thr_sigmas: int = 4,
):
    # perform sanity check on input to ensure safety limits:
    # Input range should be inside [0, Nyquist_freq] with thr_sigmas
    # of upper and lower space.

    if isinstance(Q_values, list):
        Q_min = Q_values[0]
        Q_max = Q_values[1]
    else:
        Q_min = Q_values
        Q_max = Q_values

    if isinstance(p_values, list):
        p_min = p_values[0]
    else:
        p_min = p_max = p_values

    assert Q_min >= 2 * numpy.pi, f"Q must be bigger then 2pi"
    # assert p_values[1] < 1 / p_values[1], f"p must be smaller then 1 / Q"
    assert p_min >= 0, f"p must be grater than 0"

    lowest_acceptable_phi = (1 / time_series_duration) * (
        1 - (thr_sigmas * numpy.sqrt(1 + ((2 * p_min * Q_max) ** 2))) / Q_max
    )
    highest_acceptable_phi = (data_sampling_rate / 2) * (
        1 + (thr_sigmas * numpy.sqrt(1 + ((2 * p_min * Q_max) ** 2))) / Q_max
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

    n_points = numpy.ceil(
        numpy.log(max_phi / min_phi)
        / numpy.log(1.0 + (alpha * numpy.sqrt(1 + ((2 * Q_max * p_min) ** 2))) / Q_max),
    ).astype(numpy.int32)

    # building an array of phi values
    phi_axis = numpy.geomspace(
        min_phi,
        max_phi,
        n_points,
        dtype=numpy.float32,
    )

    return phi_axis
