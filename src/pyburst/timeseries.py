"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Monday, February 12th 2024, 10:08:37 am
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

import xarray
import gwpy
import astropy

import numpy
from numpy import float32, float64, int32, int64, complex64, complex128

from ._typing import type_check


class TimeSeries:

    @classmethod
    @type_check(classmethod=True)
    def _input_check(
        cls,
        times: numpy.ndarray[float32] | numpy.ndarray[float64] | None,
        dt: float | float32 | float64 | None,
        sampling_rate: int | int32 | int64 | None,
        t0_gps: float | float32 | float64 | None,
        duration: float | float32 | float64 | None,
    ):
        if not (times is not None or ((dt or sampling_rate) and t0_gps and duration)):
            raise ValueError(
                f"provide a time array or the correct parameters to build one"
            )

    @classmethod
    @type_check(classmethod=True)
    def _compute_missing_params(
        cls,
        times: numpy.ndarray[float32] | numpy.ndarray[float64] | None,
        dt: float | float32 | float64 | None,
        sampling_rate: int | int32 | int64 | None,
        t0_gps: float | float32 | float64 | None,
        duration: float | float32 | float64 | None,
    ):

        if times is not None:
            _dt = times[1] - times[0]
            _sampling_rate = int32(1 / _dt)
            if dt is not None:
                assert (
                    _dt == dt
                ), f"time data has a dt of {_dt} s, input dt is {dt} s. Please correct!"
            if sampling_rate is not None:
                assert (
                    _sampling_rate == sampling_rate
                ), f"time data has a sampling rate of {_sampling_rate} s, input sampling rate is {sampling_rate} s. Please correct!"
        if times is None:
            t_start = t0_gps
            t_stop = t0_gps + duration
            times = numpy.arange(t_start, t_stop, dt)

        return (times, dt, sampling_rate, t0_gps, duration)

    @classmethod
    @type_check(classmethod=True)
    def build(
        cls,
        strain: numpy.ndarray[float32] | numpy.ndarray[float64],
        name: str | list[str],
        detector_id: str | list[str],
        detector_name: str | None = None,
        times: numpy.ndarray[float32] | numpy.ndarray[float64] | None = None,
        dt: float | float32 | float64 | None = None,
        sampling_rate: int | int32 | int64 | None = None,
        t0_gps: float | float32 | float64 | None = None,
        duration: float | float32 | float64 | None = None,
    ):
        input_time_params = (times, dt, sampling_rate, t0_gps, duration)
        cls._input_check(*input_time_params)
        (times, dt, sampling_rate, t0_gps, duration) = cls._compute_missing_params(
            *input_time_params
        )
