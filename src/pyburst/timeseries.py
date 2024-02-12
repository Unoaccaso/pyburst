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

import warnings

import xarray
import gwpy
import astropy
import pandas

import numpy
from numpy import float32, float64, int32, int64, complex64, complex128


import cupy

from ._typing import type_check


class TimeSeries:

    _DECODE_DETECTOR = {
        "L1": "Ligo Livingston (L1)",
        "H1": "Ligo Hanford (H1)",
        "V1": "Virgo (V1)",
    }

    @classmethod
    @type_check(classmethod=True)
    def _input_check(
        cls,
        gps_times: numpy.ndarray[float32] | numpy.ndarray[float64] | None,
        dt: float | float32 | float64 | None,
        sampling_rate: int | int32 | int64 | None,
        t0_gps: float64 | None,
        duration: float | float32 | float64 | None,
        detector_id: str,
        detector_name: str | None,
        reference_gps_time: float | float32 | float64 | None,
    ):
        if not (
            gps_times is not None or ((dt or sampling_rate) and t0_gps and duration)
        ):
            raise ValueError(
                f"Provide a time array or the correct parameters to build one"
            )
        if detector_id not in cls._DECODE_DETECTOR or (
            detector_name is not None
            and detector_name != cls._DECODE_DETECTOR[detector_id]
        ):
            raise ValueError(f"Check detector ID and/or name")

        if reference_gps_time is not None and (
            reference_gps_time < t0_gps or reference_gps_time > (t0_gps + duration)
        ):
            warnings.warn(
                f"Reference time {reference_gps_time} is outside of time interval"
            )

    @classmethod
    @type_check(classmethod=True)
    def _fill_missing_params(
        cls,
        gps_times: numpy.ndarray[float32] | numpy.ndarray[float64] | None,
        dt: float | float32 | float64 | None,
        sampling_rate: int | int32 | int64 | None,
        t0_gps: float64 | None,
        duration: float | float32 | float64 | None,
        detector_id: str,
        detector_name: str | None,
        reference_gps_time: float | float32 | float64 | None,
    ):

        if gps_times is not None:
            _dt = gps_times[1] - gps_times[0]
            _sampling_rate = int32(1 / _dt)
            _t0_gps = gps_times[0]
            _duration = gps_times.max() - gps_times.min()
            if dt is not None and _dt != dt:
                raise ValueError(
                    f"time data has a dt of {_dt} s, input dt is {dt} s. Please correct!"
                )
            if sampling_rate is not None and _sampling_rate != sampling_rate:
                raise ValueError(
                    f"time data has a sampling rate of {_sampling_rate} s, input sampling rate is {sampling_rate} s. Please correct!"
                )
            if duration is not None and _duration != duration:
                raise ValueError(
                    f"time data ha a duration of {_duration} s, input duration is {duration}. Please correct!"
                )
            if t0_gps is not None and _t0_gps != t0_gps:
                raise ValueError(
                    f"time data ha a t0 of {_t0_gps} s, input t0_gps is {t0_gps}. Please correct!"
                )
            dt = _dt
            sampling_rate = _sampling_rate
            duration = _duration
            t0_gps = _t0_gps
        else:
            t_start = t0_gps
            t_stop = t0_gps + duration
            gps_times = numpy.arange(t_start, t_stop, dt)

        if detector_name is None:
            detector_name = cls._DECODE_DETECTOR[detector_id]

        if reference_gps_time is None:
            reference_gps_time = t0_gps + duration / 2
            warnings.warn(
                f"Reference time set to half interval: {reference_gps_time} s. Ensure this is fine."
            )

        return (
            gps_times,
            dt,
            sampling_rate,
            t0_gps,
            duration,
            detector_name,
            reference_gps_time,
        )

    @classmethod
    @type_check(classmethod=True)
    def _compute_time_axis(
        cls, gps_times: numpy.ndarray[float32] | numpy.ndarray[float64]
    ):
        ap_time_obj = astropy.time.Time(gps_times, format="gps")
        ap_time_obj.format = "iso"
        datetime = ap_time_obj.to_datetime()
        multi_time_axis = pandas.MultiIndex.from_arrays(
            [gps_times, datetime], names=("gps", "iso")
        )
        return multi_time_axis

    @classmethod
    @type_check(classmethod=True)
    def build(
        cls,
        strain: numpy.ndarray[float32] | numpy.ndarray[float64],
        name: str,
        detector_id: str,
        reference_gps_time: float32 | float64 | None = None,
        detector_name: str | None = None,
        gps_times: numpy.ndarray[float32] | numpy.ndarray[float64] | None = None,
        dt: float | float32 | float64 | None = None,
        sampling_rate: int | int32 | int64 | None = None,
        t0_gps: float64 | None = None,
        duration: float | float32 | float64 | None = None,
        use_gpu: bool = True,
    ):
        input_params = (
            gps_times,
            dt,
            sampling_rate,
            t0_gps,
            duration,
            detector_id,
            detector_name,
            reference_gps_time,
        )

        # cheking input parameters and filling missing ones
        cls._input_check(*input_params)
        (
            gps_times,
            dt,
            sampling_rate,
            t0_gps,
            duration,
            detector_name,
            reference_gps_time,
        ) = cls._fill_missing_params(*input_params)

        # building time multi-axis
        time_multi_axis = cls._compute_time_axis(gps_times)

        # creating the custom xarray
        array_gps_timeseries = xarray.DataArray(
            strain,
            name="strain",
            coords=dict(time=time_multi_axis),
            attrs=dict(
                segment_name=name,
                detector_id=detector_id,
                detector_name=detector_name,
                t0_gps=t0_gps,
                reference_gps_time=reference_gps_time,
                duration=duration,
                dt=dt,
                sampling_rate=sampling_rate,
                white=False,
            ),
        )

        if use_gpu:
            array_gps_timeseries.timeseries.copy_on_gpu()

        return array_gps_timeseries


@xarray.register_dataarray_accessor("timeseries")
class TimeSeriesAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._strain_gpu = None
        self._time_gpu = None

    @type_check(classmethod=True)
    def _update_gpu_arr(
        self, strain: numpy.ndarray[float32], time: numpy.ndarray[float32]
    ):
        self._strain_gpu = cupy.array(strain, dtype=float32)
        self._time_gpu = cupy.array(time, dtype=float32)
        warnings.warn(f"Values on GPU updated")

    def copy_on_gpu(self):
        # TODO: non sono ancora sicuro che questa sia la cosa migliore
        if self._obj.values.dtype != float32 or self._obj.gps.values.dtype != float32:
            warnings.warn(f"Data will be cast to float32 before copying to GPU!")
        self._strain_gpu = cupy.array(self._obj.values, dtype=float32)
        self._time_gpu = cupy.array(self._obj.gps.values, dtype=float32)
        warnings.warn(f"Array content copied on GPU!")

    @property
    def strain_gpu(self):
        return self._strain_gpu

    @property
    def time_gpu(self):
        return self._time_gpu

    @property
    def dt(self):
        t0 = self._obj.gps.values[0]
        t1 = self._obj.gps.values[1]
        _dt = t1 - t0
        # Me being paraniod
        assert (
            _dt == self._obj.attrs["dt"]
        ), "Data dt rate not matching the value in attributes!"
        return _dt

    @property
    def sampling_rate(self):
        _sampling_rate = int32(1 / self.dt)
        # Me being paranoid
        assert (
            _sampling_rate == self._obj.attrs["sampling_rate"]
        ), "Data sampling rate not matching the value in attributes!"
        return _sampling_rate

    def whiten(self): ...

    def crop(self): ...
