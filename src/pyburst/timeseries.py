"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, February 13th 2024, 8:15:05 pm
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

from dataclasses import dataclass, fields
import abc
import typing
import numpy
from numpy import float32, float64, int32, int64, complex64, complex128
import dask.array
import cupy
import warnings

import concurrent.futures

import gwpy.timeseries
import gwosc.datasets
import astropy

from ._typing import type_check


_ARRAY_LIKE = typing.Union[
    numpy.ndarray[float32],
    numpy.ndarray[float64],
    dask.array.Array,
    cupy.typing.NDArray[float32],
    cupy.typing.NDArray[float64],
]
_FLOAT_LIKE = typing.Union[
    float,
    float32,
    float64,
]
_INT_LIKE = typing.Union[
    int,
    int32,
    int64,
]


@dataclass
class BaseSeriesAttrs:
    """
    Attributes
    ----------
    segment_name : str
        Name of the segment.
    detector_id : str
        Identifier of the detector.
    detector_name : str
        Name of the detector.
    t0_gps : float
        GPS time of the starting point.
    duration : float
        Duration of the time series.
    dt : float
        Time resolution.
    sampling_rate : int
        Sampling rate.
    reference_time_gps : float
        Reference GPS time.
    """

    segment_name: str
    detector_id: str
    detector_name: str
    t0_gps: float
    duration: float
    dt: float
    sampling_rate: int
    reference_time_gps: float


class TimeSeries(abc.ABC):

    _DECODE_DETECTOR = {
        "L1": "Ligo Livingston (L1)",
        "H1": "Ligo Hanford (H1)",
        "V1": "Virgo (V1)",
    }

    _CACHED_DATA = {}

    @classmethod
    def _check_input_and_fill(
        cls,
        *args,
        **kwargs,
    ):

        new_kwargs = {}
        gps_time = kwargs["gps_time"]
        dt = kwargs["dt"]
        sampling_rate = kwargs["sampling_rate"]
        duration = kwargs["duration"]
        t0_gps = kwargs["t0_gps"]
        detector_id = kwargs["detector_id"]
        detector_name = kwargs["detector_name"]
        reference_time_gps = kwargs["reference_time_gps"]

        if detector_id not in cls._DECODE_DETECTOR or (
            detector_name is not None
            and detector_name != cls._DECODE_DETECTOR[detector_id]
        ):
            raise ValueError("Check detector ID and/or name")

        if gps_time is not None:
            _dt = gps_time[1] - gps_time[0]
            _sampling_rate = (1 / _dt).astype(int32)
            _t0_gps = gps_time[0]
            _duration = gps_time.max() - gps_time.min()
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
            new_kwargs["dt"] = _dt
            new_kwargs["sampling_rate"] = _sampling_rate
            new_kwargs["duration"] = _duration
            new_kwargs["t0_gps"] = _t0_gps

        else:
            if (
                (dt is None or sampling_rate is None)
                and t0_gps is None
                and duration is None
            ):
                raise ValueError(
                    "Provide a time array or the correct parameters to build one"
                )

            if dt is None:
                new_kwargs["dt"] = 1 / sampling_rate
                warnings.warn(
                    f"dt value was derived from sampling rate: {1/sampling_rate}"
                )

            elif sampling_rate is None:
                new_kwargs["sampling_rate"] = int(1 / dt)
                warnings.warn(f"sampling rate value was derived from dt: {int(1 / dt)}")
            elif int(1 / dt) != sampling_rate:
                raise ValueError(
                    f"dt: {dt} and sampling_rate: {sampling_rate} are not compatible!"
                )

        if detector_name is None:
            new_kwargs["detector_name"] = cls._DECODE_DETECTOR[detector_id]

        if reference_time_gps is None:
            new_kwargs["reference_time_gps"] = t0_gps + duration / 2
            warnings.warn(
                f"Reference time set to half interval: { t0_gps + duration / 2} s. Ensure this is fine."
            )

        elif reference_time_gps is not None and (
            reference_time_gps < new_kwargs["t0_gps"]
            or reference_time_gps > (new_kwargs["t0_gps"] + new_kwargs["duration"])
        ):
            warnings.warn(
                f"Reference time {reference_time_gps} is outside of time interval"
            )

        return new_kwargs

    @classmethod
    @type_check(classmethod=True)
    def from_array(
        cls,
        strain: _ARRAY_LIKE,
        segment_name: str,
        detector_id: str,
        gps_time: _ARRAY_LIKE | None = None,
        reference_time_gps: _FLOAT_LIKE = None,
        detector_name: str = None,
        dt: _FLOAT_LIKE = None,
        sampling_rate: _INT_LIKE = None,
        t0_gps: _FLOAT_LIKE = None,
        duration: _FLOAT_LIKE = None,
    ):
        """
        Construct a TimeSeries object from an array.

        Parameters
        ----------
        strain : numpy.ndarray[float32 | float64], dask.array.Array, cupy.ndarray[float32]
            The strain data. The returned TimeSeries object depends on the type of input strain:
            - If strain is a NumPy array, a ShortSeries object is returned.
            - If strain is a Dask array or a string, a LongSeries object is returned.
            - If strain is a CuPy array, a GPUSeries object is returned.
            The GPUSeries object utilizes GPU computing for array operations.
            The LongSeries object supports lazy computing with Dask arrays.

        gps_time : numpy.ndarray[float32 | float64], dask.array.Array, cupy.ndarray[float32], optional
            GPS time array (if available). It must have the same dtype of `strain`.
            If not provided, it will be calculated when accessed for the first time.
            Strain and GPS time can be sliced using time indices without immediate calculation of the array axes.

        segment_name : str
            Name of the segment.
        detector_id : str
            Identifier of the detector. Optional if `detector_name` is provided.
        reference_time_gps : Union[float, float32, float64], optional
            Reference GPS time.
        detector_name : str, optional
            Name of the detector. Required if `detector_id` is provided.
        dt : Union[float, float32, float64], optional
            Time resolution. If not provided, it will be calculated based on the sampling rate.
        sampling_rate : Union[int, int32, int64], optional
            Sampling rate. If not provided, it will be calculated based on the time resolution.
        t0_gps : Union[float, float32, float64], optional
            GPS time of the starting point. Required if `gps_time` is not provided.
        duration : Union[float, float32, float64], optional
            Duration of the time series. Required if `gps_time` is not provided.

        Returns
        -------
        TimeSeries
            Time series object.

        Raises
        ------
        ValueError
            If the input parameters are invalid.
        NotImplementedError
            If the strain type is not supported.

        Examples
        --------
        >>> strain = numpy.random.randn(1000).astype(numpy.float32)
        >>> t_series = TimeSeries.from_array(
        ...     strain=strain,
        ...     segment_name="Test Segment",
        ...     detector_id="H1",
        ...     dt=0.001,
        ...     t0_gps=1000000000.0,
        ...     duration=1.0,
        ... )
        """

        kwargs = {name: value for name, value in locals().items() if name != "cls"}

        new_kwargs = cls._check_input_and_fill(**kwargs)
        kwargs.update(new_kwargs)

        if isinstance(strain, numpy.ndarray):
            return ShortSeries(**kwargs)
        elif isinstance(strain, cupy.ndarray):
            return GPUSeries(**kwargs)
        elif isinstance(strain, dask.array.Array | str):
            return LongSeries(**kwargs)
        else:
            raise NotImplementedError(
                f"{type(strain)} type for strain is not supported."
            )

    @classmethod
    @type_check(classmethod=True)
    def from_gwpy(
        cls,
        gwpy_timeseries: gwpy.timeseries.TimeSeries,
        segment_name: str,
        detector_id: str,
        duration: _FLOAT_LIKE | None = None,
        detector_name: str | None = None,
        reference_time_gps: _FLOAT_LIKE | None = None,
        use_gpu: bool = False,
    ):

        if use_gpu:
            strain = cupy.array(gwpy_timeseries.value, dtype=float32)
            time_axis = cupy.array(gwpy_timeseries.times.value, dtype=float64)
        else:
            strain = numpy.array(gwpy_timeseries.value, dtype=float32)
            time_axis = numpy.array(gwpy_timeseries.times.value, dtype=float64)

        if detector_name is None:
            detector_name = cls._DECODE_DETECTOR[detector_id]
        if reference_time_gps is None:
            reference_time_gps = (time_axis[0] + time_axis[-1]) / 2
            warnings.warn(
                f"Reference time set to half interval: { reference_time_gps} s. Ensure this is fine."
            )

        return cls.from_array(
            strain,
            segment_name,
            detector_id,
            time_axis,
            reference_time_gps,
            detector_name,
            duration=duration,
        )

    @classmethod
    # @type_check(classmethod=True)
    def _fetch_remote(
        cls,
        event_name: str,
        detector_id: str,
        duration: _FLOAT_LIKE,
        sampling_rate: _INT_LIKE,
        repeat_on_falure: bool,
        max_attempts: _INT_LIKE,
        attempts_delay_s: _FLOAT_LIKE,
        current_attempt: _INT_LIKE,
        verbose: bool = False,
        use_gpu: bool = False,
    ):
        try:
            reference_time_gps = gwosc.datasets.event_gps(event_name)
            start_time = reference_time_gps - duration / 2
            end_time = (
                reference_time_gps + duration / 2 + 1 / sampling_rate
            )  # to inlcude last
            timeserie = gwpy.timeseries.TimeSeries.fetch_open_data(
                detector_id,
                start_time,
                end_time,
                sampling_rate,
                verbose=verbose,
            )
            new_duration = timeserie.times.value[-1] - timeserie.times.value[0]
            if new_duration != duration:
                duration = new_duration
                warnings.warn(f"Duration of downloaded data set to: {new_duration}")
            return cls.from_gwpy(
                timeserie,
                event_name,
                detector_id,
                reference_time_gps=reference_time_gps,
                use_gpu=use_gpu,
                duration=duration,
            )
        except ValueError:
            if current_attempt < max_attempts:
                warnings.warn(
                    f"Failed downloading {current_attempt}/{max_attempts} times, retrying...",
                )
                cls._fetch_remote(
                    event_name=event_name,
                    detector_id=detector_id,
                    duration=duration,
                    sampling_rate=sampling_rate,
                    max_attempts=max_attempts,
                    repeat_on_falure=repeat_on_falure,
                    attempts_delay_s=attempts_delay_s,
                    current_attempt=current_attempt + 1,
                    verbose=verbose,
                    use_gpu=use_gpu,
                )
            else:
                raise ConnectionError(
                    f"Failed downloading too many times ({current_attempt})"
                )

    @classmethod
    @type_check(classmethod=True)
    def fetch_open_data(
        cls,
        event_names: str | list[str],
        detector_ids: str | list[str],
        duration: _FLOAT_LIKE = 100.0,
        sampling_rate: _INT_LIKE = 4096,
        repeat_on_falure: bool = True,
        max_attempts: _INT_LIKE = 100,
        attempts_delay_s: _FLOAT_LIKE = 0.1,
        verbose: bool = False,
        use_gpu: bool = False,
        force_cache_overwrite: bool = False,
        cache_results: bool = True,
    ):
        if isinstance(event_names, str):
            event_names = [event_names]
        if isinstance(detector_ids, str):
            detector_ids = [detector_ids]
        if any(
            detector_id not in cls._DECODE_DETECTOR.keys()
            for detector_id in detector_ids
        ):
            raise NotImplementedError(f"Unsupported detector id!")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    cls._fetch_remote,
                    event_name,
                    detector_id,
                    duration,
                    sampling_rate,
                    repeat_on_falure,
                    max_attempts,
                    attempts_delay_s,
                    1,
                    verbose,
                    use_gpu,
                )
                for event_name in event_names
                for detector_id in detector_ids
                if event_name not in cls._CACHED_DATA
                or detector_id not in cls._CACHED_DATA.setdefault(event_name, {})
                or duration
                != cls._CACHED_DATA.setdefault(event_name, {})[detector_id].duration
                or force_cache_overwrite
            ]

            for future in futures:
                timeserie = future.result()
                cls._CACHED_DATA.setdefault(timeserie.segment_name, {})[
                    timeserie.detector_id
                ] = timeserie

        out_var = dict(cls._CACHED_DATA)
        if not cache_results:
            cls._CACHED_DATA.clear()

        return out_var

    def __getattribute__(self, attr: str):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return getattr(self._strain, attr)

    def __getitem__(self, key):
        return self._strain.__getitem__(key)

    def time_loc(self, start, end, fmt="gps"):
        if fmt != "gps":
            start = astropy.time.Time(start, format=fmt).gps
            end = astropy.time.Time(end, format=fmt).gps
        start_idx = int((start - self.t0_gps) // self.dt)
        end_idx = int((end - self.t0_gps) // self.dt)
        return self._strain[start_idx:end_idx]

    @abc.abstractproperty
    def strain(self): ...

    @abc.abstractproperty
    def time(self): ...


class LongSeries(TimeSeries, BaseSeriesAttrs):
    def __init__(
        self,
        strain: dask.array.Array | str,
        gps_time: dask.array.Array | None,
        *args,
        **kwargs,
    ):
        self._strain = strain
        self._gps_time = gps_time
        self._kwargs = kwargs
        super().__init__(*args, **kwargs)

    @property
    def strain(self):
        return self._strain

    @property
    def time(self):
        if self._gps_time is None:
            return dask.array.arange(
                self.t0_gps,
                self.t0_gps + self.duration,
                self.dt,
                dtype=float64,
            )
        else:
            return self._gps_time


class ShortSeries(TimeSeries, BaseSeriesAttrs):
    def __init__(
        self,
        strain: numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.float64],
        gps_time: numpy.ndarray[numpy.float32] | numpy.ndarray[numpy.float64] | None,
        *args,
        **kwargs,
    ):
        self._strain = strain
        self._gps_time = gps_time
        self._kwargs = kwargs
        super().__init__(*args, **kwargs)

    @property
    def strain(self):
        return self._strain

    @property
    def time(self):
        if self._gps_time is None:
            self._gps_time = numpy.arange(
                self.t0_gps,
                self.t0_gps + self.duration,
                self.dt,
                dtype=float64,
            )
        return self._gps_time


class GPUSeries(TimeSeries, BaseSeriesAttrs):
    def __init__(
        self,
        strain: cupy.typing.NDArray[numpy.float32],
        gps_time: cupy.typing.NDArray[numpy.float32] | None,
        *args,
        **kwargs,
    ):
        self._strain = strain
        self._gps_time = gps_time
        self._kwargs = kwargs
        super().__init__(*args, **kwargs)

    @property
    def strain(self):
        return self._strain

    @property
    def time(self):
        if self._gps_time is None:
            self._gps_time = cupy.arange(
                self.t0_gps,
                self.t0_gps + self.duration,
                self.dt,
                dtype=float64,
            )
        return self._gps_time