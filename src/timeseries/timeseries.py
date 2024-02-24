"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Thursday, February 22nd 2024, 2:58:17 pm
Author: unoaccaso

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
import concurrent

from .common._typing import type_check, _ARRAY_LIKE, _FLOAT_LIKE, _INT_LIKE, _FLOAT_EPS
from .core._cpu_series import _CPUSeries
from .core._caching import LRUDownloadCache

import numpy, cupy
import gwpy.timeseries, gwosc


class TimeSeries:
    _cache_size_mb = 1_000
    _DOWNLOAD_CACHE = LRUDownloadCache(max_size_mb=_cache_size_mb)

    @classmethod
    @type_check(classmethod=True)
    def set_cache_size_mb(cls, value: _FLOAT_LIKE | _INT_LIKE):
        cls._cache_size_mb = value
        warnings.warn("cache is being deleted!")
        cls._DOWNLOAD_CACHE = LRUDownloadCache(value)

    @classmethod
    @type_check(classmethod=True)
    def from_array(
        cls,
        values: _ARRAY_LIKE,
        segment_name: str = "-",
        detector_id: str = "-",
        gps_times: _ARRAY_LIKE | None = None,
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
        values : numpy.ndarray[float32 | float64], dask.array.Array, cupy.ndarray[float32]
            The values data. The returned TimeSeries object depends on the type of input values:
            - If values is a NumPy array, a ShortSeries object is returned.
            - If values is a Dask array or a string, a LongSeries object is returned.
            - If values is a CuPy array, a GPUSeries object is returned.
            The GPUSeries object utilizes GPU computing for array operations.
            The LongSeries object supports lazy computing with Dask arrays.

        gps_time : numpy.ndarray[float32 | float64], dask.array.Array, cupy.ndarray[float32], optional
            GPS time array (if available). It must have the same dtype of `values`.
            If not provided, it will be calculated when accessed for the first time.
            values and GPS time can be sliced using time indices without immediate calculation of the array axes.

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
            If the values type is not supported.

        Examples
        --------
        >>> values = numpy.random.randn(1000).astype(numpy.float32)
        >>> t_series = TimeSeries.from_array(
        ...     values=values,
        ...     segment_name="Test Segment",
        ...     detector_id="H1",
        ...     dt=0.001,
        ...     t0_gps=1000000000.0,
        ...     duration=1.0,
        ... )
        """

        kwargs = {name: value for name, value in locals().items() if name != "cls"}

        # * Checking time stuff
        if gps_times is None:
            if (
                t0_gps is None
                or duration is None
                or (dt is None and sampling_rate is None)
            ):
                raise ValueError(
                    f"Please provide time time data: gps_times or (t0_gps, duration, dt or sampling_rate)"
                )
            elif dt is None:
                kwargs["sampling_rate"] = numpy.int64(sampling_rate)
                kwargs["dt"] = numpy.float64(1 / sampling_rate)
                warnings.warn(f'dt set to {kwargs["dt"]}')
            elif sampling_rate is None:
                kwargs["dt"] = numpy.float64(dt)
                kwargs["sampling_rate"] = numpy.round(1 / dt).astype(numpy.int64)
                warnings.warn(f'sampling_rate set to {kwargs["sampling_rate"]}')

            warnings.warn(
                f"The time array will not include the last element: [start time, end time).\nThis is to avoid superprosition of data and repetition.\nSo the actual duration is {duration - kwargs['dt']}"
            )
            kwargs["duration"] = numpy.float64(duration) - kwargs["dt"]
            kwargs["t0_gps"] = numpy.float64(t0_gps)

            gps_times_shape = numpy.ceil(duration * kwargs["sampling_rate"] - 1).astype(
                numpy.int64
            )
            if len(values) != gps_times_shape:
                raise ValueError(
                    f"values have shape: {len(values)}, but input time data gives a gps_times array with shape {gps_times_shape}"
                )
        else:
            assert (
                gps_times.shape == values.shape
            ), f"Time array and value must have same shape."
            if t0_gps is not None:
                assert (
                    numpy.abs(t0_gps - gps_times[0]) < _FLOAT_EPS
                ), f"t0_gps: {t0_gps} is incompatible with gps_times[0]: {gps_times[0]}"
                kwargs["t0_gps"] = numpy.float64(t0_gps)
            else:
                kwargs["t0_gps"] = numpy.float64(gps_times[0])
                warnings.warn(f't0_gps set to {kwargs["t0_gps"]}')

            if duration is not None:
                assert (
                    numpy.abs(duration - (gps_times[-1] - gps_times[0])) < _FLOAT_EPS
                ), f"duration: {duration} is incompatible with gps_times duration: {gps_times[-1] - gps_times[0]}"
                kwargs["duration"] = numpy.float64(duration)
            else:
                kwargs["duration"] = numpy.float64(gps_times[-1] - gps_times[0])
                warnings.warn(f'duration set to {kwargs["duration"]}')

            if dt is not None:
                assert (
                    numpy.abs(dt - (gps_times[1] - gps_times[0])) < _FLOAT_EPS
                ), f"dt: {dt} is incompatible with gps_times dt: {gps_times[1] - gps_times[0]}"
                kwargs["dt"] = numpy.float64(dt)
            else:
                kwargs["dt"] = numpy.float64(gps_times[1] - gps_times[0])
                warnings.warn(f'dt set to {kwargs["dt"]}')

            if sampling_rate is not None:
                assert (
                    numpy.abs(
                        sampling_rate - numpy.round(1 / (gps_times[1] - gps_times[0]))
                    )
                    < _FLOAT_EPS
                ), f"sampling_rate: {sampling_rate} is incompatible with gps_times sampling rate: {numpy.round(1 / (gps_times[1] - gps_times[0]))}"
                kwargs["sampling_rate"] = numpy.int64(sampling_rate)
            else:
                kwargs["sampling_rate"] = numpy.round(
                    1 / (gps_times[1] - gps_times[0])
                ).astype(numpy.int64)
                warnings.warn(f'sampling_rate set to {kwargs["sampling_rate"]}')

        if numpy.log2(kwargs["sampling_rate"]) % 1 != 0:
            warnings.warn(
                f"sampling rate should be a power of 2. Other values are allowed but can result in unexpected behaviour."
            )

        if isinstance(values, numpy.ndarray):
            return _CPUSeries(**kwargs)
        if isinstance(values, cupy.ndarray):
            return
        else:
            raise NotImplementedError(
                f"{type(values)} type for values is not supported."
            )

    # TODO: DOCSTRINGS!!!
    @classmethod
    @type_check(classmethod=True)
    def from_gwpy(
        cls,
        gwpy_timeseries: gwpy.timeseries.TimeSeries,
        segment_name: str,
        detector_id: str,
        duration: _FLOAT_LIKE | None = None,
        reference_time_gps: _FLOAT_LIKE | None = None,
        use_gpu: bool = False,
    ):

        if use_gpu:
            values = cupy.array(gwpy_timeseries.value, dtype=numpy.float32)
            time_axis = cupy.array(gwpy_timeseries.times.value, dtype=numpy.float64)
        else:
            values = numpy.array(gwpy_timeseries.value, dtype=numpy.float64)
            time_axis = numpy.array(gwpy_timeseries.times.value, dtype=numpy.float64)

        return cls.from_array(
            values,
            segment_name,
            detector_id,
            time_axis,
            numpy.float64(reference_time_gps),
            duration=numpy.float64(duration),
        )

    @classmethod
    @type_check(classmethod=True)
    def _fetch_remote(
        cls,
        event_name: str,
        detector_id: str,
        duration: _FLOAT_LIKE,
        sampling_rate: _INT_LIKE,
        repeat_on_falure: bool,
        max_attempts: _INT_LIKE,
        current_attempt: _INT_LIKE,
        verbose: bool = False,
        use_gpu: bool = False,
    ):
        try:
            if verbose:
                print(f"Connecting to gwosc for {event_name}({detector_id})...")
            reference_time_gps = gwosc.datasets.event_gps(event_name)
            if verbose:
                print("done!")
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
                    current_attempt=current_attempt + 1,
                    verbose=verbose,
                    use_gpu=use_gpu,
                )
            else:
                raise ConnectionError(
                    f"Failed downloading too many times ({current_attempt})"
                )

    # TODO
    @classmethod
    @type_check(classmethod=True)
    def fetch_event(
        cls,
        event_names: str | list[str],
        detector_ids: str | list[str],
        duration: _FLOAT_LIKE = 100.0,
        sampling_rate: _INT_LIKE = 4096,
        repeat_on_falure: bool = True,
        max_attempts: _INT_LIKE = 100,
        verbose: bool = False,
        use_gpu: bool = False,
        force_cache_overwrite: bool = False,
        cache_results: bool = True,
    ):
        if isinstance(event_names, str):
            event_names = [event_names]
        if isinstance(detector_ids, str):
            detector_ids = [detector_ids]
        # if any(
        #     detector_id not in cls._DECODE_DETECTOR.keys()
        #     for detector_id in detector_ids
        # ):
        #     raise NotImplementedError(f"Unsupported detector id!")
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
                    1,
                    verbose,
                    use_gpu,
                )
                for event_name in event_names
                for detector_id in detector_ids
                if (event_name, detector_id) not in cls._DOWNLOAD_CACHE
                or duration != cls._DOWNLOAD_CACHE[(event_name, detector_id)].duration
                or force_cache_overwrite
                or (
                    use_gpu
                    and not isinstance(
                        cls._DOWNLOAD_CACHE[(event_name, detector_id)],
                        cupy.ndarray,
                    )
                )
            ]
            out_var = {}
            for future in futures:
                timeserie = future.result()
                event_name = timeserie.segment_name
                detector_id = timeserie.detector_id
                out_var[(event_name, detector_id)] = timeserie

        if cache_results:
            cls._DOWNLOAD_CACHE.update(out_var)

        return out_var

    # TODO
    @classmethod
    @type_check(classmethod=True)
    def fetch_open_data(cls): ...

    # TODO
    @classmethod
    @type_check(classmethod=True)
    def from_zarr(cls): ...

    # TODO
    @classmethod
    @type_check(classmethod=True)
    def save(cls): ...
