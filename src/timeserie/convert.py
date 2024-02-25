"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Sunday, February 25th 2024, 9:46:01 pm
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

from timeserie.common._typing import (
    type_check,
    _ARRAY_LIKE,
    _FLOAT_LIKE,
    _INT_LIKE,
    _FLOAT_EPS,
)
from timeserie.core.cpu import CPUSerie
from timeserie import CACHE

import warnings
import numpy
import gwpy.timeseries
import cupy


@type_check(classmethod=False)
def from_array(
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
    force_cache_overwrite: bool = True,
    cache_results: bool = True,
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
        if t0_gps is None or duration is None or (dt is None and sampling_rate is None):
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
        if numpy.prod(values.shape) != gps_times_shape:
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
    non_series_attr = ["cache_results", "force_cache_overwrite"]
    for attr in non_series_attr:
        kwargs.pop(attr)

    if isinstance(values, numpy.ndarray):
        out_series = CPUSerie(**kwargs)
    elif isinstance(values, cupy.ndarray):
        return
    else:
        raise NotImplementedError(f"{type(values)} type for values is not supported.")

    if cache_results:
        if (
            segment_name,
            detector_id,
        ) not in CACHE or force_cache_overwrite:
            CACHE[(segment_name, detector_id)] = out_series

    return out_series


@type_check(classmethod=False)
def from_gwpy(
    gwpy_timeseries: gwpy.timeseries.TimeSeries,
    segment_name: str,
    detector_id: str,
    duration: _FLOAT_LIKE | None = None,
    reference_time_gps: _FLOAT_LIKE | None = None,
    use_gpu: bool = False,
):
    """
    Create a TimeSeries object from a gwpy.timeseries.TimeSeries object.

    This method converts a `gwpy.timeseries.TimeSeries` object into a `TimeSeries` object,
    which is part of the current library. It allows for seamless integration of GWPy data
    with other data processing tools provided in this library.

    Parameters
    ----------
    gwpy_timeseries : gwpy.timeseries.TimeSeries
        The gwpy TimeSeries object to be converted.
    segment_name : str
        Name of the segment.
    detector_id : str
        Identifier of the detector.
    duration : Union[float, float32, float64], optional
        Duration of the time series. If not provided, it will be inferred from the gwpy Timeseries.
    reference_time_gps : Union[float, float32, float64], optional
        Reference GPS time. If provided, it sets the GPS time of the starting point for the time series.
        If not provided, it will be inferred from the first time stamp of the gwpy Timeseries.
    use_gpu : bool, optional
        If True, utilize GPU for processing.

    Returns
    -------
    TimeSeries
        Time series object.

    Raises
    ------
    NotImplementedError
        If conversion is not supported for the given input type.

    Examples
    --------
    Convert a gwpy TimeSeries object into a TimeSeries object:

    >>> import gwpy.timeseries as gw
    >>> from mylibrary import TimeSeries
    >>> # Assuming 'gwpy_timeseries' is a gwpy.timeseries.TimeSeries object
    >>> # with some data loaded.
    >>> ts = TimeSeries.from_gwpy(
    ...     gwpy_timeseries=gwpy_timeseries,
    ...     segment_name="GWPy Data",
    ...     detector_id="H1",
    ... )

    Convert a gwpy TimeSeries object into a TimeSeries object with a specified duration and reference time:

    >>> ts = TimeSeries.from_gwpy(
    ...     gwpy_timeseries=gwpy_timeseries,
    ...     segment_name="GWPy Data",
    ...     detector_id="H1",
    ...     duration=10.0,  # Duration set to 10 seconds
    ...     reference_time_gps=123456789.0,  # Reference GPS time set to a specific value
    ... )
    """

    if use_gpu:
        values = cupy.array(gwpy_timeseries.value, dtype=numpy.float32)
        time_axis = cupy.array(gwpy_timeseries.times.value, dtype=numpy.float64)
    else:
        values = numpy.array(gwpy_timeseries.value, dtype=numpy.float64)
        time_axis = numpy.array(gwpy_timeseries.times.value, dtype=numpy.float64)

    return from_array(
        values,
        segment_name,
        detector_id,
        time_axis,
        numpy.float64(reference_time_gps),
        duration=numpy.float64(duration),
    )
