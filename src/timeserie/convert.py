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

from timeserie.common import _typing
from timeserie.common._typing import type_check, _FLOAT_EPS
from timeserie.core import BaseSeriesAttrs, Detectors
from timeserie.core.cpu import CPUSerie
from timeserie import CACHE

import warnings
import numpy
import gwpy.timeseries
import cupy


@type_check(classmethod=False)
def from_array(
    strain: _typing.ARRAY_LIKE,
    segment_name: str = "_",
    detector_id: str = "_",
    gps_times: _typing.ARRAY_LIKE = None,
    dt: _typing.FLOAT_LIKE = None,
    sampling_rate: _typing.INT_LIKE = None,
    t0_gps: _typing.FLOAT_LIKE = None,
    duration: _typing.FLOAT_LIKE = None,
    force_cache_overwrite: bool = True,
    cache_results: bool = True,
    use_gpu: bool = False,
):
    """
    Create TimeSeries object from an array of strain data.

    This function creates a TimeSeries object from an array of strain data. It allows
    for specifying various attributes such as segment name, detector ID, GPS times, etc.
    It also handles caching of the created TimeSeries object for improved performance.

    Parameters
    ----------
    strain : array_like
        Array of strain data.
    segment_name : str, optional
        Name of the segment. Default is "_".
    detector_id : str, optional
        ID of the detector. Default is "_".
    gps_times : array_like, optional
        Array of GPS times. Default is None. [OPTIONAL: If provided, dt, sampling_rate, t0_gps, and duration will be ignored]
    dt : Union[float, float32, float64], optional
        Time interval between samples. Default is None. [OPTIONAL: If sampling_rate is provided, Ignored if gps_time is provided]
    sampling_rate : Union[int, int32, int64], optional
        Sampling rate of the strain data. Default is None. [OPTIONAL: If dt is provided, Ignored if gps_time is provided]
    t0_gps : Union[float, float32, float64], optional
        GPS start time of the data. Default is None. [OPTIONAL: Ignored if gps_times is provided]
    duration : Union[float, float32, float64], optional
        Duration of the data. Default is None. [OPTIONAL: Ignored if gps_times is provided]
    force_cache_overwrite : bool, optional
        Whether to force overwrite cached data. Default is True.
    cache_results : bool, optional
        Whether to cache the created TimeSeries object. Default is True.
    use_gpu : bool, optional
        If True, utilize GPU for processing. Default is False.

    Returns
    -------
    TimeSeries
        TimeSeries object created from the input strain data.

    Raises
    ------
    ValueError
        If required parameters are not provided or if shape checks fail.

    Notes
    -----
    - This function supports creating TimeSeries objects both on CPU.
    - The function handles various integrity checks including:
        - **Shape Checks**: Ensures that the shape of input data matches specified parameters.
        - **Attribute Validations**: Validates input attributes such as GPS times, duration, etc.
        - **Compatibility Checks**: Checks compatibility between dt and sampling rate if both are provided.
        - **Sparse Array Check**: Checks if the input array is sparse (not supported yet).
    - It supports caching of the created TimeSeries object for future use.

    Examples
    --------
    Create a TimeSeries object from strain data with specified sampling rate and duration:

    >>> strain_data = [0.1, 0.2, 0.3, 0.4]
    >>> ts = from_array(
    ...     strain=strain_data,
    ...     segment_name="Segment1",
    ...     detector_id="H1",
    ...     sampling_rate=100,
    ...     duration=2.0
    ... )

    Create a TimeSeries object from strain data with specified GPS times:

    >>> strain_data = [0.1, 0.2, 0.3, 0.4]
    >>> gps_times = [1123456789.0, 1123456790.0, 1123456791.0, 1123456792.0]
    >>> ts = from_array(
    ...     strain=strain_data,
    ...     segment_name="Segment1",
    ...     detector_id="H1",
    ...     gps_times=gps_times
    ... )

    """
    # Integrity checks
    if gps_times is None:
        if dt is None and sampling_rate:
            dt = numpy.float64(1 / sampling_rate)
        elif sampling_rate is None and dt:
            sampling_rate = numpy.int64(1 / dt)
        else:
            raise ValueError(
                f"If no time array is given, you must provide either dt or sampling rate!"
            )
        if t0_gps is None or duration is None:
            raise ValueError(
                f"If no time array is given, you must provide t0 and duration!"
            )
        # checking shape
        gps_times_shape = numpy.ceil(duration * sampling_rate - 1).astype(numpy.int64)
        if strain.shape != gps_times_shape:
            raise ValueError(
                f"strain have shape: {len(strain)}, but input time data gives a gps_times array with shape {gps_times_shape}"
            )
    else:
        # checking shape
        if not (gps_times.shape == strain.shape):
            raise ValueError(f"Time array and value must have same shape.")
        # checking if sparse
        dts = numpy.unique(numpy.diff(gps_times))
        if not dts.shape == (1,):
            raise NotImplementedError(f"Sparse arrays are not supported yet")

        if t0_gps:
            warnings.warn(f"t0_gps inserted ({t0_gps}) will be ignored")
        if dt:
            warnings.warn(f"dt inserted ({dt}) will be ignored")
        if sampling_rate:
            warnings.warn(f"sampling_rate inserted ({sampling_rate}) will be ignored")
        if duration:
            warnings.warn(f"duration inserted ({duration}) will be ignored")

        # Extracting attrs
        t0_gps = gps_times[0]
        dt = dts[0]
        sampling_rate = numpy.int64(1 / dt)
        duration = gps_times[-1] - t0_gps + dt

    if not isinstance(dt, numpy.float64):
        warnings.warn(
            f"Converting dt to numpy.float64\nPlease ensure that values conincide: {dt} > {numpy.float64(dt)}"
        )
        dt = numpy.float64(dt)
    if not isinstance(sampling_rate, numpy.int64):
        warnings.warn(
            f"Converting sampling_rate to numpy.int64\nPlease ensure that values conincide: {sampling_rate} > {numpy.int64(sampling_rate)}"
        )
        sampling_rate = numpy.int64(sampling_rate)
    if not isinstance(duration, numpy.float64):
        warnings.warn(
            f"Converting duration to numpy.float64\nPlease ensure that values conincide: {duration} > {numpy.float64(duration)}"
        )
        duration = numpy.float64(duration)
    if not isinstance(t0_gps, numpy.float64):
        warnings.warn(
            f"Converting t0_gps to numpy.float64\nPlease ensure that values conincide: {t0_gps} > {numpy.float64(t0_gps)}"
        )
        t0_gps = numpy.float64(t0_gps)

    attributes = BaseSeriesAttrs(
        segment_name,
        Detectors[detector_id],
        t0_gps,
        duration,
        dt,
        sampling_rate,
    )
    if isinstance(strain, numpy.ndarray):
        out_series = CPUSerie(strain, attributes)
    elif isinstance(strain, cupy.ndarray) or use_gpu:
        raise NotImplementedError(
            f"{type(strain)} type for strain is not supported yet."
        )
    else:
        raise NotImplementedError(
            f"{type(strain)} type for strain is not supported yet."
        )

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
    duration: _typing.FLOAT_LIKE | None = None,
    reference_time_gps: _typing.FLOAT_LIKE | None = None,
    use_gpu: bool = False,
):
    """
    Convert a gwpy.timeseries.TimeSeries object into a TimeSeries object.

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

    Notes
    -----
    - This function utilizes the `from_array` function internally for conversion.
    - It automatically handles data conversion based on the specified parameters.
    - If `use_gpu` is set to True, the function will attempt to use GPU for processing.

    Examples
    --------
    Convert a gwpy TimeSeries object into a TimeSeries object:

    >>> import gwpy.timeseries as gw
    >>> import timeserie
    >>> # Assuming 'gwpy_timeseries' is a gwpy.timeseries.TimeSeries object
    >>> # with some data loaded.
    >>> gwpy_timeseries = gw.TimeSeries([1, 2, 3, 4], times=[0, 1, 2, 3])
    >>> ts = TimeSeries.from_gwpy(
    ...     gwpy_timeseries=gwpy_timeseries,
    ...     segment_name="GWPy Data",
    ...     detector_id="H1",
    ... )

    Convert a gwpy TimeSeries object into a TimeSeries object with a specified duration and reference time:

    >>> ts = timeserie.from_gwpy(
    ...     gwpy_timeseries=gwpy_timeseries,
    ...     segment_name="GWPy Data",
    ...     detector_id="H1",
    ...     duration=10.0,  # Duration set to 10 seconds
    ...     reference_time_gps=123456789.0,  # Reference GPS time set to a specific value
    ... )
    """

    if use_gpu:
        strain = cupy.array(gwpy_timeseries.value, dtype=numpy.float32)
        time_axis = cupy.array(gwpy_timeseries.times.value, dtype=numpy.float64)
    else:
        strain = numpy.array(gwpy_timeseries.value, dtype=numpy.float64)
        time_axis = numpy.array(gwpy_timeseries.times.value, dtype=numpy.float64)

    return from_array(
        strain,
        segment_name,
        detector_id,
        time_axis,
        numpy.float64(reference_time_gps),
        duration=numpy.float64(duration),
    )
