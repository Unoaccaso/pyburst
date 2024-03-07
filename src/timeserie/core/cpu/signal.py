# Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>
#
# Created Date: Thursday, February 22nd 2024, 10:07:32 am
# Author: unoaccaso
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, version 3. This program is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https: //www.gnu.org/licenses/>.


import numpy

from timeserie.core.cpu import CPUSerie
from timeserie.core import GPS_Interval
from timeserie import from_array


def crop_gps(timeserie: CPUSerie, interval: GPS_Interval):
    """
    Crop the Time Series data based on the specified GPS time interval.

    This function extracts a portion of the Time Series data that falls within the specified
    GPS time interval.

    Parameters
    ----------
    timeserie : CPUSerie
        The Time Series data to be cropped.
    interval : GPS_Interval
        The GPS time interval specifying the start and stop times for cropping the data.

    Returns
    -------
    TimeSeries
        The cropped Time Series object.

    Raises
    ------
    ValueError
        If the stop time of the interval exceeds the end time of the Time Series data.
        If the start time of the interval is smaller than the start time of the Time Series data.

    Notes
    -----
    This function creates a new Time Series object from the cropped data, preserving the original
    metadata such as segment name, detector ID, sampling rate, etc. It enforces data integrity checks
    to ensure the validity of the cropping operation.

    Examples
    --------
    >>> from mylibrary import crop_gps, GPS_Interval

    # Assuming 'timeserie' is a CPUSerie object and 'interval' is a GPS_Interval object
    >>> cropped_series = crop_gps(timeserie, interval)
    """
    start = interval.time_range[0]
    stop = interval.time_range[1]

    start_idx = int((start - timeserie.attrs.t0_gps) // timeserie.attrs.dt)
    end_idx = int((stop - timeserie.attrs.t0_gps) // timeserie.attrs.dt)
    if end_idx > len(timeserie.strain):
        raise ValueError(f"stop time is too big!")
    if start_idx < 0:
        raise ValueError(f"start time is too small!")
    new_t0_gps = numpy.float64(start)
    new_duration = numpy.float64(stop - start)

    return from_array(
        strain=timeserie.strain[start_idx:end_idx],
        segment_name=timeserie.attrs.segment_name,
        detector_id=timeserie.attrs.detector.name,
        dt=timeserie.attrs.dt,
        sampling_rate=timeserie.attrs.sampling_rate,
        t0_gps=new_t0_gps,
        duration=new_duration,
        force_cache_overwrite=True,
        cache_results=True,
    )


import pandas as pd
from astropy.time import Time
from astropy import units as u


def crop_fmt(timeserie: CPUSerie, time_range: list, fmt: str = None):
    """
    Crop the Time Series data based on the specified time range.

    This function extracts a portion of the Time Series data that falls within the specified
    time range. The time range is provided as a list with two elements representing the start
    and stop times.

    Parameters
    ----------
    timeserie : CPUSerie
        The Time Series data to be cropped.
    time_range : list
        A list of two elements representing the start and stop times. The format of the times
        can be specified using the 'fmt' parameter.
    fmt : str, optional
        The format of the input times. If not provided, the function will attempt to infer
        the format.

    Returns
    -------
    TimeSeries
        The cropped Time Series object.

    Raises
    ------
    ValueError
        If the length of the time range list is not equal to 2.
        If the stop time of the interval exceeds the end time of the Time Series data.
        If the start time of the interval is smaller than the start time of the Time Series data.

    Notes
    -----
    This function reads and/or recognizes the input time format using Pandas, converts the times
    to GPS times using Astropy, and creates a GPS_Interval to crop the Time Series data.

    Examples
    --------
    >>> from mylibrary import crop_fmt

    # Assuming 'timeserie' is a CPUSerie object
    # and 'time_range' is a list with two elements representing the start and stop times
    >>> cropped_series = crop_fmt(timeserie, ['2024-03-01 10:00:00', '2024-03-01 10:10:00'])
    """
    # Check the integrity of the time range
    if len(time_range) != 2:
        raise ValueError("Time range must contain exactly two elements.")

    # Convert time range to Pandas DateTimeIndex
    time_range_pd = pd.to_datetime(time_range, format=fmt)

    # Convert Pandas DateTimeIndex to Astropy Time object
    time_range_astropy = Time(time_range_pd)

    # Convert Astropy Time to GPS times
    gps_times = time_range_astropy.gps

    # Create GPS_Interval
    interval = GPS_Interval(time_range=gps_times.value)

    # Crop Time Series data using crop_gps
    return crop_gps(timeserie, interval)


def resample(timeserie: CPUSerie):
    pass


def whiten(timeserie: CPUSerie):
    pass
