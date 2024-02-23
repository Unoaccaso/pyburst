"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Thursday, February 22nd 2024, 11:55:32 am
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

import numpy
import astropy


def _crop(cpu_series, start, stop, time_fmt: str = "gps"):
    """
    Crop the time series to the specified start and stop times.

    Parameters
    ----------
    start : numpy.float64
        Start time of the cropped segment.
    stop : numpy.float64
        Stop time of the cropped segment.
    time_fmt : str, optional
        Format of the start and stop times. Default is "gps".

    Returns
    -------
    _CPUSeries
        A cropped instance of the time series.

    Raises
    ------
    AssertionError
        If the stop time is beyond the end of the time series.
        If the start time is before the beginning of the time series.

    Notes
    -----
    This method crops the time series to the specified start and stop times.
    The start and stop times are specified in GPS time format unless otherwise specified.
    The returned instance is a cropped version of the original time series.
    If other supported formats are specified, the crop will always be done using gps time, to ensure consistency

    """
    if time_fmt != "gps":
        start = numpy.float64(astropy.time.Time(start, format=time_fmt).gps)
        end = numpy.float64(astropy.time.Time(end, format=time_fmt).gps)
    start_idx = int((start - cpu_series.t0_gps) // cpu_series.dt)
    end_idx = int((stop - cpu_series.t0_gps) // cpu_series.dt)
    assert end_idx < len(cpu_series.values), f"stop time is too big!"
    assert start_idx > 0, f"start time is too small!"
    new_t0_gps = numpy.float64(start)
    new_duration = numpy.float64(stop - start)
    gps_times = (
        numpy.arange(
            new_duration + 1 / cpu_series.sampling_rate,
            step=1 / cpu_series.sampling_rate,
        )
        + new_t0_gps
    ).astype(numpy.float64)

    return dict(
        values=cpu_series.values[start_idx : end_idx + 1],
        gps_times=gps_times,
        t0_gps=new_t0_gps,
        duration=new_duration,
        sampling_rate=cpu_series.sampling_rate,
        detector_id=cpu_series.detector_id,
        reference_time_gps=cpu_series.reference_time_gps,
        segment_name=cpu_series.segment_name,
        dt=cpu_series.dt,
    )
