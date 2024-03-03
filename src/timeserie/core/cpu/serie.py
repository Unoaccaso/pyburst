"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Thursday, February 22nd 2024, 11:55:19 am
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

# * builtin
import typing

# * my modules
from timeserie.core import baseserie

# * numpy stuff
import numpy, numpy.typing

# * GW
import astropy


class CPUSerie(baseserie._TimeSeriesBase):

    def __init__(
        self,
        values: typing.Union[
            numpy.typing.NDArray[numpy.float32],
            numpy.typing.NDArray[numpy.float64],
        ],
        gps_times: numpy.typing.NDArray[numpy.float64],
        *args,
        **kwargs,
    ):
        self._values = values
        self._gps_times = gps_times
        super().__init__(*args, **kwargs)

        self._fft_values = None
        self._fft_frequencies = None
        self._nbytes = None

    @property
    def nbytes(self):
        if self._nbytes is None:
            self._nbytes = super().nbytes
        else:
            return self._nbytes

    @property
    def values(self):
        """Values property."""
        return self._values

    @property
    def gps_times(self):
        """GPS times property."""
        if self._gps_times is None:
            self._gps_times = (
                numpy.arange(0, self.duration, self.dt) + self.t0_gps
            ).astype(numpy.float64)
        return self._gps_times

    @property
    def fft_values(self):
        if self._fft_values is None:
            self._fft_values = _fft(self)
        return self._fft_values

    @property
    def fft_frequencies(self):
        if self._fft_frequencies is None:
            self._fft_frequencies = _fftfreqs(self)
        return self._fft_frequencies

    def crop(self, start, stop, time_fmt: str = "gps"):
        """
        Crop the time self to the specified start and stop times.

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
        CPUSerie
            A cropped instance of the time self.

        Raises
        ------
        AssertionError
            If the stop time is beyond the end of the time self.
            If the start time is before the beginning of the time self.

        Notes
        -----
        This method crops the time self to the specified start and stop times.
        The start and stop times are specified in GPS time format unless otherwise specified.
        The returned instance is a cropped version of the original time self.
        If other supported formats are specified, the crop will always be done using gps time, to ensure consistency

        """
        return _crop(
            self,
            start,
            stop,
            time_fmt,
        )

    def save(self, path: str, fmt: str = "zarr"):
        return super().save(path, fmt)

    def __repr__(self) -> str:
        return super().__repr__()


def _crop(cpu_serie: CPUSerie, start, stop, time_fmt: str = "gps"):
    """
    Crop the time self to the specified start and stop times.

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
    CPUSerie
        A cropped instance of the time self.

    Raises
    ------
    AssertionError
        If the stop time is beyond the end of the time self.
        If the start time is before the beginning of the time self.

    Notes
    -----
    This method crops the time self to the specified start and stop times.
    The start and stop times are specified in GPS time format unless otherwise specified.
    The returned instance is a cropped version of the original time self.
    If other supported formats are specified, the crop will always be done using gps time, to ensure consistency

    """
    if time_fmt != "gps":
        start = numpy.float64(astropy.time.Time(start, format=time_fmt).gps)
        end = numpy.float64(astropy.time.Time(end, format=time_fmt).gps)
    start_idx = int((start - cpu_serie.t0_gps) // cpu_serie.dt)
    end_idx = int((stop - cpu_serie.t0_gps) // cpu_serie.dt)
    assert end_idx <= len(cpu_serie.values), f"stop time is too big!"
    assert start_idx >= 0, f"start time is too small!"
    new_t0_gps = numpy.float64(start)
    new_duration = numpy.float64(stop - start)
    gps_times = (
        numpy.arange(
            new_duration + 1 / cpu_serie.sampling_rate,
            step=1 / cpu_serie.sampling_rate,
        )
        + new_t0_gps
    ).astype(numpy.float64)

    return CPUSerie(
        values=cpu_serie.values[start_idx : end_idx + 1],
        gps_times=gps_times,
        t0_gps=new_t0_gps,
        duration=new_duration,
        sampling_rate=cpu_serie.sampling_rate,
        detector_id=cpu_serie.detector_id,
        reference_time_gps=cpu_serie.reference_time_gps,
        segment_name=cpu_serie.segment_name,
        dt=cpu_serie.dt,
    )


def _fft(cpu_serie: CPUSerie): ...


def _fftfreqs(cpu_serie: CPUSerie): ...
