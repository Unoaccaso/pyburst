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
from . import _cputs_analysis, _cputs_signal_processing
from .. import _ts_base
from ...common._typing import type_check

# * numpy stuff
import numpy, numpy.typing

import astropy


class _CPUSeries(_ts_base._TimeSeriesBase):

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
            self._fft_values = _cputs_analysis._compute_fft(self)
        return self._fft_values

    @property
    def fft_frequencies(self):
        if self._fft_frequencies is None:
            self._fft_frequencies = _cputs_analysis._compute_fft_freqs(self)
        return self._fft_frequencies

    def crop(self, start, stop, time_fmt: str = "gps"):
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
        kwargs = _cputs_signal_processing._crop(
            self,
            start,
            stop,
            time_fmt,
        )
        return _CPUSeries(**kwargs)

    # ! TODO
    @property
    def cache(self): ...

    def __repr__(self) -> str:
        return super().__repr__()
