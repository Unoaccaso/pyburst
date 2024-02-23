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

    @property
    def values(self):
        """Values property."""
        return self._values

    @property
    def gps_times(self):
        """GPS times property."""
        if self._gps_times is None:
            self._gps_times = (
                numpy.arange(0, self.duration, self.dt, dtype=numpy.float64)
                + self.t0_gps
            )
        return self._gps_times

    def _copy(self, *args, **kwargs):
        old_kwargs = self.__dict__.copy()
        new_kwargs = {}
        for attr_name, value in old_kwargs.items():
            attr_name = attr_name[1:]  # removing the underscore
            new_kwargs[attr_name] = value
        new_kwargs.update(kwargs)
        return _CPUSeries(*args, **new_kwargs)

    def crop(self, start, stop, time_fmt: str = "gps"):
        if time_fmt != "gps":
            start = astropy.time.Time(start, format=time_fmt).gps
            end = astropy.time.Time(end, format=time_fmt).gps
        start_idx = int((start - self.t0_gps) // self.dt)
        end_idx = int((stop - self.t0_gps) // self.dt)
        assert end_idx < len(self.values), f"stop time is too big!"
        assert start_idx > 0, f"start time is too small!"
        t0_gps = numpy.float64(start)
        duration = numpy.float64(stop - start)
        sampling_rate = self.sampling_rate
        gps_times = (
            numpy.arange(duration + 1 / sampling_rate, step=1 / sampling_rate) + t0_gps
        )

        return self._copy(
            values=self.values[start_idx : end_idx + 1],  # include last element
            t0_gps=t0_gps,
            duration=duration,
            sampling_rate=sampling_rate,
            gps_times=gps_times,
        )

    # ! TODO
    @property
    def cache(self): ...

    def __repr__(self) -> str:
        return super().__repr__()
