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


# * builtin
import typing
import sys

# * my modules
from timeserie.core import _BaseTimeSerie, BaseSeriesAttrs
from timeserie.common import _typing

# * numpy stuff
import numpy, numpy.typing

# * GW
import astropy


class CPUSerie(_BaseTimeSerie):

    def __init__(
        self,
        strain: typing.Union[
            numpy.typing.NDArray[numpy.float32],
            numpy.typing.NDArray[numpy.float64],
        ],
        attrs: BaseSeriesAttrs,
    ):
        self._strain = strain
        self._attrs = attrs

        self._gps_times = None
        self._fft_values = None
        self._fft_freqs = None

    @property
    def strain(self):
        """strain property."""
        return self._strain

    @property
    def gps_times(self):
        """GPS times property."""
        if self._gps_times is None:
            self._gps_times = (
                numpy.arange(0, self._attrs.duration, self._attrs.dt)
                + self._attrs.t0_gps
            ).astype(numpy.float64)
        return self._gps_times

    @property
    def attrs(self):
        return self._attrs

    @property
    def fft_values(self):
        raise NotImplementedError()

    @property
    def fft_freqs(self):
        raise NotImplementedError()

    @property
    def nbytes(self):
        self_content = self.__dict__
        size = 0
        for _, item in self_content.items():
            size += item.nbytes if hasattr(item, "nbytes") else sys.getsizeof(item)
        return numpy.int32(size)

    def get_time_fmt(self, new_fmt: str):
        raise NotImplementedError()

    def save(self): ...

    def __repr__(self) -> str:
        return super().__repr__()
