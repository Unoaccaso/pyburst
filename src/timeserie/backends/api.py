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

from .common import BackendBase
from .zarr import ZarrStore
from timeserie.core import _BaseTimeSerie
from timeserie.core.cpu import CPUSerie

from enum import Enum
from typing import Type, Union

# from timeserie.core.gpu import GPUSerie
# from timeserie.core.lazy import LazySerie
# from timeserie.core.sparse import SparseSerie

TSTYPE = Union[
    CPUSerie,
    # GPUSerie,
    # LazySerie,
    # SparseSerie,
]


class Engines(Enum):
    zarr = ZarrStore


def from_file(path: str | list[str], engine: Type[BackendBase] = Engines.zarr.value):

    _check_engine(engine)

    timeserie = engine.open_data(path)

    # _validate_timeserie(timeserie)

    return timeserie


def _check_engine(engine: Type[BackendBase]):
    if not issubclass(engine, BackendBase):
        raise BufferError("Provide a valid engine")


def _validate_timeserie(timeserie: Type[_BaseTimeSerie]):
    if not isinstance(timeserie, TSTYPE):
        raise ImportError(f"Engine did not load a valid timeserie")
