"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Sunday, February 25th 2024, 12:43:35 pm
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

from .zarr import ZarrStore
from timeserie.core.cpu import CPUSerie

# from timeserie.core.gpu import GPUSerie
# from timeserie.core.lazy import LazySerie
# from timeserie.core.sparse import SparseSerie


ENGINES = {
    "zarr": ZarrStore,
}


def from_file(path: str | list[str], engine: str = "zarr"):
    if engine not in ENGINES:
        raise NotImplementedError(f"{engine} is not implemented")

    ENGINES[engine].open_data(path)


def save(timeserie: CPUSerie, path: str, engine: str = "zarr"):
    if engine not in ENGINES:
        raise NotImplementedError(f"{engine} is not implemented")

    ENGINES[engine].save_data(path)
