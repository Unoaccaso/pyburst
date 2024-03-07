"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Friday, February 9th 2024, 2:29:32 pm
Author: Riccardo Felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""

from .caching import LRUCache
from .core.baseserie import _BaseTimeSerie

CACHE = _BaseTimeSerie.CACHE


from .convert import from_array, from_gwpy
from .download import fetch_by_name, fetch_by_gps
from .backends.api import from_file, save

__all__ = ["CACHE"]
