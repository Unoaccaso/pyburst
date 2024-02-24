"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Sunday, February 11th 2024, 7:39:46 pm
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

from collections import OrderedDict
import numpy
import cupy
import pandas

import sys

from ..common._sys import _format_size, _obj_size

from tabulate import tabulate

from warnings import warn


class LRUDownloadCache(OrderedDict):
    """
    A Least Recently Used (LRU) cache implemented using an OrderedDict.

    Parameters
    ----------
    max_size_mb : float, optional
        The maximum size of the cache in megabytes (default is 0.0001 MB).

    Attributes
    ----------
    nbytes : int
        The current size of the cache in bytes.
    cache_size_mb : float
        The maximum size of the cache in megabytes.

    Notes
    -----
    The cache automatically evicts the least recently used items when the size limit is reached.

    Examples
    --------
    >>> cache = LRUCache(max_size_mb=1)
    >>> cache['a'] = 1
    >>> cache['b'] = 2
    >>> cache['c'] = 3
    >>> print(cache.nbytes)
    XXX bytes
    >>> print(cache.cache_size_mb)
    1.0
    """

    def __init__(self, max_size_mb: float = 100, is_cache: bool = True):
        super().__init__()
        self._max_size_bytes = max_size_mb * (1024 * 1024)  # Convert MB to bytes

    def __getitem__(self, key):
        if key in self:
            value = super().__getitem__(key)
            self.move_to_end(key)  # Update the item as the most recent
            return value
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key not in self:
            item_size_bytes = _obj_size(value)
            while self.nbytes + item_size_bytes > self.max_size_bytes:
                self._evict_least_recently_used()
        super().__setitem__(key, value)

    @property
    def nbytes(self):
        size = 0
        for _, serie in self.items():
            size += serie.nbytes
        return size

    def _evict_least_recently_used(self):
        try:
            key, _ = self.popitem(last=False)  # Remove the least recent item
            warn(f"Removed {key} to free cache space.")
        except KeyError:
            raise ValueError(f"Serie is to big to cache!")

    @property
    def cache_size_mb(self):
        """
        float: The maximum size of the cache in megabytes.
        """
        return self.max_size_bytes / (1024 * 1024)

    @property
    def max_size_bytes(self):
        return self._max_size_bytes

    def __repr__(self):
        return _download_cache_repr(self)


def _download_cache_repr(download_cache: LRUDownloadCache):
    rows = [
        [
            "event name",
            "detector id",
            "data type",
            "duration [s]",
            "size",
            r"% of cache",
        ]
    ]
    for key, serie in download_cache.items():
        dtype_str = str(type(serie.values)).split("'")[1]
        rows.append(
            [
                key[0],
                key[1],
                f"[{dtype_str}<{serie.values.dtype}>]",
                f"{serie.duration: .1e}",
                _format_size(serie.nbytes),
                f"{serie.nbytes / download_cache.max_size_bytes * 100 : .2f} %",
            ]
        )
    rows.append(
        [
            "",
            "",
            "",
            "Tot",
            _format_size(download_cache.nbytes),
            f"{download_cache.nbytes / download_cache.max_size_bytes * 100 : .2f} %",
        ]
    )

    return tabulate(rows, headers="firstrow", tablefmt="fancy_grid")
