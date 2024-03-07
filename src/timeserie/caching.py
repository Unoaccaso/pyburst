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


from collections import OrderedDict

from warnings import warn


from timeserie.common._sys import format_size, _obj_size
from timeserie.common._typing import type_check

from tabulate import tabulate


class LRUCache(OrderedDict):
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

    @type_check(classmethod=True)
    def read_cached_data(
        self,
        segment_names: str | list["str"] = "all",
        detector_ids: str | list[str] = "all",
    ):
        if segment_names == "all":
            segment_names = [key[0] for key in list(self.keys())]
        elif isinstance(segment_names, str):
            segment_names = [segment_names]
        if detector_ids == "all":
            detector_ids = [key[1] for key in list(self.keys())]
        elif isinstance(detector_ids, str):
            detector_ids = [detector_ids]
        out_dict = LRUCache()
        for segment_name in segment_names:
            for detector_id in detector_ids:
                key = (segment_name, detector_id)
                if key in self and key not in out_dict:
                    out_dict[key] = self[key]
        return out_dict.__repr__()

    @type_check(classmethod=True)
    def get_cached_data(
        self,
        segment_names: str | list["str"] = "all",
        detector_ids: str | list[str] = "all",
    ):
        if segment_names == "all":
            segment_names = [key[0] for key in list(self.keys())]
        elif isinstance(segment_names, str):
            segment_names = [segment_names]
        if detector_ids == "all":
            detector_ids = [key[1] for key in list(self.keys())]
        elif isinstance(detector_ids, str):
            detector_ids = [detector_ids]
        out_dict = dict()
        for segment_name in segment_names:
            for detector_id in detector_ids:
                key = (segment_name, detector_id)
                if key in self and key not in out_dict:
                    out_dict[key] = self[key]
        if len(detector_ids) == 1 and len(segment_names) == 1:
            return out_dict[key]
        else:
            return out_dict

    def __repr__(self):
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
        for key, serie in self.items():
            dtype_str = str(type(serie.strain)).split("'")[1]
            rows.append(
                [
                    key[0],
                    key[1],
                    f"[{dtype_str}<{serie.strain.dtype}>]",
                    f"{serie.attrs.duration: .1e}",
                    format_size(serie.nbytes),
                    f"{serie.nbytes / self.max_size_bytes * 100 : .2f} %",
                ]
            )
        rows.append(
            [
                "",
                "",
                "",
                "Tot",
                format_size(self.nbytes),
                f"{self.nbytes / self.max_size_bytes * 100 : .2f} %",
            ]
        )

        return tabulate(rows, headers="firstrow", tablefmt="fancy_grid")
