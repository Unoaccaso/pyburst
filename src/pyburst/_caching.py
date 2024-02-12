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

from tabulate import tabulate

from warnings import warn


def nested_dict_to_table(nested_dict):
    rows = []
    for key, value in nested_dict.items():
        for detector, data in value.items():
            row = [key, detector]
            for k, v in data.items():
                if isinstance(v, list):
                    v = f"<TimeSeries(...)>"
                row.append(v)
            rows.append(row)
    headers = ["key 1", "key 2"]
    headers += list(data.keys())
    return tabulate(rows, headers=headers, tablefmt="fancy_grid")


def _obj_size(obj):
    """
    Calculate the total size in bytes of a nested object, considering numpy.ndarray, cupy.ndarray, and pandas.Series objects as well.

    Parameters
    ----------
    obj : dict or list or tuple or numpy.ndarray or cupy.ndarray or pandas.Series
        The nested object to calculate the size of.

    Returns
    -------
    int
        The total size of the nested object in bytes.

    Examples
    --------
    >>> nested_dict = {
    ...     'a': 1,
    ...     'b': {'c': 2, 'd': 3},
    ...     'e': {'f': {'g': 4, 'h': 5}},
    ...     'numpy_array': numpy.zeros((10, 10)),
    ...     'cupy_array': cupy.zeros((10, 10)),
    ...     'pandas_series': pandas.Series(range(10))
    ... }
    >>> size_in_bytes = _obj_size(nested_dict)
    >>> print("Total size of the nested object:", size_in_bytes, "bytes")
    Total size of the nested object: XXX bytes
    """
    total_size = 0

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                total_size += _obj_size(value)
            else:
                total_size += sys.getsizeof(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total_size += _obj_size(item)
    elif isinstance(obj, (numpy.ndarray, cupy.ndarray, pandas.Series)):
        total_size += obj.nbytes
    else:
        total_size += sys.getsizeof(obj)

    return total_size


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
    cache_contents : list
        The keys present in the cache.

    Notes
    -----
    The cache automatically evicts the least recently used items when the size limit is reached.

    Examples
    --------
    >>> cache = LRUCache(max_size_mb=1)
    >>> cache['a'] = 1
    >>> cache['b'] = 2
    >>> cache['c'] = 3
    >>> print(cache.cache_contents)
    ['a', 'b', 'c']
    >>> print(cache.nbytes)
    XXX bytes
    >>> print(cache.cache_size_mb)
    1.0
    """

    def __init__(self, max_size_mb: float = 100):
        super().__init__()
        self.max_size_bytes = max_size_mb * (1024 * 1024)  # Convert MB to bytes

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
            while self._get_size() + item_size_bytes > self.max_size_bytes:
                self._evict_least_recently_used()
        super().__setitem__(key, value)

    def _get_size(self):
        return _obj_size(self)

    def _evict_least_recently_used(self):
        key, _ = self.popitem(last=False)  # Remove the least recent item
        warn(f"Removed {key} to free cache space.")

    @property
    def nbytes(self):
        """
        int: The current size of the cache in bytes.
        """
        return self._get_size()

    @property
    def cache_size_mb(self):
        """
        float: The maximum size of the cache in megabytes.
        """
        return self.max_size_bytes / (1024 * 1024)

    @property
    def cache_contents(self):
        """
        list: The keys present in the cache.
        """
        return list(self.keys())

    def __repr__(self):
        return nested_dict_to_table(self)

    def __str__(self):
        return nested_dict_to_table(self)
