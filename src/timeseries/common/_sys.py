"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Saturday, February 24th 2024, 11:04:59 am
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

import sys
import numpy, cupy.typing, pandas, dask.array


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
    >>> download_cache = {
    ...     'a': 1,
    ...     'b': {'c': 2, 'd': 3},
    ...     'e': {'f': {'g': 4, 'h': 5}},
    ...     'numpy_array': numpy.zeros((10, 10)),
    ...     'cupy_array': cupy.zeros((10, 10)),
    ...     'pandas_series': pandas.Series(range(10))
    ... }
    >>> size_in_bytes = _obj_size(download_cache)
    >>> print("Total size of the nested object:", size_in_bytes, "bytes")
    Total size of the nested object: XXX bytes
    """
    total_size = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            total_size += _obj_size(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total_size += _obj_size(item)
    elif hasattr(obj, "nbytes"):
        total_size += obj.nbytes
    else:
        total_size += sys.getsizeof(obj)

    return total_size


def _format_size(size) -> str:
    """
    Format the size for readability.

    Parameters
    ----------
    size : int
        Size to format.

    Returns
    -------
    str
        Formatted size string.
    """
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size / 1024:.2f} kB"
    elif size < 1024**3:
        return f"{size / (1024 ** 2):.2f} MB"
    else:
        return f"{size / (1024 ** 3):.2f} GB"
