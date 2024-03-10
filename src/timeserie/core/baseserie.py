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


import numpy
from numpy import int32, int64, float32, float64, complex64, complex128
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, asdict
from enum import Enum
import tabulate
import sys

from timeserie.common import _typing
from timeserie.caching import LRUCache
from timeserie.common._sys import format_size


class Detectors(Enum):
    L1 = "Ligo Livingston (L1)"
    H1 = "Ligo Hanford (H1)"
    V1 = "Virgo (V1)"
    _ = "_"


@dataclass
class BaseSeriesAttrs:
    """
    Attributes of a Time Series.

    This data class defines the attributes of a Time Series, including its segment name,
    detector ID, start time, duration, time step, and sampling rate. It ensures data integrity
    checks during initialization.

    Parameters
    ----------
    segment_name : str
        Name of the segment. Must be a string representing the segment's identifier.

    detector : Detectors
        Identifier of the detector. Must be one of the predefined detector types (L1, H1, V1).

    t0_gps : float64
        Reference GPS time. It indicates the starting time of the time series data. Must be a positive float.

    duration : float64
        Duration of the time series in seconds. Must be a positive float.

    dt : float64
        Time step or sampling interval of the time series. Must be a positive float.

    sampling_rate : int64
        Sampling rate of the time series in Hertz (Hz). Must be a power of 2 and smaller than 4096.

    Raises
    ------
    ValueError
        If input data types or values are invalid.

    Data Integrity Checks:
    ----------------------
    - `segment_name`: Must be a string representing the segment's identifier.
    - `detector`: Must be one of the predefined detector types (L1, H1, V1).
    - `t0_gps`: Must be a positive float representing the reference GPS time.
    - `duration`: Must be a positive float representing the duration of the time series.
    - `dt`: Must be a positive float representing the time step or sampling interval.
    - `sampling_rate`: Must be a power of 2 and smaller than 4096, representing the sampling rate in Hertz (Hz).

    These integrity checks ensure that the Time Series attributes are valid and consistent, meeting the required criteria for further processing and analysis.
    """

    segment_name: str
    detector: Detectors
    t0_gps: float64
    duration: float64
    dt: float64
    sampling_rate: int64

    def __post_init__(self):

        # Check for data type
        # ===================
        if not isinstance(self.segment_name, str):
            raise ValueError(f"segment_name must be a string.")
        if not isinstance(self.detector, Detectors):
            raise ValueError(f"detector must be a Detector.")
        if not isinstance(self.t0_gps, float64):
            raise ValueError(f"t0_gps must be a float64.")
        if not isinstance(self.duration, float64):
            raise ValueError(f"duration must be a float64.")
        if not isinstance(self.dt, float64):
            raise ValueError(f"dt must be a float64.")
        if not isinstance(self.sampling_rate, int64):
            raise ValueError(f"sampling_rate must be a int64.")

        # Checks for data integrity
        # =========================

        # Sampling rate must be a power of 2, smaller than 4096
        if not (numpy.log2(self.sampling_rate) % 1 == 0 and self.sampling_rate <= 4096):
            raise ValueError("Sampling rate must be a power of 2, smaller than 4096")

        # dt and sampling rate compatibility
        if not (1 / self.dt == self.sampling_rate):
            raise ValueError("dt and sampling_rate are not compatible.")

        # t0 compatibility
        if self.t0_gps < 0:
            raise ValueError("t0_gps must be positive")
        elif not (self.t0_gps == numpy.floor(self.t0_gps / self.dt) * self.dt):
            raise ValueError(
                f"t0_gps: {self.t0_gps} is incompatible with dt and sampling rate.\n\nThe correct value should be {numpy.floor(self.t0_gps / self.dt) * self.dt}"
            )

        # duration compatibility
        if not (self.duration == numpy.ceil(self.duration / self.dt) * self.dt):
            raise ValueError("duration is incompatible with dt and sampling rate")

    # modifying the setter to ensure read-only behaviour
    def __setattr__(self, __name: str, __value) -> None:
        if hasattr(self, __name):
            raise PermissionError(f"Attributes are readonly!")
        else:
            super().__setattr__(__name, __value)


@dataclass
class GPS_Interval:
    """
    Represents a GPS time interval.

    This data class defines a GPS time interval specified by a start time and a stop time.
    It ensures data integrity checks during initialization.

    Parameters
    ----------
    time_range : list[float]
        A list representing the start and stop times of the GPS interval. It should contain
        two elements representing the start and stop times, respectively. Both elements
        must be positive real numbers, and the start time must be strictly less than the
        stop time.

    Raises
    ------
    ValueError
        If input data types or values are invalid.

    Examples
    --------
    >>> interval = GPS_Interval([123456789.0, 123456799.0])
    >>> print(interval)
    GPS_Interval(time_range=[123456789.0, 123456799.0])

    >>> # This will raise a ValueError due to invalid input
    >>> interval = GPS_Interval([123456799.0, 123456789.0])
    ValueError: Start time must be smaller than stop time.
    """

    time_range: list[_typing.REAL_NUM, _typing.REAL_NUM]

    def __post_init__(self):
        # check input type
        if not isinstance(self.time_range, list):
            raise ValueError(f"Range must be a list of numbers.")
        # check input shape
        if len(self.time_range) != 2:
            raise ValueError(f"Range must be a list with 2 elements.")
        # check list content
        if not isinstance(self.time_range[0], _typing.REAL_NUM) or not isinstance(
            self.time_range[1], _typing.REAL_NUM
        ):
            raise ValueError(f"Range values must be real numbers.")
        # check time integrity
        if self.time_range[0] >= self.time_range[1]:
            raise ValueError(f"Start time must be smaller than stop time.")


class _BaseTimeSerie(ABC):
    """
    Abstract base class for Time Series objects.

    Attributes
    ----------
    CACHE : LRUCache
        Cache for storing Time Series data.

    Notes
    -----
    Subclasses must implement the following properties and methods:
    - strain : numpy.ndarray
        Strain data of the Time Series.
    - gps_times : numpy.ndarray
        GPS times corresponding to the strain data.
    - attrs : _BaseSeriesAttrs
        Attributes of the Time Series.
    - fft_values : numpy.ndarray
        Fourier Transform values of the strain data.
    - fft_freqs : numpy.ndarray
        Frequencies corresponding to the Fourier Transform values.

    Methods
    -------
    get_time_fmt(new_fmt: str)
        Convert the time format of the GPS times.
    __repr__()
        Return a string representation of the Time Series object.
    """

    _cache_size_mb = 1_000
    CACHE = LRUCache(max_size_mb=_cache_size_mb)

    @abstractproperty
    def strain(self):
        """
        Strain data property.

        Returns
        -------
        Array
            Strain data array.
        """
        pass

    @abstractproperty
    def gps_times(self):
        """
        GPS times property.

        Returns
        -------
        Array
            GPS times array.
        """
        pass

    @abstractproperty
    def attrs(self):
        """
        Attributes property.

        Returns
        -------
        _BaseSeriesAttrs
            Time Series attributes.
        """
        pass

    @abstractproperty
    def fft_values(self):
        """
        FFT values property.

        Returns
        -------
        Array
            FFT values array.
        """
        pass

    @abstractproperty
    def fft_freqs(self):
        """
        FFT frequencies property.

        Returns
        -------
        Array
            FFT frequencies array.
        """
        pass

    @abstractproperty
    def nbytes(self):
        """
        Number of bytes property.

        Returns
        -------
        int
            Number of bytes occupied by the Time Series data.
        """
        pass

    @abstractmethod
    def get_time_fmt(self, new_fmt: str):
        """
        Get time format method.

        Parameters
        ----------
        new_fmt : str
            New time format.

        Returns
        -------
        TimeSeries
            Time Series object with the specified time format.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save method

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        String representation method.

        Returns
        -------
        str
            String representation of the Time Series.
        """
        # Initialize tables for array and attribute representations
        array_tab = [["name", "content", "shape", "size"]]
        attribute_tab = [["name", "value", "type", "size"]]

        array_tab.append(
            # adding strain to print
            [
                f"strain\n{type(self.strain)}<{self.strain.dtype}>",
                self.strain.__repr__(),
                self.strain.shape,
                format_size(self.strain.nbytes),
            ]
        )

        array_tab.append(
            # adding gps time
            [
                (
                    f"strain\n{type(self._gps_times)}<{self._gps_times.dtype}>"
                    if self._gps_times is not None
                    else "gps_time\n'not allocated yet'"
                ),
                self._gps_times.__repr__() if self._gps_times is not None else "-----",
                self._gps_times.shape if self._gps_times is not None else "-----",
                self._gps_times.nbytes if self._gps_times is not None else "-----",
            ]
        )

        array_tab.append(
            # adding fft values
            [
                (
                    f"fft_values\n{type(self._fft_values)}<{self._fft_values.dtype}>"
                    if self._fft_values is not None
                    else "fft_values\n'not allocated yet'"
                ),
                (
                    self._fft_values.__repr__()
                    if self._fft_values is not None
                    else "-----"
                ),
                self._fft_values.shape if self._fft_values is not None else "-----",
                self._fft_values.nbytes if self._fft_values is not None else "-----",
            ]
        )

        array_tab.append(
            # adding fft freqs
            [
                (
                    f"fft_freqs\n{type(self._fft_freqs)}<{self._fft_freqs.dtype}>"
                    if self._fft_freqs is not None
                    else "fft_freqs\n'not allocated yet'"
                ),
                self._fft_freqs.__repr__() if self._fft_freqs is not None else "-----",
                self._fft_freqs.shape if self._fft_freqs is not None else "-----",
                self._fft_freqs.nbytes if self._fft_freqs is not None else "-----",
            ]
        )

        for key, attr in asdict(self.attrs).items():

            _type = (
                attr.dtype if hasattr(attr, "dtype") else str(type(attr)).split("'")[1]
            )
            size = format_size(
                attr.nbytes if hasattr(attr, "nbytes") else sys.getsizeof(attr)
            )
            attribute_tab.append(
                [key, attr if key != "detector" else attr.value, _type, size]
            )

        # Format tables into strings using tabulate
        array_str = tabulate.tabulate(
            array_tab,
            headers="firstrow",
            tablefmt="fancy_grid",
            colalign=("left", "center", "center", "right"),
        )
        attribute_str = tabulate.tabulate(
            attribute_tab,
            headers="firstrow",
            tablefmt="outline",
            colalign=("left", "left", "left", "right"),
        )

        # Combine array and attribute strings into final output string
        out_str = f"\nTime serie content:\n\n{array_str}\n\nTime serie attributes:\n\n{attribute_str}\n"

        return out_str  # Return the final output string
