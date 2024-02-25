"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Thursday, February 22nd 2024, 11:54:28 am
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

import abc
import warnings
import sys

import numpy

from . import ts_repr
from ...common._typing import type_check
from .backend.api import ENGINES


_DECODE_DETECTOR = {
    "L1": "Ligo Livingston (L1)",
    "H1": "Ligo Hanford (H1)",
    "V1": "Virgo (V1)",
    "-": "-",
}


class _BaseSeriesAttrs:
    """
    Base class for common series attributes.

    This class defines the basic attributes common to all series classes.
    """

    @type_check(classmethod=True)
    def __init__(
        self,
        segment_name: str,
        detector_id: str,
        t0_gps: numpy.float64,
        duration: numpy.float64,
        dt: numpy.float64,
        sampling_rate: numpy.int64,
        reference_time_gps: numpy.float64 | None = None,
        detector_name: str | None = None,
    ):
        """
        Initialize the BaseSeriesAttrs object.

        Parameters
        ----------
        segment_name : str
            Name of the segment.
        detector_id : str
            ID of the detector.
        t0_gps : numpy.float64
            GPS start time.
        duration : numpy.float64
            Duration of the segment.
        dt : numpy.float64
            Time step.
        sampling_rate : numpy.int64
            Sampling rate.
        reference_time_gps : numpy.float64, optional
            Reference time in GPS. Defaults to None.
        detector_name : str, optional
            Name of the detector. Defaults to None.

        Raises
        ------
        ValueError
            If detector_id is not supported.
            If provided detector_name is not valid for the given detector_id.
            If dt and sampling rate are incompatible.
        """
        self._segment_name = segment_name
        self.validate_detector_id(detector_id)
        self.validate_detector_name(detector_name)
        self.validate_time_data(dt, sampling_rate)
        self._t0_gps = t0_gps
        self._duration = duration
        self.validate_reference_time(reference_time_gps)

        self.array_items = ["values", "gps_times", "fft_values", "fft_frequencies"]

    def validate_detector_id(self, detector_id):
        """
        Validate the detector ID.

        Parameters
        ----------
        detector_id : str
            ID of the detector.

        Raises
        ------
        ValueError
            If detector_id is not supported.
        """
        if detector_id in _DECODE_DETECTOR:
            self._detector_id = detector_id
        else:
            raise ValueError(f"{detector_id} is not a supported detector")

    def validate_detector_name(self, detector_name):
        """
        Validate the detector name.

        Parameters
        ----------
        detector_name : str
            Name of the detector.

        Raises
        ------
        ValueError
            If provided detector_name is not valid for the given detector_id.
        """
        if detector_name is None or detector_name == _DECODE_DETECTOR[self.detector_id]:
            self._detector_name = _DECODE_DETECTOR[self.detector_id]
        else:
            raise ValueError(
                f"{self._detector_id} is the id for {_DECODE_DETECTOR[self._detector_id]}, not {detector_name}"
            )

    def validate_time_data(self, dt, sampling_rate):
        """
        Validate time data.

        Parameters
        ----------
        dt : numpy.float64
            Time step.
        sampling_rate : numpy.int64
            Sampling rate.

        Raises
        ------
        ValueError
            If dt and sampling rate are incompatible.
        """
        if numpy.round(1 / dt).astype(numpy.int64) == sampling_rate:
            self._dt = dt
            self._sampling_rate = sampling_rate
        else:
            raise ValueError(
                f"dt: {dt} and sampling rate: {sampling_rate} are not compatible."
            )

    def validate_reference_time(self, reference_time):
        """
        Validate reference time.

        Parameters
        ----------
        reference_time : numpy.float64
            Reference time in GPS.

        Warnings
        --------
        UserWarning
            If reference_time is None, it will be set to t0_gps.
        """
        if reference_time is None:
            self._reference_time_gps = self.t0_gps
            warnings.warn(f"reference_time_gps set to {self.t0_gps}")
        else:
            self._reference_time_gps = reference_time

    @property
    def segment_name(self) -> str:
        """str: Name of the segment."""
        return self._segment_name

    @segment_name.setter
    def segment_name(self, value: str):
        self._segment_name = value

    @property
    def reference_time_gps(self) -> numpy.float64:
        """numpy.float64: Reference time in GPS."""
        return self._reference_time_gps

    @reference_time_gps.setter
    def reference_time_gps(self, value: numpy.float64):
        self._reference_time_gps = value

    # Readonly attrs

    @property
    def detector_id(self) -> str:
        """str: ID of the detector."""
        return self._detector_id

    @property
    def detector_name(self) -> str:
        """str: Name of the detector."""
        return self._detector_name

    @property
    def t0_gps(self) -> numpy.float64:
        """numpy.float64: GPS start time."""
        return self._t0_gps

    @property
    def duration(self) -> numpy.float32:
        """numpy.float32: Duration of the segment."""
        return self._duration

    @property
    def dt(self) -> numpy.float32:
        """numpy.float32: Time step."""
        return self._dt

    @property
    def sampling_rate(self) -> numpy.int32:
        """numpy.int32: Sampling rate."""
        return self._sampling_rate


class _TimeSeriesBase(abc.ABC, _BaseSeriesAttrs):
    """
    Base class for TimeSeries.

    This class is the base for all the Series implementation
    (ex. GPU, CPU, Lazy etc.). It implements all the basic functionalities,
    common to all the below, ensuring the firm.
    """

    # * ABSTRACT PROPERTIES

    @abc.abstractproperty
    def values(self): ...

    @abc.abstractproperty
    def gps_times(self): ...

    @abc.abstractproperty
    def fft_values(self): ...

    @abc.abstractproperty
    def fft_frequencies(self): ...

    @abc.abstractproperty
    def nbytes(self):
        self_content = self.__dict__
        size = 0
        for _, item in self_content.items():
            size += item.nbytes if hasattr(item, "nbytes") else sys.getsizeof(item)
        return size

    # * FUNCTIONS

    def __getattribute__(self, attr: str):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return getattr(self._values, attr)

    def __getitem__(self, key):
        return self._values.__getitem__(key)

    # * ABSTRACT FUNCTIONS

    @abc.abstractmethod
    def crop(self, start, stop, time_fmt: str = "gps"):
        """
        Time-based indexing.

        Parameters
        ----------
        start : numpy.float64
            Start time.
        stop : numpy.float64
            Stop time.
        time_fmt : str, optional
            Time format. Defaults to "gps".
        """
        ...

    @abc.abstractmethod
    def save(self, path: str, fmt: str = "zarr"):
        """Save data and metadata to files in parallel.

        This method saves the data to files in either Zarr or HDF5 format in parallel,
        while also saving the metadata separately in HDF5 format.

        Args:
            path (str): The directory path where the files will be saved.
            fmt (str, optional): The format for saving data ('zarr' or 'hdf5'). Defaults to 'zarr'.

        Raises:
            NotImplementedError: If the specified format is not supported.

        Example:
            To save data and metadata to files in parallel, first create an instance of YourClassName
            and set its attributes:

            >>> instance = YourClassName()
            >>> instance.segment_name = "Segment1"
            >>> instance.detector_id = 1
            >>> instance.reference_time_gps = 123456789
            >>> instance.values = ...  # Assign the data values
            >>> instance.gps_times = ...  # Assign the GPS times
            >>> instance.fft_values = ...  # Assign the FFT values
            >>> instance.fft_frequencies = ...  # Assign the FFT frequencies

            Then, call the save method with the directory path and format:

            >>> instance.save("/path/to/save", fmt="zarr")
        """
        # Check if the format is supported
        if fmt not in ENGINES:
            raise NotImplementedError(f"{fmt} is not a supported format")
        else:
            ENGINES[fmt].save_data(self, path)

    @abc.abstractmethod
    def __repr__(self) -> str:
        return ts_repr._repr_timeserie(self)
