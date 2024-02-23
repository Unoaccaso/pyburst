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

from dataclasses import dataclass, asdict
import abc
import warnings
import tabulate
import sys

import numpy
import astropy

from ...common._typing import type_check


_DECODE_DETECTOR = {
    "L1": "Ligo Livingston (L1)",
    "H1": "Ligo Hanford (H1)",
    "V1": "Virgo (V1)",
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

    # * Shared functions

    def __getattribute__(self, attr: str):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return getattr(self._values, attr)

    def __getitem__(self, key):
        return self._values.__getitem__(key)

    # * Functions
    @abc.abstractmethod
    def _copy(self, *args, **kwargs): ...

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

    # * Parameters
    @abc.abstractproperty
    def values(self): ...

    @abc.abstractproperty
    def gps_times(self): ...

    @abc.abstractproperty
    def cache(self): ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        str
            String representation.
        """
        # Initialize tables for array and attribute representations
        array_tab = [["name", "content", "shape", "size"]]
        attribute_tab = [["name", "value", "type", "size"]]

        # Iterate through attributes of the object
        for attribute_name, value in self.__dict__.items():
            # Prepare attribute name for display
            parsed_name = attribute_name.replace("_", " ").strip()

            # Check if attribute value is not None
            if value is not None:
                # Check if the value has a 'shape' attribute, indicating it's an array-like object
                if hasattr(value, "shape"):
                    # Check if the shape has length greater than 0
                    if len(value.shape) > 0:
                        # Extract data type string and append to array_tab
                        dtype_str = str(type(value)).split("'")[1]
                        array_tab.append(
                            [
                                f"{parsed_name}\n[{dtype_str}<{value.dtype}>]",  # Concatenate attribute name and type
                                value.__repr__(),  # Get string representation of the value
                                value.shape,  # Get the shape of the array
                                self._format_size(
                                    value.nbytes
                                ),  # Format the size of the array
                            ]
                        )
                    else:
                        # Append attribute details to attribute_tab
                        attribute_tab.append(
                            [
                                parsed_name,  # Attribute name
                                value,  # Value
                                value.dtype,  # Data type
                                self._format_size(value.nbytes),  # Format the size
                            ]
                        )
                else:
                    # If the value doesn't have 'shape', treat it as a scalar value
                    dtype = str(type(value)).split("'")[1]  # Extract data type
                    size = self._format_size(sys.getsizeof(value))  # Format the size
                    attribute_tab.append(
                        [parsed_name, value, dtype, size]
                    )  # Append to attribute_tab
            else:
                # If value is None, treat it as not computed
                array_tab.append(
                    [
                        f"{parsed_name}\n[not allocated yet]",  # Attribute name with indication of not computed
                        "--------",  # Placeholder for content
                        "--------",  # Placeholder for shape
                        self._format_size(0),  # Size as 0
                    ]
                )

        # Format tables into strings using tabulate
        array_str = tabulate.tabulate(
            array_tab,
            headers="firstrow",
            tablefmt="fancy_grid",
            colalign=("left", "center", "right", "right"),
        )
        attribute_str = tabulate.tabulate(
            attribute_tab,
            headers="firstrow",
            tablefmt="outline",
            colalign=("left", "left", "left", "right"),
        )

        # Combine array and attribute strings into final output string
        out_str = f"Time serie content:\n\n{array_str}\n\nTime serie attributes:\n\n{attribute_str}"

        return out_str  # Return the final output string

    def _format_size(self, size) -> str:
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
