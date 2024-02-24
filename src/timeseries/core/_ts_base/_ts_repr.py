"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Friday, February 23rd 2024, 5:25:08 pm
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

import sys, os

import tabulate

from ...common._sys import _format_size


def _repr_timeserie(ts_class) -> str:
    """
    Return a string representation of time serie.

    Returns
    -------
    str
        String representation.
    """
    # Initialize tables for array and attribute representations
    array_tab = [["name", "content", "shape", "size"]]
    attribute_tab = [["name", "value", "type", "size"]]

    # Iterate through attributes of the object
    for attribute_name, value in ts_class.__dict__.items():
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
                    value_str = value.__repr__()
                    start_idx = value_str.index("[")
                    end_idx = value_str.rindex("]") + 1
                    array_tab.append(
                        [
                            f"{parsed_name}\n[{dtype_str}<{value.dtype}>]",  # Concatenate attribute name and type
                            value_str[
                                start_idx:end_idx
                            ],  # Get string representation of the value
                            value.shape,  # Get the shape of the array
                            _format_size(value.nbytes),  # Format the size of the array
                        ]
                    )
                else:
                    # Append attribute details to attribute_tab
                    attribute_tab.append(
                        [
                            parsed_name,  # Attribute name
                            value,  # Value
                            value.dtype,  # Data type
                            _format_size(value.nbytes),  # Format the size
                        ]
                    )
            else:
                # If the value doesn't have 'shape', treat it as a scalar value
                dtype = str(type(value)).split("'")[1]  # Extract data type
                size = _format_size(sys.getsizeof(value))  # Format the size
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
                    _format_size(0),  # Size as 0
                ]
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
    out_str = f"Time serie content:\n\n{array_str}\n\nTime serie attributes:\n\n{attribute_str}"

    return out_str  # Return the final output string
