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

from functools import wraps


import typing
import types

import cupy
import numpy
import dask.array

T = typing.TypeVar("T")

from numpy import float32, float64, int32, int64, complex64, complex128


ARRAY_LIKE = typing.Union[
    numpy.ndarray[float32],
    numpy.ndarray[float64],
    dask.array.Array,
    cupy.typing.NDArray[float32],
    cupy.typing.NDArray[float64],
]
FLOAT_LIKE = typing.Union[
    float,
    float32,
    float64,
]
INT_LIKE = typing.Union[
    int,
    int32,
    int64,
]
COMPLEX_LIKE = typing.Union[
    complex,
    complex64,
    complex128,
]
REAL_NUM = typing.Union[INT_LIKE, FLOAT_LIKE]
COMPLEX_NUM = typing.Union[REAL_NUM, COMPLEX_LIKE]


_FLOAT_EPS = 1e-8


def _check_arg(arg, var_name, arg_type_hints):
    """
    Checks if an argument is an instance of one or more specified types.

    Args:
        arg: The argument to check.
        arg_type_hints: Type hints specifying the type(s) the argument should have.
            It can be a simple type or a composed type, such as a Union of types.

    Raises:
        TypeError: If the argument is not an instance of the specified type.
        NotImplementedError: If type checking for a certain type is not implemented.

    Examples:
        # Check if 'value' is an integer
        _check_arg(value, int)

        # Check if 'data' is a list of integers
        _check_arg(data, list[int])

        # Check if 'value' is either a string or an integer
        _check_arg(value, typing.Union[str, int])

        # Check if 'matrix' is a list of lists of integers
        _check_arg(matrix, list[list[int]])
    """
    type_str = (
        f"({type(arg)}[{arg.dtype}])" if hasattr(arg, "dtype") else f"({type(arg)})"
    )
    type_error_msg = f"'{var_name}' : {type_str} is not an instance of {arg_type_hints}"
    arg_type_origin = typing.get_origin(arg_type_hints)
    _implemented_types = [numpy.ndarray, cupy.ndarray, list, dask.array.Array]

    # Check for simple types
    if arg_type_origin is None:
        if not isinstance(arg, arg_type_hints):
            raise TypeError(type_error_msg)

    # Check for composed types
    else:
        arg_types = typing.get_args(arg_type_hints)

        if arg_type_origin in (
            typing.Union,
            types.UnionType,
            typing.Generic,
        ):
            for type_hint in arg_types:
                try:
                    _check_arg(arg, var_name, type_hint)
                    break
                except TypeError:
                    continue
            else:
                raise TypeError(type_error_msg)

        elif arg_type_origin in _implemented_types:
            if len(arg_types) > 1 and arg_type_origin != cupy.ndarray:
                raise NotImplementedError(
                    f"Cannot check {arg_type_origin}, non unadic list not supported!"
                )
            elif len(arg_types) > 2 and arg_type_origin == cupy.ndarray:
                raise NotImplementedError(
                    f"Cannot check {arg_type_origin}, non unadic list not supported!"
                )
            if type(arg) != arg_type_origin:
                raise TypeError(type_error_msg)

            if arg_type_origin == list:
                if not all(type(elem) in arg_types for elem in arg):
                    raise TypeError(type_error_msg)

            elif arg_type_origin == numpy.ndarray:
                if arg.dtype not in arg_types:
                    raise TypeError(type_error_msg)

            elif arg_type_origin == cupy.ndarray:
                if arg.dtype not in arg_types[1].__args__:
                    raise TypeError(type_error_msg)
            else:
                raise NotImplementedError(
                    f"Type checking for {arg_type_hints} not implemented"
                )
        else:
            raise NotImplementedError(
                f"Type checking for {arg_type_hints} not implemented"
            )


def type_check(classmethod: bool = False) -> typing.Callable[..., T]:
    """
    Decorator that enforces strong typing for the arguments of a function based on type hints.

    Parameters
    ----------
    func : callable
        The function to be decorated.
    classmethod : bool
        Specifies if function is a classmethod.

    Returns
    -------
    callable
        A wrapper function that performs type checking before calling the original function.

    Raises
    ------
    AssertionError
        If any argument does not match its annotated type.

    Examples
    --------

    >>> from pyburst import type_check
    >>> @type_check
    ... def my_function(x: int, y: float):
    ...     return x + y
    >>> my_function(3, 4.5)  # No type errors
    7.5
    >>> my_function("a", 4.5)  # Raises AssertionError
    Traceback (most recent call last):
        ...
    AssertionError: a is not an instance of <class 'int'>

    It also supports numpy and cupy arrays

    >>> from numpy.typing import NDArray
    >>> from numpy import int32, float64
    >>> @type_check
    ... def my_function(x: NDArray[int32]):
    ...     pass
    >>> arr_int = numpy.array([1, 2, 3, 4], dtype = int32)
    >>> arr_float = numpy.array([1, 2, 3, 4], dtype = float64)
    >>> my_function(arr_int)  # No type errors
    >>> my_function(arr_float)  # Raises AssertionError
    Traceback (most recent call last):
        ...
    AssertionError: a is not an instance of <class 'numpy.int32'>

    When applied to a function inside a class, use the parameter classmethod=True

    >>> class  MyClass:
    ...
    >>>     @classmethod
    >>>     @type_check(classmethod=True)
    >>>     def my_func():
    ...         pass



    """

    def decorator(func):
        var_name_and_type = typing.get_type_hints(func)
        var_names = list(var_name_and_type.keys())

        if classmethod:
            # adding an empty element to account for cls argument
            var_names = [""] + var_names

        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args):
                if classmethod and i == 0:
                    # skipping first element for classmethods
                    pass
                else:
                    arg_type = var_name_and_type[var_names[i]]
                    _check_arg(arg, var_names[i], arg_type)

            for kwarg_name, kwarg in kwargs.items():
                _check_arg(kwarg, kwarg_name, var_name_and_type[kwarg_name])

            return func(*args, **kwargs)

        return wrapper

    return decorator
