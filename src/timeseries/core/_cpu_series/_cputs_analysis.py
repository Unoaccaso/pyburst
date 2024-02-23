"""
Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>

Created Date: Thursday, February 22nd 2024, 11:55:30 am
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

import numpy


def _compute_fft(cpu_series):
    if cpu_series._values.dtype == numpy.float64:
        return numpy.fft.fft(cpu_series._values).astype(numpy.complex128)
    elif cpu_series._values.dtype == numpy.float32:
        return numpy.fft.fft(cpu_series._values).astype(numpy.complex64)
    else:
        raise Exception


def _compute_fft_freqs(cpu_series):
    return (
        numpy.fft.fftfreq(len(cpu_series._values)) * cpu_series.sampling_rate
    ).astype(numpy.float64)
