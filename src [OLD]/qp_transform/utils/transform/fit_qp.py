"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:41:32 am
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
# self include
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# cpu side
import numpy
import scipy.fft

# gpu side
import cupy
import cupy.fft as cufft

# type hinting
import cupy.typing
from typing import Union

# settings
from utils.commons import CONFIG

# custom
from utils.preprocessing import build_frequency_axis
from utils.transform import qp_transform, get_energy_density

from tqdm import tqdm


def fit_qp(
    signal_strain: numpy.ndarray,
    time_axis: numpy.ndarray,
    Q_range: list[Union[numpy.float32, numpy.float32]],
    p_range: list[Union[numpy.float32, numpy.float32]],
    phi_range: list[Union[numpy.float32, numpy.float32]],
    sampling_rate: numpy.int32 = numpy.int32(
        CONFIG["signal.preprocessing"]["NewSamplingRate"]
    ),
    number_of_Qs: int = numpy.int32(CONFIG["computation.parameters"]["N_q"]),
    number_of_ps: int = numpy.int32(CONFIG["computation.parameters"]["N_p"]),
    integration_kernel_size: int = numpy.int32(
        CONFIG["computation.parameters"]["IntegrationKernelSize"],
    ),
    optimization_method: str = CONFIG["computation.parameters"]["OptimizationMethod"],
    verbose=False,
) -> list[
    Union[numpy.float32, numpy.float32, list[Union[numpy.float32, numpy.float32]]]
]:
    if (Q_range[1] - Q_range[0]) > 100:
        Q_values = numpy.geomspace(
            Q_range[0], Q_range[1], number_of_Qs, dtype=numpy.float32
        )
    else:
        Q_values = numpy.linspace(
            Q_range[0], Q_range[1], number_of_Qs, dtype=numpy.float32
        )
    if (p_range[1] - p_range[0]) > 100:
        p_values = numpy.geomspace(
            p_range[0], p_range[1], number_of_ps, dtype=numpy.float32
        )
    else:
        p_values = numpy.linspace(
            p_range[0], p_range[1], number_of_ps, dtype=numpy.float32
        )

    time_series_duration = time_axis.max() - time_axis.min()
    data_sampling_rate = numpy.ceil(1 / numpy.diff(time_axis)[0]).astype(numpy.float32)

    # sanity check on sampling rate
    assert (
        data_sampling_rate == sampling_rate
    ), f"Data sampling rate is {data_sampling_rate}, while settings require {sampling_rate}. Please do preprocessing before."

    phi_axis = build_frequency_axis(
        phi_range,
        Q_range,
        p_range,
        time_series_duration,
        sampling_rate,
    )

    # moving data to gpu
    phi_axis_GPU = cupy.array(phi_axis, dtype=numpy.float32)
    Q_values_GPU = cupy.array(Q_values, dtype=numpy.float32)
    time_axis_GPU = cupy.array(time_axis, dtype=numpy.float32)
    signal_strain_GPU = cupy.array(signal_strain, dtype=numpy.complex64)
    signal_fft_GPU = cufft.fft(signal_strain_GPU).astype(numpy.complex64)
    fft_frequencies_GPU = (cufft.fftfreq(len(signal_strain)) * sampling_rate).astype(
        numpy.float32
    )

    highest_value_qp = 0

    if verbose == True:
        iter_p = tqdm(p_values)
    else:
        iter_p = p_values

    for p_value in iter_p:
        tau_phi_plane_GPU = qp_transform(
            signal_fft_GPU,
            fft_frequencies_GPU,
            phi_axis_GPU,
            Q_values_GPU,
            p_value,
            sampling_rate,
        )

        if optimization_method == "energy_peak":
            energy_GPU = cupy.abs(tau_phi_plane_GPU) ** 2
            loudest_pixel = cupy.unravel_index(
                cupy.argmax(energy_GPU), energy_GPU.shape
            )
            highest_value_q = energy_GPU[loudest_pixel]

        elif optimization_method == "global_energy_density":
            energy_GPU = cupy.abs(tau_phi_plane_GPU) ** 2
            energy_tot = cupy.trapz(
                cupy.trapz(
                    energy_GPU,
                    time_axis_GPU,
                ),
                phi_axis_GPU,
            )
            plane = cupy.ones_like(energy_GPU)
            area = cupy.trapz(
                cupy.trapz(
                    plane,
                    time_axis_GPU,
                ),
                phi_axis_GPU,
            )
            energy_density_GPU = energy_tot / area
            loudest_pixel = cupy.unravel_index(
                cupy.argmax(energy_density_GPU), energy_density_GPU.shape
            )
            highest_value_q = energy_density_GPU[loudest_pixel]

        elif optimization_method == "local_energy_density":
            energy_density_GPU = get_energy_density(
                tau_phi_plane_GPU,
                phi_axis_GPU,
                time_axis_GPU,
                kernel_dim=integration_kernel_size,
            )
            loudest_pixel = cupy.unravel_index(
                cupy.argmax(energy_density_GPU), energy_density_GPU.shape
            )
            highest_value_q = energy_density_GPU[loudest_pixel]

        if highest_value_qp < highest_value_q:
            highest_value_qp = highest_value_q
            best_fit_Q = numpy.float32(Q_values[loudest_pixel[0].get()])
            best_fit_p = numpy.float32(p_value)
            coordinates_of_energy_dens_peak = [
                time_axis[loudest_pixel[2].get()].astype(numpy.float32),
                phi_axis[loudest_pixel[1].get()].astype(numpy.float32),
            ]

    return best_fit_Q, best_fit_p, coordinates_of_energy_dens_peak
