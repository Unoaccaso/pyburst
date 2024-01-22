""""""
"""
Copyright (C) 2024 riccardo felicetti <https://github.com/Unoaccaso>

Created Date: Friday, January 12th 2024, 2:13:14 pm
Author: riccardo felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""
# system libs for package managing
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../"
sys.path.append(PATH_TO_MASTER)

# cuda section
from cupyx import jit
import cupy
import cupy.typing

# cpu section
import numpy
import numpy.typing
import scipy.fft
import matplotlib.pyplot as plt
from tqdm import tqdm

# fft computation
import cupyx.scipy.fft as cufft

# type hinting
from typing import Union

# errors and wanings
import warnings

# custom settings
import configparser


PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
CONFIG = configparser.ConfigParser()
CONFIG.read(PATH_TO_SETTINGS)
from utils.commons import FLOAT_PRECISION, INT_PRECISION, COMPLEX_PRECISION

BLOCK_SHAPE = (
    INT_PRECISION(CONFIG["cuda"]["BlockSizeX"]),
    INT_PRECISION(CONFIG["cuda"]["BlockSizeY"]),
    INT_PRECISION(CONFIG["cuda"]["BlockSizeZ"]),
)


# WARNING: AT THE TIME OF WRITING, JIT.RAWKERNEL IS EXPERIMENTAL!!!
@jit.rawkernel()
def Q_fft_phi_kernel(
    phi_values: cupy.typing.NDArray,
    fft_values: cupy.typing.NDArray,
    fft_frequencies: cupy.typing.NDArray,
    Q_values: cupy.typing.NDArray,
    p_value: numpy.float32 | numpy.float64,
    sampling_rate: numpy.int32 | numpy.int64,
    out_height: numpy.int32 | numpy.int64,
    out_width: numpy.int32 | numpy.int64,
    out_depth: numpy.int32 | numpy.int64,
    fft_phi_plane: cupy.typing.NDArray,
):
    # TODO: MAGARI INSERISCI LA FORMULA IN LATEX
    # supposing that the max gridsize is big enough to cover the whole out plane
    fft_freq_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    Q_idx = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    # the grid shape can (and will) in general be larger than the out plane
    if (Q_idx < out_height) and (phi_idx < out_width) and (fft_freq_idx < out_depth):
        phi = phi_values[phi_idx]
        fft_frequency = fft_frequencies[fft_freq_idx]
        fft_value = fft_values[fft_freq_idx]
        Q = Q_values[Q_idx]

        # wavelet parameters
        q_tilde = Q / cupy.sqrt(1.0 + 2.0j * Q * p_value)
        norm = (
            ((2.0 * cupy.pi / (Q**2)) ** (1.0 / 4.0))
            * cupy.sqrt(1.0 / (2.0 * cupy.pi * phi))
            * q_tilde
        )
        exponent = ((0.5 * q_tilde * (fft_frequency - phi)) / phi) ** 2

        # the actual wavelet
        wavelet = norm * cupy.exp(-exponent)

        # scalar product and normalization for fft
        fft_normalization_factor = cupy.sqrt(sampling_rate)
        fft_phi_plane[Q_idx, phi_idx, fft_freq_idx] = (
            fft_value * wavelet * fft_normalization_factor
        )


@jit.rawkernel()
def fft_phi_kernel(
    phi_values: cupy.typing.NDArray,
    fft_values: cupy.typing.NDArray,
    fft_frequencies: cupy.typing.NDArray,
    Q_value: cupy.float32 | cupy.float64,
    p_value: numpy.float32 | numpy.float64,
    sampling_rate: numpy.int32 | numpy.int64,
    out_height: numpy.int32 | numpy.int64,
    out_width: numpy.int32 | numpy.int64,
    fft_phi_plane: cupy.typing.NDArray,
):
    # TODO: MAGARI INSERISCI LA FORMULA IN LATEX
    # supposing that the max gridsize is big enough to cover the whole out plane
    phi_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    fft_freq_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y

    # the grid shape can (and will) in general be larger than the out plane
    if (phi_idx < out_height) and (fft_freq_idx < out_width):
        phi = phi_values[phi_idx]
        fft_frequency = fft_frequencies[fft_freq_idx]
        fft_value = fft_values[fft_freq_idx]

        # wavelet parameters
        q_tilde = Q_value / cupy.sqrt(1.0 + 2.0j * Q_value * p_value)
        norm = (
            ((2.0 * cupy.pi / (Q_value**2)) ** (1.0 / 4.0))
            * cupy.sqrt(1.0 / (2.0 * cupy.pi * phi))
            * q_tilde
        )
        exponent = ((0.5 * q_tilde * (fft_frequency - phi)) / phi) ** 2

        # the actual wavelet
        wavelet = norm * cupy.exp(-exponent)

        # scalar product and normalization for fft
        fft_normalization_factor = cupy.sqrt(sampling_rate)
        fft_phi_plane[phi_idx, fft_freq_idx] = (
            fft_value * wavelet * fft_normalization_factor
        )


@jit.rawkernel()
def energy_density_ker(
    tau_phi_plane: cupy.typing.NDArray,
    phi_axis: cupy.typing.NDArray,
    time_axis: cupy.typing.NDArray,
    kernel_dim: cupy.typing.NDArray,
    ker_times: cupy.typing.NDArray,
    ker_phis: cupy.typing.NDArray,
    ker_matrix: cupy.typing.NDArray,
    out_height: numpy.int32 | numpy.int64,
    out_width: numpy.int32 | numpy.int64,
    out_depth: numpy.int32 | numpy.int64,
    energy_density_GPU: cupy.typing.NDArray,
):
    # Extracting the indeces of the out matrix
    # The input matrix is bigger, it has kernel-dim more element on each border
    Q_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    tau_idx = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    if (
        (Q_idx < out_height)
        and ((phi_idx >= kernel_dim) and (phi_idx < out_width - kernel_dim - 1))
        and ((tau_idx >= kernel_dim) and (tau_idx < out_depth - kernel_dim - 1))
    ):
        out_phi_idx = phi_idx + kernel_dim
        out_tau_idx = tau_idx + kernel_dim

        # The for cycle is 1 element shorter, because the trapezoidal rule is
        # 1 / 2 * (f(x + 1) - f(x)) * dx
        # dx and dy factors are simplified whe computing the density
        energy_density = 0.0
        for i in range(2 * kernel_dim):
            ker_phi_idx = phi_idx - kernel_dim + i
            for j in range(2 * kernel_dim):
                ker_tau_idx = tau_idx - kernel_dim + i
                ker_tau_idx = out_tau_idx - kernel_dim + j
                energy_11 = (
                    cupy.abs(tau_phi_plane[Q_idx, ker_phi_idx, ker_tau_idx]) ** 2
                )
                energy_12 = (
                    cupy.abs(tau_phi_plane[Q_idx, ker_phi_idx, ker_tau_idx + 1]) ** 2
                )
                energy_21 = (
                    cupy.abs(tau_phi_plane[Q_idx, ker_phi_idx + 1, ker_tau_idx]) ** 2
                )
                energy_density += 1 / 2 * (2 * energy_11 + energy_12 + energy_21)

        energy_density_GPU[Q_idx, out_phi_idx, out_tau_idx] = energy_density


def qp_transform(
    signal_fft: cupy.typing.NDArray,
    fft_freqs: cupy.typing.NDArray,
    phi_axis: cupy.typing.NDArray,
    Q_values: cupy.typing.NDArray | float,
    p_value: numpy.float32 | numpy.float64,
    sampling_rate: numpy.int32 | numpy.int64,
):
    if isinstance(Q_values, cupy.ndarray):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        # preallocating the fft-phi plane.
        Q_fft_phi_tensor = cupy.zeros(
            (Q_values.shape[0], phi_axis.shape[0], signal_fft.shape[0]),
            dtype=COMPLEX_PRECISION,
        )
        height = INT_PRECISION(Q_fft_phi_tensor.shape[0])
        width = INT_PRECISION(Q_fft_phi_tensor.shape[1])
        depth = INT_PRECISION(Q_fft_phi_tensor.shape[2])

        # instatiating cuda variables
        grid_shape = (
            Q_fft_phi_tensor.shape[0] // BLOCK_SHAPE[0] + 1,  # X
            Q_fft_phi_tensor.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
            Q_fft_phi_tensor.shape[2] // BLOCK_SHAPE[2] + 1,  # Z
        )
        block_shape = BLOCK_SHAPE

        # here the qp transform is calculated
        Q_fft_phi_kernel[grid_shape, block_shape](
            phi_axis,
            signal_fft,
            fft_freqs,
            Q_values,
            p_value,
            sampling_rate,
            height,
            width,
            depth,
            Q_fft_phi_tensor,
        )

        # here the inverse fft is computed and the phi-tau plane is returned
        normalized_Q_tau_phi_tensor = cufft.ifft(Q_fft_phi_tensor).astype(
            COMPLEX_PRECISION
        )
        return normalized_Q_tau_phi_tensor

    elif isinstance(Q_values, cupy.float32):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        # preallocating the fft-phi plane.
        fft_phi_plane = cupy.zeros(
            (phi_axis.shape[0], signal_fft.shape[0]),
            dtype=COMPLEX_PRECISION,
        )
        height = INT_PRECISION(fft_phi_plane.shape[0])
        width = INT_PRECISION(fft_phi_plane.shape[1])

        # instatiating cuda variables
        grid_shape = (
            fft_phi_plane.shape[0] // BLOCK_SHAPE[0] + 1,  # X
            fft_phi_plane.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
        )
        block_shape = BLOCK_SHAPE

        fft_phi_kernel[grid_shape, block_shape](
            phi_axis,
            signal_fft,
            fft_freqs,
            Q_values,
            p_value,
            sampling_rate,
            height,
            width,
            fft_phi_plane,
        )

        normalized_tau_phi_plane = cufft.ifft(fft_phi_plane).astype(COMPLEX_PRECISION)
        return normalized_tau_phi_plane

    else:
        raise Exception("Q_values must be an istance of cupy.ndarray, or cupy.float32")


def get_phi_axis(
    phi_range: list[Union[float, float]],
    Q_range: list[Union[float, float]],
    p_range: list[Union[float, float]],
    time_series_duration: float,
    data_sampling_rate: int = INT_PRECISION(
        CONFIG["signal.preprocessing"]["NewSamplingRate"]
    ),
    alpha: numpy.float32
    | numpy.float64 = FLOAT_PRECISION(CONFIG["computation.parameters"]["Alpha"]),
    thr_sigmas: int = 4,
):
    # perform sanity check on input to ensure safety limits:
    # Input range should be inside [0, Nyquist_freq] with thr_sigmas
    # of upper and lower space.

    assert Q_range[0] >= 2 * numpy.pi, f"Q must be bigger then 2pi"
    # assert p_range[1] < 1 / p_range[1], f"p must be smaller then 1 / Q"
    assert p_range[0] >= 0, f"p must be grater than 0"

    lowest_acceptable_phi = (1 / time_series_duration) * (
        1
        - (thr_sigmas * numpy.sqrt(1 + ((2 * p_range[0] * Q_range[1]) ** 2)))
        / Q_range[1]
    )
    highest_acceptable_phi = (data_sampling_rate / 2) * (
        1
        + (thr_sigmas * numpy.sqrt(1 + ((2 * p_range[0] * Q_range[1]) ** 2)))
        / Q_range[1]
    )
    # Adjusting phi range
    if phi_range[0] < lowest_acceptable_phi:
        min_phi = lowest_acceptable_phi
        warnings.warn(
            f"lower bound of frequency range set to {lowest_acceptable_phi} Hz",
            RuntimeWarning,
        )
    else:
        min_phi = phi_range[0]

    if phi_range[1] > highest_acceptable_phi:
        max_phi = highest_acceptable_phi
        warnings.warn(
            f"upper bound of frequency range set to {highest_acceptable_phi} Hz",
            RuntimeWarning,
        )
    else:
        max_phi = phi_range[1]

    n_points = numpy.ceil(
        numpy.log(max_phi / min_phi)
        / numpy.log(
            1.0
            + (alpha * numpy.sqrt(1 + ((2 * Q_range[1] * p_range[0]) ** 2)))
            / Q_range[1]
        ),
    ).astype(INT_PRECISION)

    # building an array of phi values
    phi_axis = numpy.geomspace(
        min_phi,
        max_phi,
        n_points,
        dtype=FLOAT_PRECISION,
    )

    return cupy.array(phi_axis, dtype=FLOAT_PRECISION)


def get_energy_density(
    tau_phi_plane,
    phi_axis,
    tau_axis,
    kernel_dim: int = 5,
):
    # preallocating the fft-phi plane.
    N_Qs = INT_PRECISION(CONFIG["computation.parameters"]["N_q"])
    assert N_Qs == tau_phi_plane.shape[0]
    energy_density_GPU = cupy.zeros(
        (
            N_Qs,
            phi_axis.shape[0] - 2 * kernel_dim,
            tau_axis.shape[0] - 2 * kernel_dim,
        ),
        dtype=FLOAT_PRECISION,
    )
    height = INT_PRECISION(energy_density_GPU.shape[0])
    width = INT_PRECISION(energy_density_GPU.shape[1] + 2 * kernel_dim)
    depth = INT_PRECISION(energy_density_GPU.shape[2] + 2 * kernel_dim)

    ker_times = cupy.zeros(kernel_dim, dtype=FLOAT_PRECISION)
    ker_phis = cupy.zeros(kernel_dim, dtype=FLOAT_PRECISION)
    ker_matrix = cupy.zeros((kernel_dim, kernel_dim), dtype=FLOAT_PRECISION)

    # instatiating cuda variables
    grid_shape = (
        energy_density_GPU.shape[0] // BLOCK_SHAPE[0] + 1,  # X
        energy_density_GPU.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
        energy_density_GPU.shape[2] // BLOCK_SHAPE[2] + 1,  # Z
    )
    block_shape = BLOCK_SHAPE

    energy_density_ker[grid_shape, block_shape](
        tau_phi_plane,
        phi_axis,
        tau_axis,
        kernel_dim,
        ker_times,
        ker_phis,
        ker_matrix,
        height,
        width,
        depth,
        energy_density_GPU,
    )

    return energy_density_GPU


def fit_qp(
    signal_strain: cupy.typing.NDArray,
    time_axis: cupy.typing.NDArray,
    Q_range: list[Union[float, float]],
    p_range: list[Union[float, float]],
    phi_range: list[Union[float, float]],
    sampling_rate: numpy.int32
    | numpy.int64 = INT_PRECISION(CONFIG["signal.preprocessing"]["NewSamplingRate"]),
    number_of_Qs: int = INT_PRECISION(CONFIG["computation.parameters"]["N_q"]),
    number_of_ps: int = INT_PRECISION(CONFIG["computation.parameters"]["N_p"]),
    integration_kernel_size: int = INT_PRECISION(
        CONFIG["computation.parameters"]["IntegrationKernelSize"]
    ),
    energy_density_threshold: float = FLOAT_PRECISION(
        CONFIG["computation.parameters"]["EnergyDensityThreshold"]
    ),
):
    if (Q_range[1] - Q_range[0]) > 100:
        Q_values = numpy.geomspace(
            Q_range[0], Q_range[1], number_of_Qs, dtype=FLOAT_PRECISION
        )
    else:
        Q_values = numpy.linspace(
            Q_range[0], Q_range[1], number_of_Qs, dtype=FLOAT_PRECISION
        )
    if (p_range[1] - p_range[0]) > 100:
        p_values = numpy.geomspace(
            p_range[0], p_range[1], number_of_ps, dtype=FLOAT_PRECISION
        )
    else:
        p_values = numpy.linspace(
            p_range[0], p_range[1], number_of_ps, dtype=FLOAT_PRECISION
        )

    time_series_duration = time_axis.max() - time_axis.min()
    data_sampling_rate = cupy.ceil(1 / numpy.diff(time_axis)[0]).astype(FLOAT_PRECISION)

    # sanity check on sampling rate
    assert (
        data_sampling_rate == sampling_rate
    ), f"Data sampling rate is {data_sampling_rate}, while settings require {sampling_rate}. Please do preprocessing before."

    # preparing data for scan
    signal_fft = cufft.fft(signal_strain).astype(COMPLEX_PRECISION)
    fft_frequencies = cupy.array(
        cufft.fftfreq(len(signal_strain)) * sampling_rate, dtype=FLOAT_PRECISION
    )
    phi_axis = get_phi_axis(
        phi_range,
        Q_range,
        p_range,
        time_series_duration,
        sampling_rate,
    )
    Q_values_GPU = cupy.array(Q_values, dtype=FLOAT_PRECISION)
    highest_energy_density = 0
    for p_value in tqdm(p_values):
        tau_phi_plane = qp_transform(
            signal_fft,
            fft_frequencies,
            phi_axis,
            Q_values_GPU,
            p_value,
            sampling_rate,
        )
        energy_density = get_energy_density(
            tau_phi_plane,
            phi_axis,
            time_axis,
            kernel_dim=integration_kernel_size,
        )
        # plt.pcolormesh(
        #     time_axis.get()[integration_kernel_size:-(integration_kernel_size)],
        #     phi_axis.get()[integration_kernel_size:-(integration_kernel_size)],
        #     energy_density.get()[25],
        #     cmap="viridis",
        # )
        # plt.yscale("log")
        # plt.colorbar()
        # plt.show()

        loudest_pixel = cupy.unravel_index(
            cupy.argmax(energy_density), energy_density.shape
        )
        if highest_energy_density < energy_density[loudest_pixel]:
            highest_energy_density = energy_density[loudest_pixel]
            highest_energy = cupy.abs(tau_phi_plane[loudest_pixel]) ** 2
            best_fit_Q = Q_values[loudest_pixel[0].get()]
            best_fit_p = p_value
            coords = [time_axis[loudest_pixel[2]], phi_axis[loudest_pixel[1]]]

    return best_fit_Q, best_fit_p, coords
