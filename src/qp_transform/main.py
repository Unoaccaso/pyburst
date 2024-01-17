# %%
"""
Copyright (C) 2024 riccardo felicetti <https://github.com/Unoaccaso>

Created Date: Friday, January 12th 2024, 10:59:29 am
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
# system
import sys
import os.path

from time import time

PATH_TO_THIS = os.path.dirname(__file__)


# custom settings
import configparser

CONFIG = configparser.ConfigParser()
CONFIG.read(PATH_TO_THIS + "/config.ini")
from utils.commons import FLOAT_PRECISION, INT_PRECISION, COMPLEX_PRECISION


# custom modules
from utils import commons, signal, transform

# demo
import cupy, numpy, scipy, cupy.fft
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark


if __name__ == "__main__":
    # system_infos = commons.get_sys_info()

    events_list = ["GW150914-v3"]
    detectors_list = ["L1"]
    duration = 20
    sampling_rate = INT_PRECISION(CONFIG["signal.parameters"]["SamplingRate"])
    data_collection = signal.get_data_from_gwosc(
        events_list,
        detectors_list,
        crop=True,
        extracted_segment_duration=duration,
        new_sampling_rate=sampling_rate,
    )

    P = FLOAT_PRECISION(0.129)
    Q = FLOAT_PRECISION(10.55)

    # print(f"Duration of segment extracted from gwosc: {duration:.2f} s")
    t0 = time()
    for event in events_list:
        for detector in detectors_list:
            data = data_collection[event][detector]
            data_sampling_rate = numpy.ceil(1.0 / data.dx).value.astype(INT_PRECISION)
            assert data_sampling_rate == sampling_rate

            duration = data.times.value[-1] - data.times.value[0]
            data_fft = scipy.fft.fft(data).astype(COMPLEX_PRECISION)
            fft_freqs = FLOAT_PRECISION(scipy.fft.fftfreq(len(data)) * sampling_rate)

            alpha = FLOAT_PRECISION(CONFIG["computation.parameters"]["Alpha"])
            # performing qp-transform
            tau_phi_plane_cp, phi_tiles = transform.get_tau_phi_plane_cp(
                data_fft,
                fft_freqs,
                alpha,
                P,
                Q,
                [20, 500],
                duration,
                sampling_rate,
            )
            # performing qp-transform
            tau_phi_plane_cuda, phi_tiles = transform.get_tau_phi_plane_cuda(
                data_fft,
                fft_freqs,
                alpha,
                P,
                Q,
                [20, 500],
                duration,
                sampling_rate,
            )

    print("Running a benchmark on custom kernel:")
    print(
        benchmark(
            transform.get_tau_phi_plane_cuda,
            (
                data_fft,
                fft_freqs,
                alpha,
                P,
                Q,
                [20, 500],
                duration,
                sampling_rate,
            ),
            n_repeat=100,
            n_warmup=2,
        )
    )
    print("Running a benchmark on cupy version:")
    print(
        benchmark(
            transform.get_tau_phi_plane_cp,
            (
                data_fft,
                fft_freqs,
                alpha,
                P,
                Q,
                [20, 500],
                duration,
                sampling_rate,
            ),
            n_repeat=100,
            n_warmup=2,
        )
    )
    energy_cp = cupy.asnumpy(cupy.abs(tau_phi_plane_cp) ** 2)
    energy_cuda = cupy.asnumpy(cupy.abs(tau_phi_plane_cuda) ** 2)
    # print(f"Execution time: {time() - t0: .2e} s")
    phis = cupy.asnumpy(phi_tiles)
    tau = data.times.value

    fig, axis = plt.subplots(1, 2, figsize=(25, 10))
    axis[0].pcolormesh(
        tau,
        phis,
        energy_cp,
        cmap="viridis",
    )
    axis[0].set_yscale("log")
    axis[1].pcolormesh(
        tau,
        phis,
        energy_cuda,
        cmap="viridis",
    )
    axis[1].set_yscale("log")
    # max_idx = numpy.unravel_index(numpy.argmax(energy, axis=None), energy.shape)
    # print(f"\nMaximum energy value: {energy[max_idx]: .2f}")
    # CRs = (energy - numpy.median(energy)) / numpy.median(energy)
    # max_idx = numpy.unravel_index(numpy.argmax(CRs, axis=None), CRs.shape)
    # print(f"\nCR: {CRs[max_idx] : .2f}")

    # print(energy.shape)
