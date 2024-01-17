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
from utils import signal, transform

# demo
import cupy, numpy, scipy, cupy.fft
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark


if __name__ == "__main__":
    # system_infos = commons.get_sys_info()

    events_list = ["GW150914-v3", "GW190521-v2"]
    detectors_list = ["L1", "H1"]
    detectors_names = ["LIGO Livingston", "LIGO Hanford"]
    # preparing for plot
    scale = 4
    fig, axis = plt.subplots(
        len(events_list),
        len(detectors_list),
        figsize=((scale + 1) * len(detectors_list), scale * len(events_list)),
    )

    duration = 20
    sampling_rate = INT_PRECISION(CONFIG["signal.parameters"]["SamplingRate"])
    data_collection = signal.get_data_from_gwosc(
        events_list,
        detectors_list,
        crop=True,
        extracted_segment_duration=duration,
        new_sampling_rate=sampling_rate,
        verbose=False,
    )

    P = FLOAT_PRECISION(0.129)
    Q = FLOAT_PRECISION(10.55)
    alpha = FLOAT_PRECISION(CONFIG["computation.parameters"]["Alpha"])

    print(f"alpha is currently set to: {alpha:.2e}")
    print(f"sampling rate is currently set to: {sampling_rate}")

    # print(f"Duration of segment extracted from gwosc: {duration:.2f} s")
    t0 = time()
    for i, event in enumerate(events_list):
        for j, detector in enumerate(detectors_list):
            data = data_collection[event][detector]
            data_sampling_rate = numpy.ceil(1.0 / data.dx).value.astype(INT_PRECISION)
            assert data_sampling_rate == sampling_rate

            duration = data.times.value[-1] - data.times.value[0]
            data_fft = scipy.fft.fft(data).astype(COMPLEX_PRECISION)
            fft_freqs = FLOAT_PRECISION(scipy.fft.fftfreq(len(data)) * sampling_rate)

            # performing qp-transform
            tau_phi_plane_cp, phi_tiles = transform.get_tau_phi_plane(
                data_fft,
                fft_freqs,
                P,
                Q,
                [20, 500],
                duration,
                sampling_rate,
                alpha,
            )

            print(f"Running a benchmark for the event {event} at {detectors_names[j]}:")
            print(
                benchmark(
                    transform.get_tau_phi_plane,
                    (
                        data_fft,
                        fft_freqs,
                        P,
                        Q,
                        [20, 500],
                        duration,
                        sampling_rate,
                        alpha,
                    ),
                    n_repeat=100,
                    n_warmup=3,
                )
            )
            energy = cupy.asnumpy(cupy.abs(tau_phi_plane_cp) ** 2)
            phis = cupy.asnumpy(phi_tiles)
            tau = data.times.value

            # plotting region
            if len(events_list) == 1:
                coords = j
            elif len(detectors_list) == 1:
                coords = i
            else:
                coords = (i, j)
            axis[coords].pcolormesh(
                tau,
                phis,
                energy,
                cmap="viridis",
            )
            axis[coords].set_yscale("log")
            if i == 0:
                axis[coords].set_title(detectors_names[j])

    print("Plotting...")
    plt.show()
    # max_idx = numpy.unravel_index(numpy.argmax(energy, axis=None), energy.shape)
    # print(f"\nMaximum energy value: {energy[max_idx]: .2f}")
    # CRs = (energy - numpy.median(energy)) / numpy.median(energy)
    # max_idx = numpy.unravel_index(numpy.argmax(CRs, axis=None), CRs.shape)
    # print(f"\nCR: {CRs[max_idx] : .2f}")
