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
import cupy.fft as cufft


if __name__ == "__main__":
    # system_infos = commons.get_sys_info()

    events_list = [
        "GW150914-v3",
        "GW190521-v2",
    ]
    detectors_list = [
        "L1",
        "H1",
    ]
    detectors_names = ["LIGO Livingston", "LIGO Hanford"]
    # preparing for plot
    scale = 4
    fig, axis = plt.subplots(
        len(events_list),
        len(detectors_list),
        figsize=((scale + 1) * len(detectors_list), scale * len(events_list)),
    )

    sampling_rate = INT_PRECISION(CONFIG["signal.preprocessing"]["NewSamplingRate"])
    data_collection = signal.get_data_from_gwosc(
        events_list,
        detectors_list,
        verbose=True,
    )

    alpha = FLOAT_PRECISION(CONFIG["computation.parameters"]["Alpha"])

    print(f"alpha is currently set to: {alpha:.2e}")
    print(f"sampling rate is currently set to: {sampling_rate}")

    # print(f"Duration of segment extracted from gwosc: {duration:.2f} s")
    t0 = time()
    for i, event in enumerate(events_list):
        for j, detector in enumerate(detectors_list):
            time_series = signal.preprocessing(
                data_collection[event][detector]["time_serie"],
                data_collection[event][detector]["gps_time"],
            )
            signal_strain = cupy.array(time_series.value)
            time_axis = cupy.array(time_series.times.value)
            phi_range = [30, 500]
            Q_range = [2 * numpy.pi, 6 * numpy.pi]
            p_range = [0, 1]
            best_Q, best_p, coords = transform.fit_qp(
                signal_strain,
                time_axis,
                Q_range,
                p_range,
                phi_range,
            )

            print(f"Best fit for Q and p: ({best_Q:.3f}, {best_p:.3f})")

            # preparing data for scan
            signal_fft = cufft.fft(signal_strain).astype(COMPLEX_PRECISION)
            fft_frequencies = cupy.array(
                cufft.fftfreq(len(signal_strain)) * sampling_rate, dtype=FLOAT_PRECISION
            )
            time_series_duration = time_axis.max() - time_axis.min()
            phi_axis = transform.get_phi_axis(
                phi_range,
                Q_range,
                p_range,
                time_series_duration,
                sampling_rate,
            )

            tau_phi_plane = transform.qp_transform(
                signal_fft,
                fft_frequencies,
                phi_axis,
                best_Q,
                best_p,
                sampling_rate,
            )

            energy = cupy.abs(tau_phi_plane) ** 2

            # plotting region
            if (len(events_list) == 1) and (len(detectors_list) == 1):
                ax = axis
            elif len(events_list) == 1:
                ax = axis[j]
            elif len(detectors_list) == 1:
                ax = axis[i]
            else:
                ax = axis[i, j]

            ax.pcolormesh(
                time_axis.get(),
                phi_axis.get(),
                energy.get(),
                cmap="viridis",
            )
            ax.scatter(coords[0].get(), coords[1].get(), c="red")

            ax.set_yscale("log")
            if i == 0:
                ax.set_title(detectors_names[j])

    plt.tight_layout()
    plt.show()

    # =========================
    # Benchmarking

    # performing qp-transform
    # print(f"Performing benchmarks on event {event}, at {detectors_names[j]}")
    # print(
    #     benchmark(
    #         transform.fit_qp,
    #         (
    #             signal_strain,
    #             time_axis,
    #             [2 * numpy.pi, 6 * numpy.pi],
    #             [0, 1],
    #             [30, 500],
    #         ),
    #         n_repeat=5,
    #         n_warmup=1,
    #     )
    # )

    # max_idx = numpy.unravel_index(numpy.argmax(energy, axis=None), energy.shape)
    # print(f"\nMaximum energy value: {energy[max_idx]: .2f}")
    # CRs = (energy - numpy.median(energy)) / numpy.median(energy)
    # max_idx = numpy.unravel_index(numpy.argmax(CRs, axis=None), CRs.shape)
    # print(f"\nCR: {CRs[max_idx] : .2f}")
