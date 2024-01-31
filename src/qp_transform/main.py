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


# custom modules
from utils import transform, filter
from utils.preprocessing import signal, build_frequency_axis

# demo
import cupy, numpy, scipy, cupy.fft
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark
import cupy.fft as cufft
import scipy.fft


if __name__ == "__main__":
    # system_infos = commons.get_sys_info()

    events_list = [
        "GW150914-v3",
        # "GW190521-v2",
    ]
    detectors_list = [
        "L1",
        # "H1",
    ]
    detectors_names = ["LIGO Livingston", "LIGO Hanford"]
    # preparing for plot
    scale = 10
    # fig, axis = plt.subplots(
    #     len(events_list),
    #     len(detectors_list),
    #     figsize=((scale + 1) * len(detectors_list), scale * len(events_list)),
    # )

    sampling_rate = numpy.int32(CONFIG["signal.preprocessing"]["NewSamplingRate"])

    data_collection = signal.get_data_from_gwosc(
        events_list,
        detectors_list,
        verbose=True,
    )

    alpha = numpy.float32(CONFIG["computation.parameters"]["Alpha"])
    number_of_Qs: int = numpy.int32(CONFIG["computation.parameters"]["N_q"])

    print(f"alpha is currently set to: {alpha:.2e}")
    print(f"sampling rate is currently set to: {sampling_rate}")

    # print(f"Duration of segment extracted from gwosc: {duration:.2f} s")
    for i, event in enumerate(events_list):
        for j, detector in enumerate(detectors_list):
            time_series = signal.preprocessing(
                data_collection[event][detector]["time_series"],
                data_collection[event][detector]["gps_time"],
            )
            signal_strain = numpy.array(time_series.value)
            time_axis = numpy.array(time_series.times.value)
            phi_range = [30, 500]
            Q_range = [2 * numpy.pi, 6 * numpy.pi]
            p_range = [0, 1]
            (
                best_Q,
                best_p,
                coords,
            ) = transform.fit_qp(
                signal_strain,
                time_axis,
                Q_range,
                p_range,
                phi_range,
            )

            print(f"Best fit for Q and p: ({best_Q:.3f}, {best_p:.3f})")

            # preparing data for scan
            signal_fft = scipy.fft.fft(signal_strain).astype(numpy.complex64)
            fft_frequencies = (
                scipy.fft.fftfreq(len(signal_strain)) * sampling_rate
            ).astype(numpy.float32)

            time_series_duration = time_axis.max() - time_axis.min()
            phi_axis = build_frequency_axis(
                phi_range,
                best_Q,
                best_p,
                time_series_duration,
                sampling_rate,
                alpha=0.01,
            )

            signal_fft = cupy.array(signal_fft)
            fft_frequencies = cupy.array(fft_frequencies)
            phi_axis = cupy.array(phi_axis)

            tau_phi_plane = transform.qp_transform(
                signal_fft,
                fft_frequencies,
                phi_axis,
                numpy.float32(10),
                numpy.float32(0.2),
                sampling_rate,
            )
            energy = cupy.abs(tau_phi_plane) ** 2

            filtered_signal, outliers, outliers_phi, contour = filter.filter(
                cupy.array(signal_fft),
                cupy.array(fft_frequencies),
                phi_axis,
                time_axis,
                best_Q,
                best_p,
                energy,
                2,
            )

            # plt.plot(time_axis, signal_strain)
            # plt.plot(time_axis, filtered_signal.get())
            # plt.show()

            # plotting region
            # if (len(events_list) == 1) and (len(detectors_list) == 1):
            #     ax = axis
            # elif len(events_list) == 1:
            #     ax = axis[j]
            # elif len(detectors_list) == 1:
            #     ax = axis[i]
            # else:
            #     ax = axis[i, j]
            plt.figure(figsize=(20, 20))
            plt.pcolormesh(
                time_axis,
                phi_axis.get(),
                contour.get(),
                # cmap="viridis",
            )
            plt.pcolormesh(
                time_axis,
                phi_axis.get(),
                (energy).get(),
                # cmap="viridis",
                alpha=0.8,
            )
            plt.vlines(time_axis[outliers.get()], 30, 500)
            plt.hlines(outliers_phi.get(), time_axis.min(), time_axis.max(), lw=7)
            plt.xlim(1126259462 + numpy.array([0.289, 0.290]))
            plt.ylim([20, 510])
            plt.yscale("log")
            plt.show()

            # ax.plot(time_axis[borders[:, 1].get()], phi_axis[borders[:, 0]].get())

            # ax.set_xscale("log")
            # if i == 0:
            #     ax.set_title(detectors_names[j])

            # =========================
            # Benchmarking

            # performing qp-transform
            # Q_values = cupy.linspace(
            #     Q_range[0], Q_range[1], number_of_Qs, dtype=numpy.float32
            # )
            # print(f"Performing benchmarks on event {event}, at {detectors_names[j]}")
            # print(
            #     benchmark(
            #         transform.fit_qp,
            #         (
            #             signal_strain,
            #             time_axis,
            #             Q_range,
            #             p_range,
            #             phi_range,
            #         ),
            #         n_repeat=50,
            #         n_warmup=3,
            #     )
            # )

            # max_idx = numpy.unravel_index(numpy.argmax(energy, axis=None), energy.shape)
            # print(f"\nMaximum energy value: {energy[max_idx]: .2f}")
            # CRs = (energy - numpy.median(energy)) / numpy.median(energy)
            # max_idx = numpy.unravel_index(numpy.argmax(CRs, axis=None), CRs.shape)
            # print(f"\nCR: {CRs[max_idx] : .2f}")

    # plt.tight_layout()
    # plt.show()
