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
# %%
# system
import sys
import os.path

from time import time

PATH_TO_THIS = os.path.dirname(__file__)


# custom settings
import configparser

config = configparser.ConfigParser()
config.read(PATH_TO_THIS + "/config.ini")


# custom modules
from utils import commons, signal, transform

# demo
import cupy, numpy, scipy, cupy.fft
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # system_infos = commons.get_sys_info()

    events_list = ["GW150914-v3"]
    detectors_list = ["L1"]
    duration = 20
    data_collection = signal.get_data_from_gwosc(
        events_list,
        detectors_list,
        crop=True,
        extracted_segment_duration=duration,
        new_sampling_rate=2048,
    )

    P = 0.129
    Q = 10.55

    print(f"Duration of segment extracted from gwosc: {duration:.2f} s")
    t0 = time()
    for event in events_list:
        for detector in detectors_list:
            data = data_collection[event][detector]
            sampling_rate = numpy.ceil(1.0 / data.dx).value
            duration = data.times.value[-1] - data.times.value[0]
            data_fft = scipy.fft.fft(data)
            fft_freqs = scipy.fft.fftfreq(len(data)) * sampling_rate

            # performing qp-transform

            tau_phi_plane, phi_tiles = transform.get_tau_phi_plane(
                data_fft,
                fft_freqs,
                P,
                Q,
                [20, 500],
                duration,
                sampling_rate,
            )
    energy = cupy.asnumpy(cupy.abs(tau_phi_plane)**2)
    print(f"Execution time: {time() - t0: .2e} s")
    phis = cupy.asnumpy(phi_tiles)
    tau = data.times.value
    plt.pcolormesh(tau, phis, energy, cmap='viridis', )
    plt.yscale('log')
    max_idx = numpy.unravel_index(numpy.argmax(energy, axis=None), energy.shape)
    print(f"Maximum energy value: {energy[max_idx]: .2f}")
    CRs = (energy - numpy.median(energy)) / numpy.median(energy)
    max_idx = numpy.unravel_index(numpy.argmax(CRs, axis=None), CRs.shape)
    print(f"CR: {CRs[max_idx] : .2f}")
