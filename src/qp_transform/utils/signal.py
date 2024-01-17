"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Monday, January 15th 2024, 4:06:50 pm
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
# system libs for package managing
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../"
sys.path.append(PATH_TO_MASTER)

# type hinting
from typing import Union

# signal
import gwosc.datasets
import gwpy.timeseries

# cpu
import numpy
import scipy.signal

# custom settings
import configparser

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
CONFIG = configparser.ConfigParser()
CONFIG.read(PATH_TO_SETTINGS)
from utils.commons import FLOAT_PRECISION, INT_PRECISION, COMPLEX_PRECISION


def get_data_from_gwosc(
    event_names: list[str],
    detectors: list[str],
    extracted_segment_duration: int = 20,
    crop=True,
    downsample: bool = True,
    new_sampling_rate: int = int(CONFIG["signal.parameters"]["SamplingRate"]),
    whitening: bool = True,
    verbose: bool = True,
):
    # making sure that typing is correct
    assert isinstance(event_names, list), "Ensure that event_names is a list of strings"
    assert isinstance(detectors, list), "Ensure that detectors is a list of strings"
    out_data_dict = {}
    for event_name in event_names:
        out_data_dict[event_name] = {}
        # sanity check, looking inside gwosc to make sure that event exists
        assert (
            event_name in gwosc.datasets.find_datasets()
        ), f"The event {event_name} is not in the gwosc dataset."

        gps_time = gwosc.datasets.event_gps(event_name)
        gps_time_segment = (
            gps_time - extracted_segment_duration / 2,
            gps_time + extracted_segment_duration / 2,
        )

        for detector in detectors:
            signal_data = gwpy.timeseries.TimeSeries.fetch_open_data(
                detector, *gps_time_segment, verbose=verbose
            )
            if downsample:
                q_value = numpy.ceil(
                    numpy.ceil(
                        1.0 / (signal_data.times.value[1] - signal_data.times.value[0])
                    )
                    / new_sampling_rate
                ).astype(INT_PRECISION)
                downsampled_data = scipy.signal.decimate(signal_data, q_value)
                signal_data = gwpy.timeseries.TimeSeries(
                    downsampled_data,
                    x0=gps_time_segment[0],
                    dx=1 / new_sampling_rate,
                    copy=False,
                )

            if whitening:
                white_data = signal_data.whiten()
                signal_data = white_data

            if crop:
                cropped_data = signal_data.crop(
                    gps_time
                    - int(CONFIG["computation.parameters"]["LeftCropMilliseconds"])
                    * 1e-3,
                    gps_time
                    + int(CONFIG["computation.parameters"]["RightCropMilliseconds"])
                    * 1e-3,
                )
                signal_data = cropped_data

            out_data_dict[event_name][detector] = signal_data.astype(FLOAT_PRECISION)

    return out_data_dict
