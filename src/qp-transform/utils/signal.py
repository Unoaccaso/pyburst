""""""
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
    segment_duration: int = INT_PRECISION(CONFIG["signal.download"]["SegmentDuration"]),
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
            gps_time - segment_duration / 2,
            gps_time + segment_duration / 2,
        )

        for detector in detectors:
            if verbose:
                print(f"Downloaing '{event_name}' data from '{detector}'.")
            signal_data = gwpy.timeseries.TimeSeries.fetch_open_data(
                detector, *gps_time_segment, verbose=verbose
            )
            out_data_dict[event_name][detector] = {}
            out_data_dict[event_name][detector]["time_serie"] = signal_data
            out_data_dict[event_name][detector]["gps_time"] = gps_time

    return out_data_dict


def preprocessing(
    time_series: gwpy.timeseries.TimeSeries,
    event_gps_time: FLOAT_PRECISION,
    crop: int = int(
        CONFIG["signal.preprocessing"]["Resample"],
    ),
    left_dt_ms: int = int(
        CONFIG["signal.preprocessing"]["LeftCropMilliseconds"],
    ),
    right_dt_ms: int = int(
        CONFIG["signal.preprocessing"]["RightCropMilliseconds"],
    ),
    resample: int = int(
        CONFIG["signal.preprocessing"]["Resample"],
    ),
    new_sampling_rate: int = int(
        CONFIG["signal.preprocessing"]["NewSamplingRate"],
    ),
    whitening: int = int(
        CONFIG["signal.preprocessing"]["Whiten"],
    ),
):
    """preprocessing

    Used to perform some preprocessing on the signal.

    Parameters
    ----------
    time_series : gwpy.timeseries.TimeSeries
        The data should be downloaded using `gwpy`
    event_gps_time : float32 | float64
        Should be obtained using `gwpy`
    crop : int, optional
        by default uses the configuration file.
    left_dt_ms : int, optional
        by default uses the configuration file.
    right_dt_ms : int, optional
        by default uses the configuration file.
    resample : int, optional
        by default uses the configuration file.
    new_sampling_rate : int, optional
        by default uses the configuration file.
    whitening : int, optional
        by default uses the configuration file.

    Returns
    -------
    gwpy.timeseries.TimeSeries
        The processed data.
    """
    signal_data = time_series
    segment_duration = INT_PRECISION(CONFIG["signal.download"]["SegmentDuration"])

    # resampling
    if resample:
        q_value = numpy.ceil(
            numpy.ceil(1.0 / (signal_data.times.value[1] - signal_data.times.value[0]))
            / new_sampling_rate
        ).astype(INT_PRECISION)
        x0 = event_gps_time - segment_duration / 2
        downsampled_data = scipy.signal.decimate(signal_data, q_value)

        signal_data = gwpy.timeseries.TimeSeries(
            downsampled_data,
            x0=x0,
            dx=1 / new_sampling_rate,
            copy=False,
        )

        data_sampling_rate = numpy.ceil(1.0 / signal_data.dx).value.astype(
            INT_PRECISION
        )

        assert data_sampling_rate == new_sampling_rate

    # whitening
    if whitening:
        white_data = signal_data.whiten()
        signal_data = white_data

    # cropping
    if crop:
        cropped_data = signal_data.crop(
            event_gps_time - left_dt_ms * 1e-3,
            event_gps_time + right_dt_ms * 1e-3,
        )
        signal_data = cropped_data

    return signal_data.astype(FLOAT_PRECISION)
