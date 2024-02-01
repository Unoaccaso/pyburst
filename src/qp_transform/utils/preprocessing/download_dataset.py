"""
Copyright (C) 2024 riccardo felicetti <https://github.com/Unoaccaso>

Created Date: Thursday, February 1st 2024, 10:08:00 am
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
import warnings

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# custom settings
from utils.commons import CONFIG, DETECTORS

import gwosc.datasets
from gwpy.timeseries import TimeSeries

import numpy

DETECTORS = ["L1", "H1", "V1"]


def download_all():
    timeseries_duration = numpy.int32(CONFIG["signal.preprocessing"]["SegmentDuration"])
    save_path = CONFIG["signal.preprocessing"]["DatasetSavePath"]
    for detector in DETECTORS:
        events = gwosc.datasets.find_datasets(detector=detector)
        for event in events:
            file_save_path = os.path.join(save_path, detector, event)
            fetch_and_download(
                detector,
                event,
                timeseries_duration,
                file_save_path,
                verbose=True,
                max_attempts=100,
            )


def fetch_and_download(
    detector,
    event_name,
    timeseries_duration,
    file_save_path,
    verbose,
    attempt=1,
    max_attempts=100,
):
    try:
        event_gps_time = gwosc.datasets.event_gps(event=event_name)
        start_time = event_gps_time - timeseries_duration / 2
        end_time = event_gps_time + timeseries_duration / 2

        if not os.path.exists(file_save_path):
            timeserie = TimeSeries.fetch_open_data(
                detector, start_time, end_time, verbose=verbose
            )
            os.makedirs(file_save_path)
            timeserie.write(
                file_save_path + f"/timeserie.hdf5",
            )
    except ValueError as err:
        if attempt < max_attempts:
            warnings.warn(
                f"Failed {attempt} / {max_attempts} at downloading: retrying..."
            )
            return fetch_and_download(
                detector,
                event_name,
                timeseries_duration,
                file_save_path,
                verbose,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
        else:
            warnings.warn(f"Failed too many attempts: skipping...")
            return


if __name__ == "__main__":
    download_all()
