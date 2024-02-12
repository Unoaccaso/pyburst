"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Friday, February 9th 2024, 8:55:52 pm
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

"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Friday, February 9th 2024, 8:55:52 pm
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
import sys

from ._typing import type_check
from ._caching import LRUCache

from warnings import warn


from gwpy.timeseries import TimeSeries
import gwosc.datasets
import os
from concurrent.futures import ThreadPoolExecutor


class EventDataLoader:
    _AVAILABLE_SOURCES = {
        "remote_open": "_fetch_remote",
        "local": "_fetch_local",
    }

    _CACHED_DATA = LRUCache()

    @classmethod
    @type_check(classmethod=True)
    def _validate_source(cls, source: str):
        if source not in cls._AVAILABLE_SOURCES.keys():
            raise NotImplementedError(f"{source} is not a valid source.")

    @classmethod
    @type_check(classmethod=True)
    def _fetch_remote(
        cls,
        event_name: str,
        detector_name: str,
        duration: float,
        sample_rate: int,
        url: str,
        format: str,
        max_attempts: int,
        this_attempt: int = 1,
        verbose: bool = False,
    ):
        try:
            event_gps_time = gwosc.datasets.event_gps(event_name)
            start_time = event_gps_time - duration / 2
            end_time = event_gps_time + duration / 2
            signal = TimeSeries.fetch_open_data(
                detector_name,
                start_time,
                end_time,
                sample_rate,
                format=format,
                verbose=verbose,
            )
            result = {
                "event_name": event_name,
                "detector_name": detector_name,
                "time_series": signal,
                "gps_time": event_gps_time,
                "duration": duration,
            }
            return result
        except:
            if this_attempt < max_attempts:
                warn(
                    f"Failed downloading {this_attempt}/{max_attempts} times, retrying...",
                    ResourceWarning,
                )
                cls._fetch_remote(
                    event_name,
                    detector_name,
                    duration,
                    sample_rate,
                    format,
                    max_attempts,
                    this_attempt + 1,
                )
            else:
                raise ConnectionError(
                    f"Failed downloading too many times ({this_attempt})"
                )

    @classmethod
    @type_check(classmethod=True)
    def _fetch_local(
        cls,
        event_name: str,
        detector_name: str,
        duration: float,
        sample_rate: int,
        url: str,
        format: str,
        max_attempts: int,
        this_attempt: int = 1,
        verbose: bool = False,
    ): ...

    @classmethod
    @type_check(classmethod=True)
    def _save_event(cls, event_data: dict, save_path: str, fmt: str):
        event_name = event_data["event_name"]
        detector_name = event_data["detector_name"]
        gps_time = event_data["gps_time"]
        timeseries = event_data["time_series"]

        file_path = os.path.join(save_path, event_name, detector_name)
        if not os.path.exists(file_path):
            os.makedirs(
                file_path,
            )
        timeseries.write(
            file_path + f"/{event_name}_{detector_name}_{gps_time}_.{fmt}",
            format=fmt,
            overwrite=True,
        )

    @classmethod
    @type_check(classmethod=True)
    def save_event_data(cls, data_dict: dict, save_path: str, fmt: str = "hdf5"):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    cls._save_event,
                    data_dict[event_name][detector_name],
                    save_path,
                    fmt,
                )
                for event_name in data_dict.keys()
                for detector_name in data_dict[event_name]
            ]
            for future in futures:
                future.result()

    @classmethod
    @type_check(classmethod=True)
    def get_event_data(
        cls,
        event_names: list[str],
        detector_names: list[str],
        duration: float = 50.0,
        source: str = "remote_open",
        url: str = "",
        sample_rate: int = 4096,
        format: str = "hdf5",
        max_attempts: int = 100,
        cache_results: bool = True,
        force_cache_overwrite: bool = False,
        verbose: bool = False,
    ):
        # checking if source is supported
        cls._validate_source(source)

        # getting the correct fetch function depending on input
        _fetch_function = getattr(cls, cls._AVAILABLE_SOURCES[source])

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _fetch_function,
                    event_name,
                    detector_name,
                    duration,
                    sample_rate,
                    url,
                    format,
                    max_attempts,
                    1,
                    verbose,
                )
                for event_name in event_names
                for detector_name in detector_names
                if event_name not in cls._CACHED_DATA
                or detector_name not in cls._CACHED_DATA[event_name]
                or force_cache_overwrite
                or cls._CACHED_DATA[event_name][detector_name]["duration"] != duration
            ]

            for future in futures:
                result = future.result()
                event_name, detector_name = (
                    result["event_name"],
                    result["detector_name"],
                )
                cls._CACHED_DATA.setdefault(event_name, {})[detector_name] = result

        out_var = dict(cls._CACHED_DATA)
        if not cache_results:
            cls._CACHED_DATA.clear()

        return out_var
