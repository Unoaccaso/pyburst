# Copyright (C) 2024 unoaccaso <https://github.com/Unoaccaso>
#
# Created Date: Thursday, February 22nd 2024, 10:07:32 am
# Author: unoaccaso
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, version 3. This program is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https: //www.gnu.org/licenses/>.
from dataclasses import dataclass
from typing import Type
import pathlib


@dataclass
class PathBase:
    path: str

    def _check_path_firm(self, path: str):
        raise NotImplementedError

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise ValueError(f"path must be a string")
        self._check_path_firm(self.path)
        self.path = pathlib.Path(self.path)


class BackendBase:
    def open_data(self, path: Type[PathBase]):
        return NotImplementedError
