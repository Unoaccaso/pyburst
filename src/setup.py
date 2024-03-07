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


import os
from setuptools import setup, find_packages


# Funzione per ottenere la versione dal tag di Git
def get_version():
    try:
        tag = os.getenv(
            "GITHUB_REF", ""
        )  # Ottieni il tag di Git dalla variabile di ambiente GITHUB_REF
        if tag.startswith("refs/tags/"):
            return tag[len("refs/tags/") :]  # Rimuovi il prefisso "refs/tags/" dal tag
    except Exception as e:
        print(f"Errore durante l'ottenimento della versione dal tag di Git: {e}")
    return "1.0.0"  # Se non riesci a ottenere il tag, usa una versione predefinita


# Dipendenze opzionali per verificare la disponibilità di CUDA
extras_require = {
    "cuda": ["cupy-cuda"],
}

setup(
    name="timeserie",
    version=get_version(),
    description="A static structure to work with time series from GW detectors",
    author="Riccardo Felicetti",
    author_email="felicettiriccardo1@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "polars",
        "astropy",
        "gwpy",
        "gwosc",
        "scipy",
        "tabular",
        "zarr",
        "dask",
        "sphinx",
    ],
    extras_require=extras_require,  # Specifica le dipendenze opzionali
)
