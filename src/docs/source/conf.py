# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "timeserie"
copyright = "2024, Riccardo Felicetti"
author = "Riccardo Felicetti"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

# Aggiungi la directory del tuo pacchetto al percorso di ricerca dei moduli Python
sys.path.insert(0, os.path.abspath("../.."))

# Imposta la modalità di estrazione delle docstring
autodoc_typehints = "description"

# Imposta la lista di estensioni
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"

# Configura la directory dei moduli da documentare
autodoc_mock_imports = (
    []
)  # Se necessario, puoi aggiungere qui le dipendenze esterne che non possono essere installate durante la generazione della documentazione.

# Configura la sorgente del codice da documentare
source_suffix = ".rst"
master_doc = "index"

# Imposta la directory in cui verranno generati i file .rst per i moduli
# (questo dovrebbe essere relativo alla directory "docs")
autodoc_output_dir = "modules"

# Se vuoi che Sphinx includa anche i docstring degli oggetti di tipo "__init__", impostalo su True
autoclass_content = "both"
