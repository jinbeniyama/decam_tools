# DECam Tools 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[developer mail](mailto:jinbeniyama@gmail.com)

## Overview
Photometry of moving objects as well as reference stars could be done in this repository.

## Structure
```
decam_tools/
  decam_tools/
    __init__.py
    phot.py           # Photometry
    util.py           # Useful functions
  scripts/            # Executable scripts
    sssb_snapshot     # Outputs V, EL, RA, and DEC
  README.md           # This file
  pyproject.toml      # To install 
```

## Set up
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Dependencies

This library is depending on `NumPy`, `SciPy`, `SEP`, `Astropy` 
and `Astroquery`.
Scripts are developed on `Python 3.7.10`, `NumPy 1.19.2`, `SciPy 1.6.1`,
`SEP 1.0.3`, `Astropy 4.2` and `Astroquery 0.4.1`.

## LICENCE
This software is released under the MIT License.
