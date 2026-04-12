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
Just run the command when you download this repo.
```
pipx install --editable . --force
```

## Command
0. Find chip of interest by eye
You can check and find a chip of interest (e.g., with a minor planet) with `ds9`.
The following command tells you the coordinate, V mag of a minor body.
```
# Itokawa on May 18, 1996
sssb_snapshot Itokawa --utc 1996-05-18
```

1. Extract chip of interest
```
# Extract `S14` of `c4d_160111_064029_opi_r_v1.fits.fz`, and save as `extracted.fits`
extract_chip_decam c4d_160111_064029_opi_r_v1.fits.fz S14 extracted.fits --overwrite
```


## Dependencies

This library is depending on `NumPy`, `SciPy`, `SEP`, `Astropy` 
and `Astroquery`.
Scripts are developed on `Python 3.7.10`, `NumPy 1.19.2`, `SciPy 1.6.1`,
`SEP 1.0.3`, `Astropy 4.2` and `Astroquery 0.4.1`.

## LICENCE
This software is released under the MIT License.
