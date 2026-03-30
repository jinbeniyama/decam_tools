#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Useful functions
'''
import os
import numpy as np
import astropy.io.fits as fits

def extract_chip(fin, chipname, fout, overwrite=False):
    """Extract chip of interest with chipname such as N14.

    Parameters
    ----------
    fin : str
        input fits file
    chipname : str
        Chip name such as N14
    fout : str
        output fits file
    """
    src = fits.open(fin)
    hdr0 = src[0].header

    # Filter
    fltr_full = hdr0["FILTER"]
    fltr_short = fltr_full[0]
    magzpt = hdr0["MAGZPT"]
    exptime = hdr0["EXPTIME"]
    utc0 = hdr0["DATE-OBS"]

    for idx_s, s in enumerate(src):
        if idx_s == 0:
            continue
        hdr = s.header
        data = s.data

        if hdr["EXTNAME"] == chipname:
            data_out = data
            hdr_out = hdr
            idx_out = idx_s

    hdr_out["FILTER"] = fltr_short
    hdr_out["FILTER0"] = fltr_full
    hdr_out["MAGZPT"] = magzpt
    hdr_out["EXPTIME"] = exptime
    # Starting time
    hdr_out["UTC0"] = utc0

    # Save fits
    nfits = fits.PrimaryHDU(data=data_out, header=hdr_out)
    nfits.writeto(fout, overwrite=overwrite)
