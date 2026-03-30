# Useful functions
import os
import numpy as np
import astropy.io.fits as fits

def extract_chip(fin, chipname, fout):
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
    for idx_s, s in enumerate(src):
        if idx_s == 0:
            continue
        hdr = s.header
        data = s.data

        if hdr["EXTNAME"] == chipname:
            data_out = data
            hdr_out = hdr
            idx_out = idx_s

    # Save fits
    nfits = fits.PrimaryHDU(data=data_out, header=hdr_out)
    nfits.writeto(fout)
