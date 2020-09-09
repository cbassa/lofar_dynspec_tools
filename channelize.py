#!/usr/bin/env python
"""Channelize LOFAR complex voltage data"""

import argparse
import re
import time
import os.path
import os
import numpy as np
import h5py
from scipy.fftpack import fft, fftshift, fftfreq
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x):
        """Dummy function for if tqdm is not present"""
        return x


def estimate_chunk_size(bytes_per_spectrum, chunk_size):
    """Find chunk size, rounded down to the nearest power of 2"""
    nexp = int(np.floor((np.log2(chunk_size / bytes_per_spectrum))))

    return 2**nexp, 2**nexp * bytes_per_spectrum


def find_stokes_group(h5):
    """Find first SAP"""
    sapid = [key for key in h5.keys() if "SUB_ARRAY_POINTING" in key][0]

    # Find first BEAM in first SAP
    beamid = [key for key in h5[sapid].keys() if "BEAM" in key][0]

    # Find first STOKES in first BEAM of first SAP
    stokesid = [
        key for key in h5[sapid + "/" + beamid].keys() if "STOKES" in key
    ][0]

    return str(sapid + "/" + beamid + "/" + stokesid)


def find_beam_group(h5):
    """Find first SAP"""
    sapid = [key for key in h5.keys() if "SUB_ARRAY_POINTING" in key][0]

    # Find first BEAM in first SAP
    beamid = [key for key in h5[sapid].keys() if "BEAM" in key][0]

    return str(sapid + "/" + beamid)


def channelize(filename, nchan=16, nbin=128, nof_samples=0, start=0, total=None, verbose=False,
               frequency=None, bandwidth=None, stokesi=False):
    """
    Extract data from HDF5 and channelize

    Args:
        nchan (int): number of channels per subband (power of 2; default 16)
        nbin (int): decimate in time (power of 2; default 128)
        stokesi (bool): Stokes I only
        verbose (bool): verbose mode
        nof_samples: override number of samples stored in HDF5
        start (float): start processing t=tstart (seconds; default 0)
        total (float): process only t=total seconds (seconds; default all)
        frequency (float): center frequency to extract (Hz; default none
        bandwidth (float): bandwidth to extract (Hz; default all)

    Returns:
        data (np.array), header (as fits object)
    """
    # Parse HDF5 file name
    match = re.search(r"L(\d+)_SAP(\d+)_B(\d+)_S(\d)_P(\d+)_bf.h5", filename)
    obsid, sapid, beamid, _, partid = match.groups()
    inputdir = os.path.dirname(filename)

    # Format HDF5 filenames
    fnames = [
        os.path.join(
            inputdir, "L%s_SAP%s_B%s_S%d_P%s_bf.h5" %
            (obsid, sapid, beamid, stokesid, partid)) for stokesid in range(4)
    ]

    currentdir = os.getcwd()
    if inputdir != "":
        os.chdir(inputdir)

    # Open files
    fp = [h5py.File(fname, "r") for fname in fnames]

    # Get groups
    groups = [find_stokes_group(fptr) for fptr in fp]
    beamgroups = [find_beam_group(fptr) for fptr in fp]

    # Get parameters
    if nof_samples == 0:
        nsamp = fp[0][groups[0]].attrs["NOF_SAMPLES"]
    else:
        nsamp = nof_samples

    # Extract data parameters
    nsub = fp[0][groups[0]].attrs["NOF_SUBBANDS"]
    tsamp = fp[0]["/%s/COORDINATES/COORDINATE_0" %
                  beamgroups[0]].attrs["INCREMENT"]
    freq = fp[0]["/%s/COORDINATES/COORDINATE_1" %
                 beamgroups[0]].attrs["AXIS_VALUES_WORLD"]
    dtype = fp[0][groups[0]].attrs["DATATYPE"]

    # Set time selection
    if isinstance(start, Time):
        obs_start = Time(fp[0]["/"].attrs["OBSERVATION_START_MJD"], format='mjd')
        start = (start - obs_start).to(u.s).value

    if start > 0:
        istart = int(start / tsamp)
    else:
        istart = 0
    if total is not None:
        iend = int((start + total) / tsamp)
    else:
        iend = nsamp
    nsamp = iend - istart

    # Bytes per spectrum: 4 values per sample at 4 bytes per value
    bytes_per_spectrum = nsub * nchan * 4 * 4

    # Compute chunk sizes
    if np.dtype(dtype) == np.int8:
        recsize = fp[0][groups[0] + "_i2f"].attrs["STOKES_0_recsize"]
        nint = recsize // (nsub * nchan)
        nchunk = nsamp // (nint * nchan) + 1
        chunk_size = nint * bytes_per_spectrum
    else:
        nint, chunk_size = estimate_chunk_size(bytes_per_spectrum, 500e6)
        nchunk = nsamp // (nint * nchan) + 1

    # Output file sizes
    msamp = nchunk * nint // nbin
    mchan = nsub * nchan
    mint = nint // nbin

    # Subband frequencies
    freqsub = fftshift(fftfreq(nchan, d=tsamp))
    freqmin = np.min(freqsub) + freq[0]
    freqstep = freqsub[1] - freqsub[0]
    freqs = freqmin + np.arange(mchan) * freqstep

    # Set frequency selection
    if frequency is not None and bandwidth is not None:
        fmin = frequency - 0.5 * bandwidth
        fmax = frequency + 0.5 * bandwidth
        cfreq = (freqs >= fmin) & (freqs < fmax)
    else:
        cfreq = np.ones(len(freqs), dtype="bool")
    mchan = np.sum(cfreq)
    freqmin = np.min(freqs[cfreq])

    if verbose:
        print(
            "\n----------------------------- INPUT DATA ----------------------"
        )
        print("Filename                     : %s" % fnames[0])
        print("Sample time                  : %g s" % tsamp)
        print("Number of samples            : %ld" % nsamp)
        print("Number of subbands           : %d" % nsub)
        print("Observation duration         : %g s" % (nsamp * tsamp))
        print("Center frequency             : %g MHz" % np.mean(freq * 1e-6))
        print("Bandwidth                    : %g MHz" % (nsub * 0.1953125))
        print("Datatype                     : %s" % np.dtype(dtype))

    # Allocate output arrays
    s0 = np.zeros(msamp * mchan).astype("float32").reshape(msamp, mchan)
    if not stokesi:
        s1 = np.zeros(msamp * mchan).astype("float32").reshape(msamp, mchan)
        s2 = np.zeros(msamp * mchan).astype("float32").reshape(msamp, mchan)
        s3 = np.zeros(msamp * mchan).astype("float32").reshape(msamp, mchan)

    # Set header
    hdr = fits.Header()
    hdr["DATE-OBS"] = fp[0]["/"].attrs["OBSERVATION_START_UTC"].decode()
    hdr["MJD-OBS"] = fp[0]["/"].attrs["OBSERVATION_START_MJD"]
    hdr["SOURCE"] = fp[0]["/"].attrs["TARGETS"][0].decode()
    hdr["CRPIX1"] = 0.0
    hdr["CRPIX2"] = 0.0
    hdr["CRPIX3"] = 0.0
    hdr["CRVAL1"] = start
    hdr["CRVAL2"] = freqmin
    hdr["CRVAL3"] = 0.0
    hdr["CDELT1"] = nchan * nbin * tsamp
    hdr["CDELT2"] = freqstep
    hdr["CDELT3"] = 1.0
    hdr["CUNIT1"] = "sec"
    hdr["CUNIT2"] = "Hz"
    hdr["CTYPE1"] = "TIME"
    hdr["CTYPE2"] = "FREQ"
    hdr["CTYPE3"] = "STOKES"

    if verbose:
        print(
            "\n----------------------------- OUTPUT DATA ---------------------"
        )
        print("Sample time                  : %g s" % (nchan * nbin * tsamp))
        print("Number of samples            : %d" % msamp)
        print("Number of channels           : %d" % mchan)
        print("Channels per subband         : %d" % nchan)
        print("Time decimation factor       : %d" % nbin)
        print("Number of chunks             : %d" % nchunk)
        print("Chunk size                   : %.2f MB" % (chunk_size * 1e-6))
        if not stokesi:
            print("Output size                  : %.2f MB" %
                  (4 * msamp * mchan * 1e-6))
        else:
            print("Output size                  : %.2f MB" %
                  (msamp * mchan * 1e-6))

    # Loop over chunks
    for ichunk in tqdm(range(nchunk)):
        # Set slice to read
        imin = ichunk * nchan * nint + istart
        imax = (ichunk + 1) * nchan * nint + istart

        # Extract values from HDF5 files
        t0 = time.time()
        if np.dtype(dtype) == np.int8:
            xr = fp[0][groups[0]][imin:imax].astype("float32") * fp[0][
                groups[0] + "_i2f"][ichunk]
            xi = fp[1][groups[1]][imin:imax].astype("float32") * fp[1][
                groups[1] + "_i2f"][ichunk]
            yr = fp[2][groups[2]][imin:imax].astype("float32") * fp[2][
                groups[2] + "_i2f"][ichunk]
            yi = fp[3][groups[3]][imin:imax].astype("float32") * fp[3][
                groups[3] + "_i2f"][ichunk]
        else:
            xr = fp[0][groups[0]][imin:imax]
            xi = fp[1][groups[1]][imin:imax]
            yr = fp[2][groups[2]][imin:imax]
            yi = fp[3][groups[3]][imin:imax]
        tread = time.time() - t0

        # Use actual size to deal with last chunk
        nint_act = xr.shape[0] // nchan
        mint_act = nint_act // nbin

        # Form complex timeseries
        cx = xr + 1j * xi
        cy = yr + 1j * yi

        # Fourier transform
        t0 = time.time()
        px = fftshift(fft(cx.reshape(nint_act, nchan, -1), axis=1), axes=1)
        py = fftshift(fft(cy.reshape(nint_act, nchan, -1), axis=1), axes=1)

        # Detect signals
        xx = np.real(px) * np.real(px) + np.imag(px) * np.imag(px)
        yy = np.real(py) * np.real(py) + np.imag(py) * np.imag(py)

        # Set slice to output
        jmin = ichunk * mint_act
        jmax = (ichunk + 1) * mint_act

        # Form Stokes
        # (IAU/IEEE convention [van Straten et al. 2010, PASP 27, 104])
        s0[jmin:jmax] = np.mean(
            ((xx + yy).reshape(nint_act, -1,
                               order="F")).reshape(mint_act, nbin, -1),
            axis=1)[:, cfreq]
        if not stokesi:
            s1[jmin:jmax] = np.mean(
                ((xx - yy).reshape(nint_act, -1,
                                   order="F")).reshape(mint_act, nbin, -1),
                axis=1)[:, cfreq]
            s2[jmin:jmax] = np.mean(
                ((2.0 * (np.real(px) * np.real(py) + np.imag(px) * np.imag(py))
                  ).reshape(nint_act, -1,
                            order="F")).reshape(mint_act, nbin, -1),
                axis=1)[:, cfreq]
            s3[jmin:jmax] = np.mean(
                ((2.0 * (np.imag(px) * np.real(py) - np.real(px) * np.imag(py))
                  ).reshape(nint_act, -1,
                            order="F")).reshape(mint_act, nbin, -1),
                axis=1)[:, cfreq]
        tproc = time.time() - t0

    os.chdir(currentdir)

    if stokesi:
        return np.array([s0.T]), hdr
    else:
        return np.array([s0.T, s1.T, s2.T, s3.T]), hdr


def main():
    """Read command line arguments"""
    parser = argparse.ArgumentParser(
        description="Channelize LOFAR complex voltage data")
    parser.add_argument("-t",
                        "--nsamp",
                        help="Decimate in time (power of 2; default: 128)",
                        type=int,
                        default=128)
    parser.add_argument(
        "-F",
        "--nchan",
        help="Number of channels per subband (power of 2; default 16)",
        type=int,
        default=16)
    parser.add_argument("-o", "--output", help="Output FITS file name")
    parser.add_argument("-I",
                        "--stokesi",
                        help="Stokes I only",
                        action="store_true")
    parser.add_argument("-v",
                        "--verbose",
                        help="Verbose mode",
                        action="store_true")
    parser.add_argument("-n",
                        "--nof_samples",
                        help="Additional NOF_SAMPLES",
                        type=int,
                        default=0)
    parser.add_argument("-S",
                        "--start",
                        help="Start processing t=start (seconds; default 0)",
                        type=float,
                        default=0.0)
    parser.add_argument(
        "-T",
        "--total",
        help="Process only t=total seconds (seconds; default all)",
        type=float,
        default=None)
    parser.add_argument("-f",
                        "--frequency",
                        help="Center frequency to extract (Hz; default none)",
                        type=float,
                        default=None)
    parser.add_argument("-b",
                        "--bandwidth",
                        help="Bandwidth to extract (Hz; default all)",
                        type=float,
                        default=None)
    parser.add_argument(
        "filename",
        help="HDF5 input header file name (LXXXXXX_SAPXXX_BXXX_SX_PXXX_bf.h5)")
    args = parser.parse_args()

    data, hdr = channelize(args.filename, nchan=args.nchan, nbin=args.nsamp,
                           nof_samples=args.nof_samples, start=args.start,
                           total=args.total, verbose=args.verbose, frequency=args.frequency,
                           bandwidth=args.bandwidth, stokesi=args.stokesi)

    # Set output filename
    if args.output is None:
        match = re.search(r"L(\d+)_SAP(\d+)_B(\d+)_S(\d)_P(\d+)_bf.h5", args.filename)
        obsid, sapid, beamid, _, partid = match.groups()
        outfname = "L%s_SAP%s_B%s_P%s.fits" % (obsid, sapid, beamid, partid)
    else:
        outfname = args.output

    # Write out FITS
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(outfname, overwrite=True)


if __name__ == "__main__":
    main()
