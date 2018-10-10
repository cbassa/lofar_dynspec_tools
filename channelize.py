#!/usr/bin/env python
import numpy as np
import h5py
from scipy.fftpack import fft, fftshift
from astropy.io import fits
import argparse
import re

def estimate_chunk_size(nsub, nval, nbyte, nchan, chunk_size):
    # Number of bytes per spectrum
    bytes_per_spectrum = nsub*nval*nbyte*nchan

    # Find chunk size, rounded down to the nearest power of 2
    nexp = int(np.floor((np.log10(chunk_size/bytes_per_spectrum)/np.log10(2.0))))

    return 2**nexp, 2**nexp*bytes_per_spectrum
               
if __name__ == "__main__":
    # Read commandline arguments
    parser = argparse.ArgumentParser(description="Channelize LOFAR complex voltage data")
    parser.add_argument("-t", "--nsamp", help="Decimate in time (power of 2; default: 128)", type=int, default=128)
    parser.add_argument("-F" , "--nchan", help="Number of channels per subband (power of 2; default 16)", type=int, default=16)
    parser.add_argument("-o", "--output", help="Output FITS file name")
    parser.add_argument("-v", "--verbose", help="Verbose mode", action="store_true")
    parser.add_argument("filename", help="HDF5 input header file name (LXXXXXX_SAPXXX_BXXX_SX_PXXX_bf.h5)")
    args = parser.parse_args()

    # Parse HDF5 file name
    m = re.search(r"L(\d+)_SAP(\d+)_B(\d+)_S(\d)_P(\d+)_bf.h5", args.filename)
    obsid, sapid, beamid, stokesid, partid = m.groups()

    # Set output filename
    if args.output==None:
        outfname="L%s_SAP%s_B%s_P%s.fits"%(obsid, sapid, beamid, partid)
    else:
        outfname = args.output

    # Format HDF5 filenames
    fnames = ["L%s_SAP%s_B%s_S%d_P%s_bf.h5"%(obsid, sapid, beamid, stokesid, partid) for stokesid in range(4)]

    # Open files
    fp = [h5py.File(fname, "r") for fname in fnames]

    # Get groups
    groups = ["/SUB_ARRAY_POINTING_%s/BEAM_%s/STOKES_%d"%(sapid, beamid, stokesid) for stokesid in range(4)]

    nsamp = fp[0][groups[0]].attrs["NOF_SAMPLES"]
    nsub = fp[0][groups[0]].attrs["NOF_SUBBANDS"]
    tsamp = fp[0]["/SUB_ARRAY_POINTING_%s/BEAM_%s/COORDINATES/COORDINATE_0"%(sapid, beamid)].attrs["INCREMENT"]
    freq = fp[0]["/SUB_ARRAY_POINTING_%s/BEAM_%s/COORDINATES/COORDINATE_1"%(sapid, beamid)].attrs["AXIS_VALUES_WORLD"]

    if args.verbose:
        print("\n----------------------------- INPUT DATA ---------------------------------");
        print("Filename                     : %s"%fnames[0])
        print("Sample time                  : %g s"%tsamp)
        print("Number of samples            : %ld"%nsamp)
        print("Number of subbands           : %d"%nsub)
        print("Observation duration         : %g s"%(nsamp*tsamp))
        print("Center frequency             : %g MHz"%np.mean(freq*1e-6))
        print("Bandwidth                    : %g MHz"%(nsub*0.1953125))
        
    # Set channelization and decimation
    nchan = args.nchan
    nbin = args.nsamp

    # Compute chunk sizes
    nint, chunk_size = estimate_chunk_size(nsub, 4, 4, nchan, 500e6)
    nchunk = nsamp//(nint*nchan)

    # Output file sizes
    msamp = nchunk*nint//nbin
    mchan = nsub*nchan
    mint = nint//nbin

    # Allocate output array
    s0 = np.zeros(msamp*mchan).astype("float32").reshape(msamp, mchan)

    if args.verbose:
        print("\n----------------------------- OUTPUT DATA --------------------------------");
        print("Sample time                  : %g s"%(nchan*nbin*tsamp))
        print("Number of samples            : %d"%msamp)
        print("Number of channels           : %d"%mchan)
        print("Channels per subband         : %d"%nchan)
        print("Time decimation factor       : %d"%nbin)
        print("Number of chunks             : %d"%nchunk)
        print("Chunk size                   : %.2f MB"%(chunk_size*1e-6))
        print("Output size                  : %.2f MB"%(4*msamp*mchan*1e-6))

    # Loop over chunks
    for ichunk in range(nchunk):
        # Set slice to read
        imin = ichunk*nchan*nint
        imax = (ichunk+1)*nchan*nint

        # Extract values from HDF5 files
        xr = fp[0][groups[0]][imin:imax]
        xi = fp[1][groups[1]][imin:imax]
        yr = fp[2][groups[2]][imin:imax]
        yi = fp[3][groups[3]][imin:imax]

        # Form complex timeseries
        cx = xr+1j*xi
        cy = yr+1j*yi

        # Fourier transform
        px = fftshift(fft(cx.reshape(nint, nchan, -1), axis=1), axes=1)
        py = fftshift(fft(cy.reshape(nint, nchan, -1), axis=1), axes=1)

        # Detect signals
        xx = np.real(px)*np.real(px)+np.imag(px)*np.imag(px)
        yy = np.real(py)*np.real(py)+np.imag(py)*np.imag(py)

        # Set slice to output
        jmin = ichunk*mint
        jmax = (ichunk+1)*mint
        
        # Form Stokes
        s0[jmin:jmax] = np.mean(((xx+yy).reshape(nint, -1, order="F")).reshape(nint//nbin, nbin, -1), axis=1)

#        s1 = (xx-yy).reshape(nint, -1, order="F")
#        s2 = (2.0*(np.real(px)*np.real(py)+np.imag(px)*np.imag(py))).reshape(nint, -1, order="F")
#        s3 = (2.0*(np.real(px)*np.imag(py)-np.imag(px)*np.real(py))).reshape(nint, -1, order="F")

    # Write out FITS
    hdu = fits.PrimaryHDU(s0.T)
    hdu.writeto(outfname, overwrite=True)
