#!/usr/bin/env python
import numpy as np
import h5py
from scipy.fftpack import fft, fftshift, fftfreq
from astropy.io import fits
import argparse
import re

def estimate_chunk_size(bytes_per_spectrum, chunk_size):
    # Find chunk size, rounded down to the nearest power of 2
    nexp = int(np.floor((np.log2(chunk_size/bytes_per_spectrum))))

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
    dtype = fp[0][groups[0]].attrs["DATATYPE"]

    if args.verbose:
        print("\n----------------------------- INPUT DATA ---------------------------------");
        print("Filename                     : %s"%fnames[0])
        print("Sample time                  : %g s"%tsamp)
        print("Number of samples            : %ld"%nsamp)
        print("Number of subbands           : %d"%nsub)
        print("Observation duration         : %g s"%(nsamp*tsamp))
        print("Center frequency             : %g MHz"%np.mean(freq*1e-6))
        print("Bandwidth                    : %g MHz"%(nsub*0.1953125))
        print("Datatype                     : %s"%np.dtype(dtype))
        
    # Set channelization and decimation
    nchan = args.nchan
    nbin = args.nsamp

    # Bytes per spectrum
    bytes_per_spectrum = nsub*nchan*4*4 # 4 values per sample at 4 bytes per value

    # Compute chunk sizes
    if np.dtype(dtype)==np.int8:
        recsize = fp[0][groups[0]+"_i2f"].attrs["STOKES_0_recsize"]
        nint = recsize//(nsub*nchan)
        nchunk = nsamp//(nint*nchan)
        chunk_size = nint*bytes_per_spectrum
    else:
        nint, chunk_size = estimate_chunk_size(bytes_per_spectrum, 500e6)
        nchunk = nsamp//(nint*nchan)
    nchunk = 2
    
    # Output file sizes
    msamp = nchunk*nint//nbin
    mchan = nsub*nchan
    mint = nint//nbin

    # Subband frequencies
    freqsub = fftshift(fftfreq(nchan, d=tsamp))
    freqmin = np.min(freqsub)+freq[0]
    freqstep = freqsub[1]-freqsub[0]
    
    # Allocate output arrays
    s0 = np.zeros(msamp*mchan).astype("float32").reshape(msamp, mchan)
    s1 = np.zeros(msamp*mchan).astype("float32").reshape(msamp, mchan)
    s2 = np.zeros(msamp*mchan).astype("float32").reshape(msamp, mchan)
    s3 = np.zeros(msamp*mchan).astype("float32").reshape(msamp, mchan)

    # Set header
    hdr = fits.Header()
    hdr["DATE-OBS"] = fp[0]["/"].attrs["OBSERVATION_START_UTC"].decode()
    hdr["MJD-OBS"] = fp[0]["/"].attrs["OBSERVATION_START_MJD"]
    hdr["SOURCE"] = fp[0]["/"].attrs["TARGETS"][0].decode()
    hdr["CRPIX1"] = 0.0
    hdr["CRPIX2"] = 0.0
    hdr["CRPIX3"] = 0.0
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = freqmin
    hdr["CRVAL3"] = 0.0
    hdr["CDELT1"] = nchan*nbin*tsamp
    hdr["CDELT2"] = freqstep
    hdr["CDELT3"] = 1.0
    hdr["CUNIT1"] = "sec"
    hdr["CUNIT2"] = "Hz"
    hdr["CTYPE1"] = "TIME"
    hdr["CTYPE2"] = "FREQ"
    hdr["CTYPE3"] = "STOKES"
    
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
        print("%d out %d"%(ichunk, nchunk))
        # Set slice to read
        imin = ichunk*nchan*nint
        imax = (ichunk+1)*nchan*nint

        # Extract values from HDF5 files
        if np.dtype(dtype)==np.int8:
            xr = fp[0][groups[0]][imin:imax].astype("float32")*fp[0][groups[0]+"_i2f"][ichunk]
            xi = fp[1][groups[1]][imin:imax].astype("float32")*fp[1][groups[1]+"_i2f"][ichunk]
            yr = fp[2][groups[2]][imin:imax].astype("float32")*fp[2][groups[2]+"_i2f"][ichunk]
            yi = fp[3][groups[3]][imin:imax].astype("float32")*fp[3][groups[3]+"_i2f"][ichunk]
        else:
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
        s1[jmin:jmax] = np.mean(((xx-yy).reshape(nint, -1, order="F")).reshape(nint//nbin, nbin, -1), axis=1)
        s2[jmin:jmax] = np.mean(((2.0*(np.real(px)*np.real(py)+np.imag(px)*np.imag(py))).reshape(nint, -1, order="F")).reshape(nint//nbin, nbin, -1), axis=1)
        s3[jmin:jmax] = np.mean(((2.0*(np.real(px)*np.imag(py)-np.imag(px)*np.real(py))).reshape(nint, -1, order="F")).reshape(nint//nbin, nbin, -1), axis=1)
       
    # Write out FITS
    hdu = fits.PrimaryHDU(data=[s0.T, s1.T, s2.T, s3.T], header=hdr)
    hdu.writeto(outfname, overwrite=True)

