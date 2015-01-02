from astropy.io import fits
import numpy as np
from numpy import ma
import tools

bjd0 = 2454833

def loadPixelFile(fn, tlimits=None, bjd0=2454833):
    """
    Convert a Kepler Pixel-data file into time, flux, and error on flux.

    :INPUTS:
      fn : str
        Filename of pixel file to load.

      tlimits : 2-sequence
        Valid time interval (before application of 'bjd0'). 

      bjd0 : scalar
        Value that will be added to the time index. The default
        "Kepler Epoch" is 2454833.

    :OUTPUTS:
      time, datastack, data_uncertainties, mask [, FITSheaders]
    """
    # 2014-08-27 16:23 IJMC: Created
    # 2014-09-08 09:59 IJMC: Updated for K2 C0 data: masks.
    # 2014-09-08 EAP: Pass around data with record arrays

    if tlimits is None:
        mintime, maxtime = -np.inf, np.inf
    else:
        mintime, maxtime = tlimits
        if mintime is None:
            mintime = -np.inf
        if maxtime is None:
            maxtime = np.inf

    f = fits.open(fn)
    cube = f[1].data

    # Because masks are not always rectangles, we have to deal with nans.
    flux0 = ma.masked_invalid(cube['flux'])
    flux0 = flux0.sum(2).sum(1)

    index = (cube['quality']==0) & np.isfinite(flux0) \
            & (cube['time'] > mintime) & (cube['time'] < maxtime) 

    cube = cube[index]
    cube['time'][:] = cube['time'] + bjd0

    ret = (cube,) + ([tools.headerToDict(el.header) for el in f],)
        
    f.close()
    return ret



def loadPRF(**kw):
    """Load a Kepler PRF appropriate for the specified location.

    :INPUTS:
      file : str
        Name of a Kepler pixel target file. The headers should contain
        all necessary data.  Otherwise, you need to input module,
        output, and coordinate values.

      module : int
        The CCD module of the detector used for these
        observations. Any of 2-24 (inclusive), excepting 5 & 21.

      output : int
        The CCD output used for these observations. Any of 1-4
        (inclusive).

      loc : 2-sequence of ints
        Location of target on the CCD.  This would correspond to
        (CRVAL1P, CRVAL2P) in the FITS header.
        
      _prfpath : str
        Path of the Kepler PRF files (available from
        http://archive.stsci.edu/kepler/fpc.html). Default is
        '~/proj/transit/kepler/prf/'

    :RETURNS:
      (prf, sampling)

    :EXAMPLE:
      ::
    
       import k2
       prf, sampling = k2.loadPRF(file=kplr060018142-2014044044430_lpd-targ.fits)

    """
    # 2014-09-03 17:50 IJMC: Created

    # Parse inputs:
    if 'file' in kw:
        file = kw['file']
    else:
        file = None
    if 'module' in kw:
        module = kw['module']
    else:
        module = None
    if 'output' in kw:
        output = kw['output']
    else:
        output = None
    if 'loc' in kw:
        xcen, ycen = kw['loc']
    else:
        loc = None
    if '_prfpath' in kw:
        _prfpath = kw['_prfpath']
    else:
        _prfpath = '' + prfpath()

    if file is not None:
        f = fits.open(file, mode='readonly')
        module = f[0].header['module']
        output = f[0].header['output']
        xcen = f[2].header['crval1p']
        ycen = f[2].header['crval2p']
        f.close()

    # Load the PRF FITS file:
    f = fits.open(_prfpath + 'kplr%02i.%i_2011265_prf.fits' % (module, output), mode='readonly')

    # Determine which PRF location to load (there are 5)
    #x0s = np.array([12, 12, 1111, 1111, 549.5])
    #y0s = np.array([20, 1043, 1043, 20, 511.5])
    x0s = np.array([el.header['crval1p'] for el in f[1:]])
    y0s = np.array([el.header['crval2p'] for el in f[1:]])
    dist = np.sqrt((x0s - xcen)**2 + (y0s - ycen)**2)
    best3 = (dist <= np.sort(dist)[3]).nonzero()[0][0:3]
    prfWeights = 1./(dist[best3] + 1)
    prfWeights /= prfWeights.sum()

    # Construct the appropriately-weighted PRF:
    prf = 0
    for ii in range(3):
        prf += prfWeights[ii] * f[1+best3[ii]].data

    sampling = 1./f[1].header['cdelt1p']
    f.close()

    return prf, sampling
