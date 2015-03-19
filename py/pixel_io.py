from astropy.io import fits
import numpy as np
from numpy import ma
import tools
import pandas as pd
bjd0 = 2454833


bitdesc = {
    1 : "Attitude Tweak",
    2 : "Safe Mode",
    3 : "Spacecraft is in Coarse Point",
    4 : "Spacecraft is in Earth Point",
    5 : "Reaction wheel zero crossing",
    6 : "Reaction Wheel Desaturation Event",
    7 : "Argabrightening detected across multiple channels",
    8 : "Cosmic Ray in Optimal Aperture pixel",
    9 : "Manual Exclude. The cadence was excluded because of an anomaly.",
    10 : "Reserved",
    11 : "Discontinuity corrected between this cadence and the following one",
    12 : "Impulsive outlier removed after cotrending",
    13 : "Argabrightening event on specified CCD mod/out detected",
    14 : "Cosmic Ray detected on collateral pixel row or column in optimal aperture",
    15 : "LDE parity error triggers a flag"
}

bitdesc  = pd.Series(bitdesc)

def loadPixelFile(fn, tlimits=None, bjd0=2454833, tex=None):
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
    ncad = len(cube)
    qdf = parse_bits(cube['quality'])
    sqdf = qdf.sum() # Compute the sum total of quality bits
    sqdf = pd.concat([sqdf,bitdesc],axis=1)
    sqdf.index.name = "bit"
    sqdf = sqdf.rename(columns={0:"Num cadences bit is on",1:"Bit description"})
    rembits = [1,2,3,4,5,6,7,8,9,11]
    print sqdf 

    bqual = np.array(qdf[rembits].sum(axis=1)==0)
    print "Removing %i cadences due to quality flag" % (ncad - np.sum(bqual)) 
    
    # Because masks are not always rectangles, we have to deal with nans.
    flux0 = ma.masked_invalid(cube['flux'])
    flux0 = flux0.sum(2).sum(1)
    bfinite = np.array(np.isfinite(flux0))
    print "Removing %i cadences due nans" % (ncad - np.sum(bfinite)) 
    
    btime = (cube['time'] > mintime) & (cube['time'] < maxtime)
    if type(tex)!=type(None):
        for rng in tex:
            # Include regions that are outside of time range
            brng = (cube['time'] < rng[0]) | (cube['time'] > rng[1])
            btime = btime & brng

    print "Removing %i cadences due to time limits" % (ncad - np.sum(btime)) 

    b = bqual & bfinite & btime
    print "Removing %i cadences total" % (ncad - np.sum(b)) 

    assert type(b)==type(np.ones(0)),"Boolean mask must be array"
    cube = cube[b]

    cube['time'][:] = cube['time'] + bjd0
    ret = (cube,) + ([tools.headerToDict(el.header) for el in f],)

    f.close()
    return ret

def parse_bits(quality):
    """
    Takes quality area and splits the bits up
    """

    # Integer version of quality
    nbits = 32
    fmtstr = '0%ib' % nbits
    
    # Collection of lists. One element for each bin
    sbqual = map(lambda x : list(format(x,fmtstr)  ),quality)
    nbitssave = 16
    df = pd.DataFrame(sbqual,columns=list(np.arange(nbits,0,-1)))
    df = df[range(1,nbitssave+1)]
    df = df.convert_objects(convert_numeric=True)
    return df 

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
