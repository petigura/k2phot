"""
Module with code for i/o using K2 pixel files
"""
from astropy.io import fits
from astropy import wcs

import numpy as np
from numpy import ma
import pandas as pd
import k2_catalogs

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
    cube = np.array(cube) 
    ncad = len(cube)
    qdf = parse_bits(cube['QUALITY'])
    sqdf = qdf.sum() # Compute the sum total of quality bits
    sqdf = pd.concat([sqdf,bitdesc],axis=1)
    sqdf.index.name = "bit"
    sqdf = sqdf.rename(columns={0:"Num cadences bit is on",1:"Bit description"})
    rembits = [1,2,3,4,5,6,7,8,9,11]
    print sqdf 

    bqual = np.array(qdf[rembits].sum(axis=1)==0)
    print "Removing %i cadences due to quality flag" % (ncad - np.sum(bqual)) 
    
    # Because masks are not always rectangles, we have to deal with nans.
    flux0 = ma.masked_invalid(cube['FLUX'])
    flux0 = flux0.sum(2).sum(1)
    bfinite = np.array(np.isfinite(flux0))
    print "Removing %i cadences due nans" % (ncad - np.sum(bfinite)) 
    
    btime = (cube['TIME'] > mintime) & (cube['TIME'] < maxtime)
    if type(tex)!=type(None):
        for rng in tex:
            # Include regions that are outside of time range
            brng = (cube['TIME'] < rng[0]) | (cube['TIME'] > rng[1])
            btime = btime & brng

    print "Removing %i cadences due to time limits" % (ncad - np.sum(btime)) 

    b = bqual & bfinite & btime
    print "Removing %i cadences total" % (ncad - np.sum(b)) 

    assert type(b)==type(np.ones(0)),"Boolean mask must be array"
    cube = cube[b]
    print "tmin = %i, tmax = %i" % tuple(cube['TIME'][[0,-1]])
    cube['TIME'][:] = cube['TIME'] + bjd0
    ret = (cube,) + ([headerToDict(el.header) for el in f],)

    f.close()
    return ret

def postage_stamp( pixFileIn, pixFileOut, ncolstamp=20 , nrowstamp=20):
    f = fits.open(pixFileIn)
    xcen, ycen = get_star_pos(pixFileIn)
    xcen = int(xcen)
    ycen = int(ycen)
    colslice = slice(xcen-ncolstamp/2,xcen+ncolstamp/2)
    rowslice = slice(ycen-nrowstamp/2,ycen+nrowstamp/2)
    


    hdu0 = f[0]
    hdu1 = f[1]
    hdu2 = f[2]

    # Update first record
    cutcol = 'RAW_CNTS FLUX FLUX_ERR FLUX_BKG FLUX_BKG_ERR COSMIC_RAYS'.split()
    new_column_list = []
    hdu1_new_header = hdu1.header

    for col in hdu1.columns:
        if cutcol.count(col.name)==1:
            data = hdu1.data[col.name][:,rowslice,colslice]
            dim = str(data[0].shape)
            
        else:
            data = hdu1.data[col.name]
            dim = col.dim

        new_column = fits.Column(
            name=col.name, format=col.format, unit=col.unit, disp=col.disp, 
            array=data, dim=dim
            )
        new_column_list += [new_column] 

    hdu1_new = fits.BinTableHDU.from_columns(new_column_list)
    hdu1_new.header = hdu1.header
    # Update second record
    hdu2.data = hdu2.data[rowslice,colslice]
    hdu2.header['CRPIX1'] -= colslice.start 
    hdu2.header['CRPIX2'] -= rowslice.start

    hduL = fits.HDUList([hdu0,hdu1_new,hdu2])
    hduL.writeto(pixFileOut,clobber=True)

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
    df = df.astype(int)
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


def get_wcs(f):
    """
    Get WCS object from fits header

    Parameters
    ----------
    f : path to fits file

    Returns
    -------
    w : wcs object
    """
    with fits.open(f) as hduL:
        w = wcs.WCS(header=hduL[2].header,key=' ')
    return w 

def get_stars_pix(pixfn,frame, retsynframe=False, ids='all', prfpath=None,dkepmag=5, verbose=False, refine_wcs=False):
    """
    Get Stars Position

    Query the catalog for stars near the target. For the initial
    release of C0, the WCS was offset from the true location of the
    star and we needed to refine the WCS solution by registering to a
    synthetic image (turn on refine_wcs). This registration can
    fail as in the case of C1-201609326. 

    Generate a synthetic
    image of stars and then register those stars to the image.

    Parameters
    ----------
    pixfn : pixel file name
    frame : reference frame.
    retsynframe : bool; return synthetic frame
    ids : which KIC/EPIC values to include
    prfpath : filename of PRF file, or None
    dkepmag : grab stars upto dkepmag fainter than target
    refine_wcs : Set to true if we want to refine the WCS solution
                    by registering with a synthetic image


    Return
    ------
    catcut : DataFrame with stars position in pixel coordinates
    shift : shift between WCS and data frame (in pixels)

    """
    # 2014-09-30 18:30 IJMC: Now output both catcut & shift
    # 2014-11-16 09:43 IJMC: Added options: retsynframe, ids, dkepmag.

    catcut = query_stars_in_stamp(pixfn, dkepmag=dkepmag)
    
    # Determine where stars are supposed to fall based on wcs
    w = get_wcs(pixfn)
    pix = w.wcs_world2pix(catcut['ra'],catcut['dec'],0)
    catcut['pix0'],catcut['pix1'] = pix

    if refine_wcs:
        # Generate a synthetic image
        # x and y fliped to account for python imshow convention
        y,x = np.mgrid[0:frame.shape[0],0:frame.shape[1]]
        synframe = np.zeros(frame.shape)
        synframe_special = np.zeros(frame.shape)
        catcut['A'] = catcut['kepmag'] - np.min(catcut['kepmag'])
        catcut['A'] = 10**(-0.4 * catcut.A)

        if ids=='all':
            index = catcut.index
        else:
            index = ids

        if prfpath is not None:
            prf, sampling = loadPRF(file=pixfn)
            prf = an.pad(prf, frame.shape[0]*sampling, frame.shape[1]*sampling)
            peakloc = (prf==prf.max()).nonzero()

        for i in catcut.index:
            d = catcut.ix[i]
            g = gaussian(d['pix0'],d['pix1'],0.5)
            if prfpath is not None:
                prfmod = ld.shiftImages(d['A'] * prf, d['pix0']*sampling-peakloc[0], d['pix1']*sampling-peakloc[1]).squeeze()
                thisstar = an.binarray(prfmod, sampling)
            else:
                thisstar = d['A']*g(x,y)
            synframe += thisstar
            if i in index:
                synframe_special += thisstar

        scalefactor = frame.sum() / synframe.sum()
        synframe = synframe_special * scalefactor
        #synframe *= scalefactor


        # Determine the shift between reference and synthetic images
        shift = register_images(frame, synframe, usfac=100.)
        shift = np.array(shift)

        if verbose: 
            print "stars shifted by %s pixels from header WCS" % str(shift)
        catcut['pix0']-=shift[0]
        catcut['pix1']-=shift[1]

        epic = fits.open(pixfn)[0].header['KEPLERID']
        xcen,ycen = catcut.ix[epic]['pix0 pix1'.split()]
        #print xcen,ycen

    else:
        shift = np.array([0.,0.])

    ret = catcut, shift
    if retsynframe:
        ret += (synframe,)

    return ret

def get_star_pos(f,mode='wcs'):
    """
    Get Star's Position (pixel coordinates)
    
    Parameters
    ----------
    f : path to fits file
    mode : How do we determine star's position?

    Returns
    -------
    xcen,ycen : tuple with the X and Y position of the star
    
    """

    with fits.open(f) as hduL:
        if mode=='aper':
            aper = hduL[2].data
            pos = nd.center_of_mass(aper==3)
            xcen,ycen = pos[0],pos[1]
        elif mode=='wcs':
            w = get_wcs(f)
            ra,dec = hduL[0].header['RA_OBJ'],hduL[0].header['DEC_OBJ']
            xcen0,ycen0 = w.wcs_world2pix(ra,dec,0)

    return xcen0,ycen0

def query_stars_in_stamp(pixfn,dkepmag=5):
    """
    Query stars falling in fits stamp

    Parameters
    ----------
    pixfn : pixel filename
    dkepmag : grab stars upto dkepmag fainter than target
    """
    hduL = fits.open(pixfn)
    ra,dec = hduL[0].header['RA_OBJ'],hduL[0].header['DEC_OBJ']
    epic = hduL[0].header['KEPLERID']
    cat = k2_catalogs.read_cat('C%i' % hduL[0].header['CAMPAIGN'] )
    
    frame = hduL[1].data['flux'][0]
    pixw = max(frame.shape) / 2
    degw = 4*pixw/3600.
    rarng = (ra-degw,ra+degw)
    decrng = (dec-degw,dec+degw)
    kepmagmax = cat.ix[epic,'kepmag'] + dkepmag
    catcut = cat[cat.ra.between(*rarng) & cat.dec.between(*decrng) & 
                (cat.kepmag < kepmagmax)]

    # Return copy so that pandas doesn't complain about working on a view
    catcut = catcut.copy() 
    return catcut

def cardUndefined(headercard):
    try:
        from astropy.io import fits as pyfits
    except:
        import pyfits

    return (headercard is pyfits.card.UNDEFINED) or \
            (headercard is pyfits.card.Undefined)

def headerToDict(header):
    """Convert a PyFITS header into a standard NumPy dict.
    """
    # 2014-08-27 11:33 IJMC: Created

    ret = dict()
    comments = dict()
    for card in header.cards:
        key, val, comment = card
        if cardUndefined(val):
            val = ''
        ret[key] = val
        comments[key] = comment

    ret['comments'] = comments
    return ret
