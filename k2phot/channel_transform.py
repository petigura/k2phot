from argparse import ArgumentParser
import os
import sqlite3

from astropy.io import fits
import numpy as np
from numpy import ma
import h5py
import pandas as pd
from matplotlib import mlab
from matplotlib import pylab as plt
from scipy import ndimage as nd

import image_transform as imtran
from io_utils.pixel import get_wcs
from io_utils import h5plus
from config import bjd0 
def channel_transform(fitsfiles, h5file, iref= None):
    """
    Channel Transformation

    Take a list of k2 pixel files (must be from the same
    channel). Find the centroids of each image and solve for the
    linear transformation that takes one scene to another
    """
    nstars = len(fitsfiles)

    # Pull the first file to get length and data type
    fitsfile0 = fitsfiles[0]
    cent0 = fits_to_chip_centroid(fitsfile0)
    channel = get_channel(fitsfile0)
    print "Using channel = %i" % channel

    # Determine the refence frame
    if iref==None:
        dfcent0 = pd.DataFrame(LE(cent0))
        ncad = len(dfcent0)
        med = dfcent0.median()
        dfcent0['dist'] = (
            (dfcent0['centx'] - med['centx'])**2 +
            (dfcent0['centy'] - med['centy'])**2
            )
        dfcent0 = dfcent0.iloc[ncad/4:-ncad/4]
        dfcent0 = dfcent0.dropna(subset=['centx','centy'])
        iref = dfcent0['dist'].idxmin()
    
    print "using reference frame %i" % iref
    assert np.isnan(cent0['centx'][iref])==False,\
        "Must select a valid reference cadence. No nans"

    cent = np.zeros((nstars,cent0.shape[0]), cent0.dtype)
    for i,fitsfile in enumerate(fitsfiles):
        if (i%10)==0:
            print i
        cent[i] = fits_to_chip_centroid(fitsfile)
        channel_i = get_channel(fitsfile)
        assert channel==channel_i,"%i != %i" % (channel, channel_i)
        
    trans,pnts = imtran.linear_transform(cent['centx'],cent['centy'],iref)
    keys = cent.dtype.names
    pnts = mlab.rec_append_fields(pnts,keys,[cent[k] for k in keys])

    if h5file!=None:
        with h5plus.File(h5file) as h5:
            h5['trans'] = trans
            h5['pnts'] = pnts
            
    trans,pnts = read_channel_transform(h5file)
    plot_trans(trans, pnts)
    figpath = h5file[:-3] + '.png'
    plt.gcf().savefig(figpath)
    print "saving %s " % figpath
    return cent

def centroid(flux):
    """
    Centroid
    
    Parameters
    ----------
    flux : flux cube (already should be masked and background subtracted)
    
    Returns
    -------
    centcol : Centroid of along column axis. 0 corresponds to origin
    centrow : Centroid of along row axis. 0 corresponds to origin
    """
    nframe,nrow,ncol = flux.shape

    irow = np.arange(nrow)
    icol = np.arange(ncol)

    # Compute row centriod
    fluxrow = np.sum(flux, axis=2) # colflux.shape = (nframe,nrow)
    centrow = np.sum( (fluxrow*irow), axis=1) / np.sum(fluxrow,axis=1)

    # Compute column centriod
    fluxcol = np.sum(flux,axis=1) # colflux.shape = (nframe,ncol)
    centcol = np.sum( (fluxcol*icol), axis=1) / np.sum(fluxcol,axis=1)
    return centcol,centrow

def fits_to_chip_centroid(fitsfile):
    """
    Grab centroids from fits file

    Parameters
    ----------
    fitsfile : path to pixel file

    Returns
    -------
    centx : centroid in the x (column) axis
    centy : centroid in the y (row) axis
    """
    apsize = 7

    hdu0,hdu1,hdu2 = fits.open(fitsfile)
    cube = hdu1.data
    flux = cube['FLUX']
    t = cube['TIME']
    cad = cube['CADENCENO']

    nframe,nrow,ncol = flux.shape

    # Define rectangular aperture
    wcs = get_wcs(fitsfile)
    ra,dec = hdu0.header['RA_OBJ'],hdu0.header['DEC_OBJ']
    x,y = wcs.wcs_world2pix(ra,dec,0)
    scentx,scenty = np.round([x,y]).astype(int)
    nrings = (apsize-1)/2

    x0 = scentx - nrings
    x1 = scentx + nrings
    y0 = scenty - nrings
    y1 = scenty + nrings
    mask = np.zeros((nrow,ncol))
    mask[y0:y1+1,x0:x1+1] = 1 # 1 means use in aperture

    # Compute background flux
    # mask = True aperture, don't use to compute bg
    flux_sky = flux.copy()
    flux_sky_mask = np.zeros(flux.shape)
    flux_sky_mask += mask[np.newaxis,:,:].astype(bool)
    flux_sky = ma.masked_array(flux_sky, flux_sky_mask)
    fbg = ma.median(flux_sky.reshape(flux.shape[0],-1),axis=1)

    # Subtract off background
    flux = flux - fbg[:,np.newaxis,np.newaxis]
    flux = ma.masked_invalid(flux)
    flux.fill_value = 0 
    flux = flux.filled()

    # Compute aperture photometry
    fsap = flux * mask
    fsap = np.sum(fsap.reshape(fsap.shape[0],-1),axis=1)

    # Compute centroids
    centx,centy = centroid(flux * mask)

    # table column physical WCS ax 1 ref value       
    # hdu1.header['1CRV4P'] corresponds to column of flux[:,0,0]
    # starting counting at 1. 
    centx += hdu1.header['1CRV4P'] - 1
    centy += hdu1.header['2CRV4P'] - 1

    r = np.rec.fromarrays(
        [t,cad,centx,centy,fsap,fbg],
        names='t,cad,centx,centy,fsap,fbg'
        )

    r = mlab.rec_append_fields(r,'starname',hdu0.header['KEPLERID'])
    return r

def get_channel(fitsfile):
    """Return channel"""
    return fits.open(fitsfile)[0].header['CHANNEL']

def get_thrustermask(dtheta):
    """
    Identify thruster fire events.

    Delta theta is the change in telescope roll angle compared to the
    previous cadence. We establish the typical scatter in delta theta
    by computing the median absolute deviation of points in regions of
    100 measurements. We identfy thruster fires as cadences where the
    change in telescope roll angle exceeds the median absolute
    deviation by > 15.

    Parameters 
    ----------
    dtheta : roll angle for contiguous observations

    Returns
    -------
    thrustermask : boolean array where True means thruster fire

    """
    medfiltwid = 10 # Filter width to identify discontinuities in theta
    sigfiltwid = 100 # Filter width to establish local scatter in dtheta
    thresh = 10 # Threshold to call something a thruster fire (units of sigma)

    medtheta = nd.median_filter(dtheta,medfiltwid)
    diffdtheta = np.abs(dtheta - medtheta)
    sigma = nd.median_filter(diffdtheta,sigfiltwid) * 1.5
    thrustermask = np.array(diffdtheta > thresh*sigma)
    return thrustermask



def trans_add_columns(trans):
    """
    Add columns to transformation table.

    The A, B, C, and D components of the transformation matrix along
    with the displacement vector completely specifies the affine
    transformation. However, it is usually nice to think in terms of
    scale, skew along x and y, and rotation about the origin. This
    function casts the matrix in terms of these more useful
    parameters.

    Parameters
    ----------
    trans : record array or pandas data frame with A, B, C, D columns
    
    """

    trans = pd.DataFrame(trans)
    trans['scale'] = np.sqrt(trans.eval('A*D-B*C'))-1 
    trans['scale'][abs(trans['scale']) > 0.1] = None
    trans['skew1'] = trans.eval('(A-D)/2')
    trans['skew2'] = trans.eval('(B+C)/2')
    trans['theta'] = trans.eval('(B-C)/(A+D)')
    obs = np.arange(len(trans))
    bi = np.array(np.isnan(trans.theta))
    trans.loc[bi,'theta'] = np.interp(obs[bi],obs[~bi],trans[~bi].theta)
    theta = np.array(trans['theta'])
    dtheta = theta[1:] - theta[:-1]
    dtheta = np.hstack([[0],dtheta])
    trans['dtheta'] = dtheta
    return trans

def LE(arr):
    names = arr.dtype.names
    arrL = []
    for n in names:
        if arr.dtype[n].byteorder==">":
            arrnew = arr[n].byteswap().newbyteorder('=')
        else:
            arrnew = arr[n]
        arrL.append(arrnew)
    arrL = np.rec.fromarrays(arrL,names=names)
    return arrL

def read_channel_transform(h5file):
    with h5py.File(h5file,'r') as h5:
        trans = h5['trans'][:] 
        pnts = h5['pnts'][:]

    trans = LE(trans)
    pnts = LE(pnts)
    trans = trans_add_columns(trans)
    trans['thrustermask'] = get_thrustermask(trans['dtheta'])
    return trans,pnts

def plot_trans(trans,pnts):
    """
    Diagnostic plot that shows different transformation parameters
    """
    t = pnts[0]['t']
    keys = 'scale theta skew1 skew2'.split()
    nrows = 5
    fig,axL = plt.subplots(nrows=nrows,figsize=(20,8),sharex=True)
    fig.set_tight_layout(True)
    for i,key in enumerate(keys):
        plt.sca(axL[i])
        plt.plot(t,trans[key])
        plt.ylabel(key)
        
    plt.sca(axL[4])
    dtheta = np.array(trans['dtheta'])
    thrustermask = np.array(trans['thrustermask'])
    plt.plot(t,dtheta,'-',mew=0)
    plt.ylabel('$\Delta$ theta')
    plt.xlabel('Time BJD - %i' % bjd0)
    plt.plot(t[thrustermask],dtheta[thrustermask],'.')
