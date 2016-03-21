from cStringIO import StringIO as sio
import os

import numpy as np
from numpy import ma
import pandas as pd
import george

from pdplus import LittleEndian

from ses import total_precision_theory
from matplotlib import pylab as plt
import apertures

os.system('echo "pixel_decorrelation modules loaded:" $(date) ')

def kepmag_to_npix(kepmag):
    """
    Given the kepmag of given target star, provide a range of
    apertures to search over we want to search over.
    """
    if 0 <= kepmag < 10:
        npix = [32, 45, 64, 90, 128, 181, 256, 362, 512]
    elif 10 <= kepmag < 14:
        npix = [4.0, 5.7, 8.0, 11, 16, 22, 32, 45, 64, 90, 128, 181]
    elif 14 <= kepmag < 25:
        npix = [4.0, 5.7, 8.0, 11, 16, 22, 32, 45,]

    return npix

def npix_to_ap_type(npix):
    """
    Caculates the when we transition form circular apertures to region
    apertures.
    """
    radius_switch = 4.0 # switch from circular to region aperture.
    npix_switch = np.pi * radius_switch**2

    if npix > npix_switch:
        return 'region'
    else:
        return 'circular'

def white_noise_estimate(kepmag):
    """
    Estimate White Noise
    
    The Gaussian Process noise model assumes that some of the variance
    is white. 
    """
    fac = 2 # Factor by which to inflate Poisson and read noise estimate
    noise_floor = 100e-6 # Do not allow noise estimate to fall below this amount
    
    # Estimate from Poisson and read noise.
    sigma_th =  total_precision_theory(kepmag,10)
    sigma_th *= fac
    sigma_th = max(noise_floor,sigma_th)
    return sigma_th

def detrend_t_roll_2D(lc, sigma, length_t, length_roll, sigma_n, 
                      reject_outliers=False,debug=False):
    """
    Detrend against time and roll angle. Hyperparameters are passed
    in as arguments. Option for iterative outlier rejection.

    Parameters
    ----------
    sigma : sets the scale of the GP variance
    length_t : length scale [days] of GP covariance
    length_roll : length scale [arcsec] of GP covariance
    sigma_n : amount of white noise
    reject_outliers : True, reject outliers using iterative sigma clipping

    Returns 
    -------
    """

    # Define constants
    Xkey = 't roll'.split() # name of dependent variable
    Ykey = 'f' # name of independent variable
    fdtkey = 'fdt_t_roll_2D' 
    ftndkey = 'ftnd_t_roll_2D' 
    outlier_threshold = [None,10,5,3]

    if reject_outliers:
        maxiter = len(outlier_threshold) - 1
    else:
        maxiter = 1

    print "sigma, length_t, length_roll, sigma_n"
    print sigma, length_t, length_roll, sigma_n

    iteration = 0
    while iteration < maxiter:
        if iteration==0:
            fdtmask = np.array(lc.fdtmask)
        else:
            # Clip outliers 
            fdt = lc[fdtkey]
            sig = np.median( np.abs( fdt ) ) * 1.5
            newfdtmask = np.abs( fdt / sig ) > outlier_threshold[iteration]
            lc.fdtmask = lc.fdtmask | newfdtmask
            
        print "iteration %i, %i/%i excluded from GP" % \
            (iteration,  lc.fdtmask.sum(), len(lc.fdtmask) )

        # suffix _gp means that it's used for the training
        # no suffix means it's used for the full run
        lc_gp = lc[~lc.fdtmask] 

        # Define the GP
        kernel = sigma**2 * george.kernels.ExpSquaredKernel(
            [length_t**2,length_roll**2],ndim=2
            ) 

        gp = george.GP(kernel)
        gp.compute(lc_gp[Xkey],sigma_n)

        # Detrend againts time and roll angle
        mu,cov = gp.predict(lc_gp[Ykey],lc[Xkey])
        lc[ftndkey] = mu
        lc[fdtkey] = lc[Ykey] - lc[ftndkey]

        # Also freeze out roll angle dependence
        medroll = np.median( lc['roll'] ) 
        X_t_rollmed = lc[Xkey].copy()
        X_t_rollmed['roll'] = medroll
        mu,cov = gp.predict(lc_gp[Ykey],X_t_rollmed)
        lc['ftnd_t_rollmed'] = mu
        lc['fdt_t_rollmed'] = lc[fdtkey] + mu
        iteration+=1

    if debug:
        lc_gp = lc[~lc.fdtmask] 
        from matplotlib.pylab import *
        ion()
        fig,axL = subplots(nrows=2,sharex=True)
        sca(axL[0])
        errorbar(lc_gp['t'],lc_gp[Ykey],yerr=sigma_n,fmt='o')
        plot(lc['t'],lc[ftndkey])
        sca(axL[1])
        fm = ma.masked_array(lc[fdtkey],lc['fmask'])
        plot(lc['t'],fm)
        fig = figure()
        plot(lc_gp['roll'],lc_gp['f'],'.')
        plot(lc_gp['roll'],lc_gp['ftnd_t_roll_2D'],'.')

    return lc

def detrend_t_roll_2D_segments(*args,**kwargs):
    """
    Simple wrapper around detrend_t_roll_2D

    Parameters
    ----------
    segment_length : approximate time for the segments [days]
    
    Returns
    -------
    lc : lightcurve after being stiched back together
    """
    lc = args[0]
    segment_length = kwargs['segment_length']
    kwargs.pop('segment_length')
    nchunks = lc['t'].ptp() / segment_length 
    nchunks = int(nchunks)
    nchunks = max(nchunks,1)
    if nchunks==1:
        args_segment = (lc,) + args[1:]
        return detrend_t_roll_2D(*args_segment,**kwargs)

    lc_segments = np.array_split(lc,nchunks)
    lc_out = []
    for i,lc in enumerate(lc_segments):
        args_segment = (lc,) + args[1:]
        lc_out+=[detrend_t_roll_2D(*args_segment,**kwargs)]

    lc_out = pd.concat(lc_out)
    return lc_out




