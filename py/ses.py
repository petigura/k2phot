"""
Single Event Statistics
"""
from matplotlib.pylab import * 
from scipy import ndimage as nd
import pandas as pd

def mtd(fm,twd):
    """
    Mean Transit Depth

    Convolve time series with our locally detrended matched filter.  

    Parameters
    ----------
    t      : time series.
    fm     : masked flux array.  masked regions enter into the average
             with 0 weight.
    twd    : Width of kernel in cadances

    Notes
    -----
    Since a single nan in the convolution kernel will return a nan, we
    interpolate the entire time series.  We see some edge effects

    """

    assert isinstance(twd,int),"Box width most be integer number of cadences"

    fm = fm.copy()
    fm.fill_value = 0
    w = (~fm.mask).astype(int) # weights either 0 or 1
    f = fm.filled()

    pad = np.zeros(twd)
    f = np.hstack([pad,f,pad])
    w = np.hstack([pad,w,pad])

    assert (np.isnan(f)==False).all() ,'mask out nans'
    kern = np.ones(twd,float)

    ws = np.convolve(w*f,kern,mode='same') # Sum of points in bin
    c = np.convolve(w,kern,mode='same')    # Number of points in bin

    # Number of good points before, during, and after transit
    bc = c[:-2*twd]
    tc = c[twd:-twd]
    ac = c[2*twd:]

    # Weighted sum of points before, during and after transit
    bws = ws[:-2*twd]
    tws = ws[twd:-twd]
    aws = ws[2*twd:]
    dM = 0.5*(bws/bc + aws/ac) - tws/tc
    dM = ma.masked_invalid(dM)
    dM.fill_value =0

    # Require 0.5 of the points before, during and after transit to be good.
    gap = (bc < twd/2) | (tc < twd/2) | (ac < twd/2)
    dM.mask = dM.mask | gap

    return dM

def running_mean(fm,size):
    fm = fm.copy()
    fm.fill_value = 0
    w = (~fm.mask).astype(float) # weights either 0 or 1
    f = fm.filled()
    assert (np.isnan(f)==False).all() ,'mask out nans'
    
    # total flux in bin
    f_sum = nd.uniform_filter(f,size=size) * size 
    # number of unmasked points in bin
    f_count = nd.uniform_filter(w,size=size) * size
    f_mean = ma.masked_array( f_sum / f_count, f_count < 0.5*size) 
    return f_mean

def ses_stats(fm):
    """
    Given a light curve what is the noise level on different timescales?
    """
    dL = []
    for twd in [1,4,6,8,12]:
        fom = ma.std(running_mean(fm,twd))
        dL.append(['rms_%i-cad-mean' % twd, fom ,twd])

        fom = ma.std(mtd(fm,twd))
        dL.append(['rms_%i-cad-mtd' % twd, fom,twd])

        fom = ma.median(ma.abs(running_mean(fm,twd)))
        dL.append(['mad_%i-cad-mean' % twd, fom ,twd])

        fom = ma.median(ma.abs(mtd(fm,twd)))
        dL.append(['mad_%i-cad-mtd' % twd, fom,twd])

    dL = pd.DataFrame(dL,columns='name value twd'.split())
    dL['value']*=1e6
    return dL
