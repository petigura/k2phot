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
    
    The MTD statistic gives higher noise than RMS because there are
    uncertanties associated with measuring the wings.

    MTD = sqrt(3/2) / sqrt(twd) * sigma, where sigma is the point-to-point rms

    """
    dfses = []

    def ma_mad(x):
        fom = 1.48 * ma.median(ma.abs(x))
        if type(fom)==type(fm):
            fom = float(fom)
        return fom

    ma_std = lambda x : ma.std(x)
    for twd in [1,4,6,8,12]:
        fom = ma_std(running_mean(fm,twd))
        dfses.append(['rms_%i_cad_mean' % twd, fom ,twd])

        fom = ma_std(mtd(fm,twd))
        dfses.append(['rms_%i_cad_mtd' % twd, fom,twd])

        fom = ma_mad(running_mean(fm,twd))
        dfses.append(['mad_%i_cad_mean' % twd, fom ,twd])

        fom = ma_mad(mtd(fm,twd))
        dfses.append(['mad_%i_cad_mtd' % twd, fom,twd])


    dfses = pd.DataFrame(dfses,columns='name value twd'.split())
    dfses['value']*=1e6
    dfses.index=dfses.name
    dfses = dfses['value']
    return dfses

def source_noise(flux):
    """
    flux : Flux measued. Units are electrons/s, so convert them to
    electrons by multiplying by long-cadence duration
    """
    
    tint = 6.02 # Integration (s)
    nint = 270 # Integrations in long cadence measurement
    noise = sqrt(flux * tint * nint)
    return noise

def kepmag_to_flux(kepmag):
    """
    Convert kepmag_to_flux

    flux is electrons/s
    """
    kepmag0 = 12 
    f0 = 1.74e5
    fkep = f0*10**(-0.4 * (kepmag - kepmag0 ))
    return fkep

def total_precision_theory(kepmag,Na):  
    """Calculate the expected precsion. I used the equation listed here:

    http://keplergo.arc.nasa.gov/CalibrationSN.shtml

    Note that to get the same values listed on the website, multiply
    by their fudge factor of 1.2.

    There are different metrics of noise. In the limit of gaussian
    noise, with point-to-point rms = 1.0
    
    twd = number of candences used in running box.

    """
    Nfr = 270 # Number of frames used in exposure (long cadence) 
    tint = 6.02 # Seconds in an integration
    NR = 120 # Readnoise e-
    fkep = kepmag_to_flux(kepmag)
    numer = sqrt(fkep*tint + Na * NR**2)
    denom = sqrt(Nfr) * fkep* tint
    precision = numer/denom
    return precision


def background_noise(kepmag,Na,fbg):  
    """
    kepmag : Kepler magnitude
    Na : number of pixels in the aperture
    fbg : nominal value for background flux. e-/s/pixel
    """

    Nfr = 270 # Number of frames used in exposure (loncadence) 
    tint = 6.02 # Seconds in an integration
    sourceflux = kepmag_to_flux(kepmag) * tint * Nfr
    bgflux = fbg * tint * Nfr * Na
    bgnoise = np.sqrt(bgflux) / sourceflux
    return bgnoise


def convert_rms_to_mad_6_cad_mtd(rms):
    """
    rms_1_cad_mean = 1.0
    rms_1_cad_mtd = sqrt(3/2) = 1.225 (extra noise from measuring the
                                       depths of the wings)
    rms_4_cad_mean = 0.500007 divide by sqrt(twd)
    mad_1_cad_mean = 0.674639 to convert from rms to mad divide by 1.5
    """

    # 
    out = sqrt(3/2) * rms # accounts for wings
    out /= sqrt(6) # Account for larger width 
    out *= 0.674639 # RMS -> MAD
    return out
