import numpy as np
from numpy import ma
from scipy.optimize import fmin
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import pandas as pd

from frame import Frame
import circular_photometry
from io_utils.pixel import loadPixelFile, get_wcs
from circular_photometry import circular_photometry_weights

class ImageStack(object):
    def __init__(self, pixfile, tlimits=[-np.inf,np.inf],tex=None):
        """
        Initialize ImageStack Object
        
        Parameters
        ----------
        pixfile : Target Pixel File (download from the MAST)
        tlimits : [optional] readin only a segment of data subtracts
                  of config.bjd0
        """

        cube,headers = loadPixelFile(pixfile,tlimits=tlimits,tex=tex)
        self.pixfile = pixfile
        self.headers = headers
        self.flux = cube['FLUX'].astype(float)
        self.t = cube['TIME'].astype(float)
        self.cad = cube['CADENCENO'].astype(int)

        # Number of frames, rows, and columns
        self.nframe, self.nrow, self.ncol = self.flux.shape 
        self.npix = self.nrow * self.ncol

        ts = [(k,getattr(self,k)) for k in 't cad'.split()]
        ts = dict(ts)
        self.ts = pd.DataFrame(ts)
        self.tlimits = tlimits
        self.ap = None

    def get_xy_from_header(self):
        """
        Get x,y position of the target star from the header WCS
        """
        
        wcs = get_wcs(self.pixfile)
        ra,dec = self.headers[0]['RA_OBJ'],self.headers[0]['DEC_OBJ']
        x,y = wcs.wcs_world2pix(ra,dec,0)
        x = float(x)
        y = float(y)
        return x,y

    def set_fbackground(self):
        """
        Set flux in background

        Uses the current value of self.ap (which must be set) and
        constructs the median flux outside.
        """
        if 0:
            from matplotlib.pylab import *
            ion()
            import pdb;pdb.set_trace()

        flux = ma.masked_array(self.flux,fill_value=0)
        ap_mask = self.ap.weights > 0
        ap_mask = ap_mask[np.newaxis,:,:]
        flux.mask = flux.mask | ap_mask # Mask out if included in aperture
        flux.mask = flux.mask | np.isnan(flux.data)
        flux = flux.reshape(-1,self.npix)
        self.fbg = np.array(ma.median(flux,axis=1))
        fbgfit,bgmask = background_mask(self.cad,self.fbg)

        #Checks if every single cadence is a nan. If yes don't include at all
        is_all_nan = flux.mask.sum(1)==flux.mask.shape[1]
        bgmask = bgmask | is_all_nan
        self.bgmask = bgmask

        if ap_mask.sum() > 0.8 * flux.shape[1]:
            self.bgmask = np.zeros(flux.shape[0]).astype(bool)            
            self.fbg = np.zeros(flux.shape[0])

        self.ts['fbg'] = self.fbg
        self.ts['bgmask'] = self.bgmask

    def get_sap_flux(self):
        """
        Get aperture photometry. Subtract background
        """
        flux = self.flux.copy()
        flux -= self.fbg[:,np.newaxis,np.newaxis]  
        ap_flux = flux * self.ap.weights # flux falling in aperture
        ap_flux = ap_flux.reshape(self.nframe,-1)
        ap_flux = np.nansum(ap_flux,axis=1)
        return ap_flux

    def get_medframe(self):
        flux = self.flux
        flux = ma.masked_invalid(flux)
        flux = ma.median(flux,axis=0)
        return flux
    
    def get_percentile_frame(self,p):
        flux = np.nanpercentile(self.flux, p, 0)
        flux = ma.masked_invalid(flux)
        return flux
        

def background_mask(cad,fbg,plot=False):
    """Background Mask

    We subtract the background level from the total flux. Spikes or
    other data anomalies due to scattered light of bright moving
    sources or solar activity often lead to outliers in the processed
    photometry due to over- or under-subtraction. We mask out cadences
    with unusual background levels.

    Spurious background events are identified via either of the
    following criteria: 

      1. If the total background level changes by more than 10% of the
         median value

      2. We fit the overall trend of increasing background levels with
         a 3rd order polynomial. Then we run a median filter having
         size of 40 measurements over the background levels with the
         polynomial trend removed. Then we estimate the typcial rms in
         the background level ala sigma_bg = 1.5 *
         MAD(f_bgresidmed). Observations where the filtered background
         level exceeds sigma_bg by a factor of 10 are masked.

    """
    # Background thresh. If the background changes by more than
    # bgthresh of the median value, designate it as an outlier
    thresh = 2
    
    cad = np.array(cad)
    fbg = np.array(fbg)

    cad0 = cad[0]
    dcad = cad - cad0

    model = lambda p,x : np.polyval(p,dcad)
    obj = lambda p : np.sum(np.abs( model(p,dcad) - fbg ) )

    p0 = np.zeros(4)
    p0[3] = np.median(fbg)
    p1 = fmin(obj,p0,disp=0)

    fbgfit = model(p1,dcad)

    bgmask = (np.abs( fbg - fbgfit ) / np.median(fbg)) > thresh 
    bgresid = fbg - fbgfit
    bgresidmed = nd.median_filter(bgresid,size=40)
    
    absdiff = np.abs(bgresid - bgresidmed )
    sigbg = 1.5 * np.median(absdiff)
    bgmask = bgmask | (absdiff > 10 * sigbg)

    mfbg = ma.masked_array(fbg,bgmask)

    # If there are any 10 cadence regions where > 50 % of background
    # is masked, mask out the entire region.
    size = 10
    bgmaskcnt = np.convolve(bgmask,np.ones(size),mode='valid')
    bgmaskgroup = (bgmaskcnt > size/2.)
    # if the ith element of bgmaskcnt corresponds to bin from i to i+size
    for i in np.where(bgmaskgroup)[0]:
        s = slice(i,i+size)
        bgmask[s] = True
        
    print "bgmask=True for %i of %i cadences" % (bgmask.sum(),bgmask.size)

    plot=1
    if plot:
        plt.plot(cad,fbg)
        plt.plot(cad,fbgfit)
        plt.plot(cad,mfbg)
        plt.plot(cad,bgresid)
        plt.plot(cad,bgresidmed)
        plt.plot(cad,ma.masked_array(fbg,bgmask),label='fbg outliers removed')
        plt.plot(cad[:cad.size-size+1],bgmaskcnt,label='bgmaskcnt')
        yl = np.percentile(fbg,[5,95])
        yl += (np.array([-1,1]) * yl.ptp()*0.5)
        plt.ylim(*yl)
        plt.legend()
    return fbgfit,bgmask

def read_imagestack(pixfile,tlimits=[-np.inf,np.inf],tex=None):
    im = ImageStack(pixfile,tlimits=tlimits,tex=tex)
    x,y = im.get_xy_from_header()
    return im, x, y


