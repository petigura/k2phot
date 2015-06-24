import numpy as np
from numpy import ma
from scipy.optimize import fmin
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import pandas as pd

from frame import Frame
import circular_photometry
from io_utils.pixel import loadPixelFile, get_wcs

class ImageStack(object):
    def __init__(self,pixfile,tlimits=[-np.inf,-np.inf],tex=None):
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

        # Determine background level from median
        self.fbg = self.get_fbackground()
        fbgfit,bgmask = background_mask(self.cad,self.fbg)
        self.bgmask = bgmask

        ts = [(k,getattr(self,k)) for k in 't cad fbg bgmask'.split()]
        ts = dict(ts)
        self.ts = pd.DataFrame(ts)
        self.tlimits = tlimits

    def set_apertures(self,locx,locy,radius):
        """
        Set the apertures used to compute photometry
        """
        if hasattr(locx,'__iter__'):
            assert ((len(locx)==self.nframe) &
                    (len(locy)==self.nframe) ), "Must have same length as array"

        self.radius = radius
        self.ts['locx'] = locx
        self.ts['locy'] = locy

        def get_ap_weights(locx,locy):
            positions = np.array([[locx,locy]])
            return circular_photometry.circular_photometry_weights(
                self.flux[0],positions,radius)
            
        ap_weights = map(get_ap_weights,self.ts.locx,self.ts.locy)
        ap_weights = np.array(ap_weights)
        self.ap_weights = ap_weights
        
    def get_fbackground(self):
        flux = self.flux.reshape(-1,self.npix)
        fbackground = np.median(flux,axis=1)
        return fbackground

    def get_sap_flux(self):
        """
        Get aperture photometry. Subtract background
        """

        flux = self.flux.copy()
        flux -= self.fbg[:,np.newaxis,np.newaxis]  
        ap_flux = flux * self.ap_weights # flux falling in aperture
        ap_flux = ap_flux.reshape(self.nframe,-1)
        ap_flux = np.nansum(ap_flux,axis=1)
        return ap_flux

    def get_flux(self):
        """Get flux cube
        
        For Imstack object we don't modify the flux cube, so this is a
        no-op. For derived classes, we do and get_flux protechts
        self.flux
        """
        return self.flux

    def get_frame(self,i):
        flux = self.get_flux()

        locx = None
        locy = None
        ap_weights = None
        radius = None
        try:
            locx,locy = self.ts.iloc[i]['locx locy'.split()]
        except:
            print "locx,locy not set"

        try:
            ap_weights = self.ap_weights[i]
        except:
            print "ap_weights not set"

        frame = Frame(
            flux[i],locx=locx,locy=locy,r=self.radius,ap_weights=ap_weights)
        return frame

    def get_medframe(self):
        locx,locy =tuple(self.ts['locx locy'.split()].median())
        flux = self.get_flux()
        flux = ma.masked_invalid(flux)
        flux = ma.median(flux,axis=0)
        frame = Frame(
            flux,locx=locx,locy=locy,r=self.radius)
        return frame

    def get_frames(self):
        flux = self.get_flux()
        def get_frame(i):
            ap_weights = self.ap_weights[i]
            tsi = self.ts.iloc[i]

            frame = Frame(
                flux[i], locx=tsi['locx'], locy=tsi['locy'], r=self.radius,
                ap_weights=ap_weights)
            return frame

        frames = map(get_frame,range(self.nframe))
        return frames

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
    thresh = 0.1
    
    cad = np.array(cad)
    fbg = np.array(fbg)

    cad0 = cad[0]
    dcad = cad - cad0

    model = lambda p,x : np.polyval(p,dcad)
    obj = lambda p : np.sum(np.abs( model(p,dcad) - fbg ) )

    p0 = np.zeros(4)
    p0[3] = np.median(fbg)
    p1 = fmin(obj,p0)

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

    if plot:
        plt.plot(cad,fbg)
        plt.plot(cad,fbgfit)
        plt.plot(cad,mfbg)
        plt.plot(cad,bgresid)
        plt.plot(cad,bgresidmed)
        plt.plot(cad[:cad.size-size+1],bgmaskcnt,label='bgmaskcnt')
        yl = np.percentile(fbg,[5,95])
        yl += (np.array([-1,1]) * yl.ptp()*0.5)
        plt.ylim(*yl)
        plt.legend()
    return fbgfit,bgmask

def read_imagestack(pixfile,tlimits=[-np.inf,np.inf],tex=None):
    im = ImageStack(pixfile,tlimits=tlimits,tex=tex)
    wcs = get_wcs(im.pixfile)
    ra,dec = im.headers[0]['RA_OBJ'],im.headers[0]['DEC_OBJ']
    x,y = wcs.wcs_world2pix(ra,dec,0)
    x = float(x)
    y = float(y)
    return im, x, y


