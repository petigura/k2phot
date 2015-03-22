import numpy as np
from numpy import ma
from scipy.optimize import fmin
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import pandas as pd

from frame import Frame
import circular_photometry
from pixel_decorrelation import loadPixelFile
from pixel_decorrelation import get_wcs


class ImageStack(object):
    def __init__(self,fn,tlimits=[-np.inf,-np.inf],tex=None):
        cube,headers = loadPixelFile(fn,tlimits=tlimits,tex=tex)
        self.fn = fn
        self.headers = headers
        self.flux = cube['FLUX'].astype(float)
        self.t = cube['TIME'].astype(float)
        self.cad = cube['CADENCENO'].astype(int)

        # Number of frames, rows, and columns
        self.nframe,self.nrow,self.ncol = self.flux.shape 
        self.npix = self.nrow*self.ncol

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
    """
    Background Mask
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
    wcs = get_wcs(im.fn)
    ra,dec = im.headers[0]['RA_OBJ'],im.headers[0]['DEC_OBJ']
    x,y = wcs.wcs_world2pix(ra,dec,0)
    x = float(x)
    y = float(y)
    return im,x,y


