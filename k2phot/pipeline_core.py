"""
Core structure for multiple pipelines.
"""
import os
import contextlib

import numpy as np
from astropy.io import fits
import pandas as pd
from matplotlib import pylab as plt
from numpy import ma
from scipy.optimize import minimize

import phot
import imagestack 
import apertures
from lightcurve import Lightcurve, Normalizer
from channel_transform import read_channel_transform
from ses import total_precision_theory


class Pipeline(object):
    """Pipeline class

    Class the holds the parameters used in the k2phot pipeline and
    performs intermediate steps.
    
    :param pixfn: path to pixel file
    :type pixfn: str

    :param lcfn: path to pixel file
    :type lcfn: str

    :param tranfn: path to pixel file
    :type tranfn: str
    """

    unnormkeys = [
        "f",
        "fdt_t_roll_2D",
        "ftnd_t_roll_2D",
        "fdt_t_rollmed",
        "ftnd_t_rollmed",
    ]
    # Small, med, and large apertures 
    DEFAULT_AP_RADII = [1.5, 3, 8] 

    def __init__(self, pixfn, lcfn, transfn, tlimits=[-np.inf,np.inf], 
                 tex=None):
        hduL = fits.open(pixfn)
        self.pixfn = pixfn
        self.lcfn = lcfn
        self.transfn = transfn
        self.kepmag = hduL[0].header['KEPMAG']
        self.basename = os.path.splitext(lcfn)[0]
        self.starname = os.path.basename(self.basename)
        self.im_header = hduL[1].header

        # Define skeleton light curve. This pandas DataFrame contains all
        # the columns that don't depend on which aperture is used.
        im, x, y = imagestack.read_imagestack(
            pixfn, tlimits=tlimits, tex=tex
            )

        self.x = x
        self.y = y
        self.im = im 
        medframe = im.get_medframe()
        medframe.fill_value = 0
        self.medframe = medframe.filled()

        # As the reference image to construct apertures, use the 90th
        # percentile flux value. When the bleed columns move around,
        # we want to capture all the photometry.
        ap_im = self.im.get_percentile_frame(90)
        ap_im.fill_value = 0
        ap_im = ap_im.filled()
        self.ap_im = ap_im
        
    def get_aperture(self, ap_type, npix):
        """Convenience function for defining apertures"""
        ap = apertures.Aperture(
            self.ap_im, self.im_header, self.x, self.y, ap_type, npix
            )
        return ap
        
    def set_lc0(self, ap_type, npix):
        """
        Set Skeleton Lightcurve

        :param r0: radius of aperture used to create skeleton light curve
        :type r0: float
        """
        # Define skeleton light-curve
        # Include pixel transformation information 
        self.im.ap = apertures.Aperture(
            self.ap_im, self.im_header, self.x, self.y, ap_type, npix
            )
        self.im.set_fbackground()

        lc = self.im.ts 
        trans,pnts = read_channel_transform(self.transfn)
        trans['roll'] = trans['theta'] * 2e5
        # Hack select a representative star to use for x-y position 
        sqdist = (
            (np.median(pnts['x'],axis=1) - 500)**2 +
            (np.median(pnts['y'],axis=1) - 500)**2 
            )
        pnts500 = pnts[np.argmin(sqdist)]
        pnts = pd.DataFrame(pnts[0]['cad'.split()])
        pnts['xpr'] = pnts500['xpr']
        pnts['ypr'] = pnts500['ypr']

        trans = pd.concat([trans,pnts],axis=1)
        lc['fsap'] = self.im.get_sap_flux()
        norm = Normalizer(lc['fsap'].median())
        lc['f'] = norm.norm(lc['fsap'])
        lc = pd.merge(trans,lc,on='cad')
        lc['fmask'] = lc['f'].isnull() | lc['thrustermask'] | lc['bgmask']
        lc['fdtmask'] = lc['fmask'].copy()
        self.lc0 = lc

    def _get_dfaper_row(self):
        """Return an empty dictionary to store info from aperture scan"""
        d = dict(
            aper=None,  npix=None, to_fits=False, 
            fits_group='', phot=None, noise=None,
            )
        return d

    def detrend_dfaper(self, dfaper):
        """
        Loops over _detrend_dfaper_row
        """
        for i in range(len(dfaper)):
            dfaper_row = dfaper[i]
            dfaper[i] = self._detrend_dfaper_row(dfaper_row)

        return dfaper

    def get_dfaper_default(self):
        dfaper = []

        # Default apertures. These are automatically written out
        for r in self.DEFAULT_AP_RADII:
            d = self._get_dfaper_row()
            npix = np.pi * r **2 
            aper = self.get_aperture('circular', npix)            
            d['aper'] = aper
            d['npix'] = npix
            d['to_fits'] = True 
            d['fits_group'] = aper.name
            dfaper.append(d)

        return dfaper

    def get_dfaper_kepmag_to_npix(self):
        dfaper = []
        for npix in kepmag_to_npix(self.kepmag):
            d = self._get_dfaper_row()
            ap_type = npix_to_ap_type(npix)
            aper = self.get_aperture(ap_type, npix)
            d['aper'] = aper
            d['npix'] = npix
            d['fits_group'] = aper.name
            dfaper.append(d)
        dfaper = pd.DataFrame(dfaper)
        return dfaper 

    def optimize_aperture(self):
        hduL = fits.open(self.pixfn)
        # Start with number of pixels in the Kepler aperture
        npix0 = (hduL[2].data==3).sum() 
        if npix0==0:
            npix0 = 12
            print "No aperture from fits file, chosing npix={}".format(npix0)
        
        dfaper = []
        res = minimize(
            self.noise_aperture, npix0, args=(dfaper,),method='Powell' ,
            options=dict(xtol=1,direc=[1],ftol=1)
            )
        return dfaper

    def get_diagnostic_info(self, d):
        sdisp = "starname=%s " % self.starname
        sdisp += "r=%(r).1f " % d

        for k in "f fdt".split():
            k = k+'_'+noisename
            sdisp += "%s=%.1f " % (k,d[k] )
        return sdisp

    def name_mag(self):
        """Return formatted name and magnitdue"""
        return "EPIC-%s, KepMag=%.1f" % (self.starname,self.kepmag)

    def print_parameters(self):
        print "pixfn = {}".format(self.pixfn)
        print "lcfn = {}".format(self.lcfn)
        print "transfn = {}".format(self.transfn)

    @contextlib.contextmanager
    def FigureManager(self,suffix):
        """
        A small context manager that pops figures and resolves the output
        filepath
        """
        plt.figure() # Executes before code block
        yield # Now run the code block
        figpath = self.basename+suffix
        plt.savefig(figpath,dpi=160)
        plt.close('all')
        print "created %s " % figpath
    
    def to_fits(self, lcfn):
        dfaper = self.dfaper
        print "saving to {}".format(lcfn) 
        for i,row in dfaper[dfaper.to_fits].iterrows():
            print "saving to {}".format(row.fits_group) 
            row.phot.to_fits(lcfn,row.fits_group)

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
