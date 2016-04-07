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
from config import bjd0, noisekey, noisename
import copy
import plotting

npix_scan_fac = 4
npix_scan_trials = 6

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
        ap_im = self.im.get_percentile_frame(99.9)
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

    def _detrend_dfaper_row(self, d):
        """
        Takes a dictionary with detrending parameters, runs detrend() and
        stores results.

        :param d: 
            dictionary with the following keys:
            - phot
            - noise
            - noisename
            - npix
            - fits_group
        """

        aper = d['aper']
        _phot = self.detrend(aper)
        ap_noise = _phot.ap_noise
        ap_noise.index = ap_noise.name

        # Adding extra info to output dictionary
        d['phot'] = _phot
        d['noise'] = ap_noise.ix[noisekey+'_'+noisename].value
        d['noisename'] = noisename
        d['npix'] = aper.npix
        d['fits_group'] = aper.name
        return d

    def _get_dfaper_row(self, aper=None):
        """Return an empty dictionary to store info from aperture scan"""
        d = dict(
            aper=None,  npix=None, to_fits=False, 
            fits_group='', phot=None, noise=None,
            )
        if aper==None:
            return d

        d['aper'] = aper
        d['npix'] = aper.npix
        d['fits_group'] = aper.name
        return d

    def aperture_scan(self, dfaper):
        """
        Loops over `_detrend_dfaper_row`
        """
        for i in range(len(dfaper)):
            dfaper_row = dfaper[i]
            dfaper[i] = self._detrend_dfaper_row(dfaper_row)
        return dfaper

    def aperture_polish_iteration(self,dfaper0):
        dfaper = copy.deepcopy(dfaper0)

        dfaper = pd.DataFrame(dfaper)
        dfaper = dfaper.sort('npix')
        dfaper.index = range(len(dfaper))

        # Check to see if best aperture is bounded on both sides
        idx_best = dfaper.noise.idxmin()
        npix_best1 = dfaper.ix[idx_best,'npix']
        npix_min = dfaper.npix.min()
        npix_max = dfaper.npix.max()
        print "npix_best = {}, npix_min = {}, npix_max = {}".format(
            npix_best1, npix_min, npix_max)
 
        is_npix_best_bounded = (npix_best1 > npix_min) & (npix_best1 < npix_max)
        if is_npix_best_bounded is False:
            return dfaper
        
        idx_fit = np.arange(3) - 1 + idx_best
        noise_fit = dfaper.ix[idx_fit].noise
        npix_fit = dfaper.ix[idx_fit].npix
        c2, c1, c0 = np.polyfit(npix_fit, noise_fit,2)

        npix_best2 = -0.5 * c1 / c2
        npix_best2 = np.round( npix_best2 )
        print "new guess for npix_best2 = {}".format(npix_best2)

        d = self._get_dfaper_row(self.get_aperture('region',npix_best2))
        d = self._detrend_dfaper_row(d)
        dfaper0.append(d)
        return dfaper0, npix_best1, npix_best2

    def aperture_polish(self,dfaper, max_iterations=3):
        i = 0
        while i < max_iterations:
            dfaper, npix_best1, npix_best2 = \
                self.aperture_polish_iteration(dfaper)
            if npix_best1==npix_best2:
                return dfaper

            i += 1

        return dfaper

    def get_dfaper_default(self):
        """
        Get default values of dfaper. Returns list of dictionaries
        """
        dfaper = []
        for r in self.DEFAULT_AP_RADII:
            npix = np.pi * r **2 
            aper = self.get_aperture('circular', npix)            
            d = self._get_dfaper_row(aper=aper)
            d['to_fits'] = True
            dfaper.append(d)

        return dfaper

    def get_dfaper_scan(self):
        """
        Return trial values of dfaper computed from Kepler magnitude
        """
        dfaper = []
        for npix in kepmag_to_npix_scan(self.kepmag):
            aper = self.get_aperture('region', npix)
            d = self._get_dfaper_row(aper=aper)
            dfaper.append(d)

        return dfaper 

    def get_aperture_guess(self):
        """
        Return trial values of dfaper computed from Kepler magnitude
        """
        npix = kepmag_to_npix(self.kepmag)
        aper = self.get_aperture('region', npix)
        return aper


    def get_diagnostic_info(self, d):
        sdisp = "starname=%s " % self.starname
        sdisp += "r=%(r).1f " % d

        for k in "f fdt".split():
            k = k+'_'+noisename
            sdisp += "%s=%.1f " % (k,d[k] )
        return sdisp

    def get_noise(self,lc):
        """
        Get noise DataFrame. 

        Normalize the lightcurve. Compute noise
        characteristics. Return DataFrame

        :param lc:  Light curve. Should not be normalized.
        :type pandas.DataFrame:

        """

        lc = lc.copy()
        norm = Normalizer(lc['fsap'].median()) 
        lc['f'] = lc['fsap']

        # Cast as Lightcurve object
        lc = Lightcurve(lc)
        noise = []
        for key in [noisekey,'f']:
            lc[key] = norm.norm(lc[key])
            ses = lc.get_ses(key) 
            ses = pd.DataFrame(ses)
            ses['name'] = ses.index
            ses['name'] = key +'_' + ses['name'] 
            ses = ses[['name','value']]
            noise.append(ses)

        noise = pd.concat(noise,ignore_index=True)
        return noise

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

    def plot_diagnostics(self):
        if 0:
            from matplotlib import pylab as plt
            plt.ion()
            plt.figure()
            import pdb;pdb.set_trace()

        _phot = phot.read_fits(self.lcfn,'optimum')
        with self.FigureManager('_0-aperture.png'):
            plotting.phot.aperture(_phot)

        with self.FigureManager('_1-background.png'):
            plotting.phot.background(_phot)

        with self.FigureManager('_2-noise_vs_aperture_size.png'):
            plotting.pipeline.noise_vs_aperture_size(self)

        with self.FigureManager("_3-fdt_t_roll_2D.png"):
            plotting.phot.detrend_t_roll_2D(_phot)

        with self.FigureManager("_4-fdt_t_roll_2D_zoom.png"):
            plotting.phot.detrend_t_roll_2D(_phot,zoom=True)

        with self.FigureManager("_5-fdt_t_rollmed.png"):
            plotting.phot.detrend_t_rollmed(_phot)
    
    def to_fits(self, lcfn):
        dfaper = self.dfaper
        print "saving to {}".format(lcfn) 
        for i,row in dfaper[dfaper.to_fits].iterrows():
            print "saving to {}".format(row.fits_group) 
            row.phot.to_fits(lcfn,row.fits_group)

def kepmag_to_npix_scan(kepmag):
    npixfit = kepmag_to_npix(kepmag,plot=False)
    npixfit_min = npixfit / npix_scan_fac
    npixfit_max = npixfit * npix_scan_fac
    npix = np.logspace( 
        np.log10(npixfit_min), np.log10(npixfit_max), npix_scan_trials
        ) 
    npix = np.round(npix).astype(int)
    return npix

def kepmag_to_npix(kepmag,plot=False):
    fn = 'optimal_aperture_sizes.csv'
    dirn = os.path.dirname( os.path.dirname( __file__ ) )
    fn = os.path.join( dirn, 'data/', fn )
    df = pd.read_csv(fn)
    df['lognpix'] = np.log10(df.npix)
    p1 = np.polyfit(df.kepmag,df.lognpix,1)
    npixfit = 10**np.polyval(p1,kepmag) 

    if plot:
        plt.semilogy()
        df['npixfit'] = 10**np.polyval(p1,df.kepmag)
        df['npixfit_upper'] = df['npixfit'] * npix_scan_fac
        df['npixfit_lower'] = df['npixfit'] / npix_scan_fac
        plt.plot(df.kepmag,df.npix,'.')
        plt.plot(df.kepmag,df.npixfit)
        plt.plot(df.kepmag,df.npixfit_lower)
        plt.plot(df.kepmag,df.npixfit_upper)


    return npixfit

def kepmag_to_npix2(kepmag):
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





