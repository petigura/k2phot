"""k2phot pipeline

Contains Pipeline object that facilitates the photometry
pipeline. Also contains run_pipeline calls the relavent methods on
Pipeline instance in the right order. Core algorithms belong in
pixdecor whereas book keeping code belongs here.

"""
import os
import contextlib

import numpy as np
from astropy.io import fits
import pandas as pd

import pixdecor
import plotting
import imagestack 
import apertures
from lightcurve import Lightcurve,Normalizer
from numpy import ma
from config import bjd0
from channel_transform import read_channel_transform
from config import bjd0, noisekey, noisename, rbg
import phot
from matplotlib import pylab as plt

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
        
    def get_default_apertures(self):
        # Set default apertures to scan over
        _apertures = []
        for npix in pixdecor.kepmag_to_npix(self.kepmag):
            ap_type = pixdecor.npix_to_ap_type(npix)
            ap = self.get_aperture(ap_type, npix)
            _apertures.append(ap)
        return _apertures

    def set_lc0(self, ap_type, npix):
        """
        Set Skeleton Lightcurve

        :param r0: radius of aperture used to create skeleton light curve
        :type r0: float
        """
        # Define skeleton light-curve
        # Include pixel transformation information 
        
        im_aper = np.nanpercentile(self.im.flux,90,axis=0)
        im_aper = ma.masked_invalid(im_aper)
        im_aper.fill_value = 0 
        im_aper = im_aper.filled()

        self.im.ap = apertures.Aperture(
            im_aper, self.im_header, self.x, self.y, ap_type, npix
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

    ### k2phot pipeline ###

    def set_hyperparameters(self):
        """
        Sets the hyperparameters using information from the skeleton
        lightcurve
        """

        tchunk = 10 # split lightcurve in to tchunk-day long segments

        self.sigma_n = pixdecor.white_noise_estimate(self.kepmag)
        nchunks = self.lc0['t'].ptp() / tchunk
        nchunks = int(nchunks)
        sigma = map(lambda x : np.std(x['f']), np.array_split(self.lc0,nchunks))
        self.sigma = np.median(sigma)
        self.length_t = 4
        self.length_roll = 10

    def reject_outliers(self):
        """
        Perform an initial run with a single aperture size
        grab the outliers from this run and use it in subsequent runs
        """
        # Set the values of the GP hyper parameters
        lc = pixdecor.detrend_t_roll_2D_segments( 
            self.lc0, self.sigma, self.length_t, self.length_roll,self.sigma_n,
            reject_outliers=True, segment_length=20
            )
        self.fdtmask = lc['fdtmask'].copy() 

    def detrend_t_roll_2D(self, ap):
        # Create new lightcurve from skeleton
        lc = self.lc0.copy()
        self.im.ap = ap
        lc['fsap'] = self.im.get_sap_flux()
        norm = Normalizer(lc['fsap'].median()) 
        lc['f'] = norm.norm(lc['fsap'])        
        lc['fdtmask'] = self.fdtmask 
        lc = pixdecor.detrend_t_roll_2D( 
            lc, self.sigma, self.length_t, self.length_roll,self.sigma_n, 
            reject_outliers=False
            )

        # Cast as Lightcurve object
        lc = Lightcurve(lc)
        
        noise = []
        for key in [noisekey,'f']:
            ses = lc.get_ses(key) 
            ses = pd.DataFrame(ses)
            ses['name'] = ses.index
            ses['name'] = key +'_' + ses['name'] 
            ses = ses[['name','value']]
            noise.append(ses)

        noise = pd.concat(noise,ignore_index=True)

        for k in self.unnormkeys:
            lc[k] = norm.unnorm(lc[k])

        _phot = phot.Photometry(
            self.medframe, lc, ap.weights, ap.verts, noise, pixfn=self.pixfn
            )

        return _phot

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

    def raw_corrected(self):
        dmin = dict(self.dfaper.iloc[0])
        dmin['noisename'] = noisename
        dmin['raw'] = dmin['f_'+noisename]
        dmin['cor'] = dmin['fdt_'+noisename]
        dmin['fac'] = dmin['raw'] / dmin['cor'] *100
        outstr = "Noise Level (%(noisename)s) : Raw=%(raw).1f (ppm), Corrected=%(cor).1f (ppm); Improvement = %(fac).1f %%" % dmin
        
        return outstr

def run(pixfn, lcfn, transfn, tlimits=[-np.inf,np.inf], tex=None, 
             debug=False, ap_select_tlimits=None):
    """
    Run the pixel decorrelation on pixel file
    """

    # Small, med, and large apertures 
    DEFAULT_AP_RADII = [1.5, 3, 8] 
    print "pixfn = {}".format(pixfn)
    print "lcfn = {}".format(lcfn)
    print "transfn = {}".format(transfn)
    print "ap_select_tlimits = {}".format(ap_select_tlimits)

    pipe = Pipeline(
           pixfn, lcfn,transfn, tlimits=tlimits, tex=None
           )

    pipe.set_lc0('circular',10)
    pipe.set_hyperparameters()
    pipe.reject_outliers()
    apers = pipe.get_default_apertures()
    #apers = []

    df = pd.DataFrame(dict(aper=apers))
    df['default'] = False # 
    df['fits_group'] = '' 
    for r in DEFAULT_AP_RADII:
        npix = np.pi * r **2 
        aper = pipe.get_aperture('circular', npix)
        row = dict(aper=aper, default=True, fits_group=aper.name)
        row  = pd.Series( row ) 
        df = df.append(row, ignore_index=True)

    df['phot'] = None
    df['noise'] = None
    df['npix'] = 0
    for i,row in df.iterrows():
        _phot = pipe.detrend_t_roll_2D(row.aper)
        df.ix[i,'phot'] = _phot
        ap_noise = _phot.ap_noise
        ap_noise.index = ap_noise.name
        df.ix[i,'noise'] = ap_noise.ix[noisekey+'_'+noisename].value
        df.ix[i,'npix'] = row.aper.npix
        df.ix[i,'fits_group'] = row.aper.name
        
    df['to_fits'] = False
    row = df.loc[df.noise.idxmin()].copy()
    row['to_fits'] = True
    row['fits_group'] = 'optimum'

    df = df.append(row, ignore_index=True)
    df.loc[df.default,'to_fits'] = True
    pipe.dfaper = df
    print df.sort('npix')['fits_group npix noise to_fits'.split()]

    print "saving to {}".format(lcfn) 
    for i,row in df[df.to_fits].iterrows():
        row.phot.to_fits(lcfn,row.fits_group)

    if 0:
        from matplotlib import pylab as plt

        plt.ion()
        plt.figure()
        import pdb;pdb.set_trace()


    _phot = phot.read_fits(lcfn,'optimum')
    with pipe.FigureManager('_0-median-frame.png'):
        plotting.phot.medframe(_phot)

    with pipe.FigureManager('_1-background.png'):
        plotting.phot.background(_phot)

    with pipe.FigureManager('_2-noise_vs_aperture_size.png'):
        plotting.pipeline.noise_vs_aperture_size(pipe)

    with pipe.FigureManager("_3-fdt_t_roll_2D.png"):
        plotting.phot.detrend_t_roll_2D(_phot)

    with pipe.FigureManager("_4-fdt_t_roll_2D_zoom.png"):
        plotting.phot.detrend_t_roll_2D(_phot,zoom=True)

    with pipe.FigureManager("_5-fdt_t_rollmed.png"):
        plotting.phot.detrend_t_rollmed(_phot)
