"""
The `pixel decorrelation` pipeline
"""
import numpy as np
import pandas as pd

import pixdecor
import phot
import plotting
from pipeline_core import Pipeline, white_noise_estimate
from lightcurve import Lightcurve,Normalizer
from config import bjd0, noisekey, noisename

class PipelinePixDecor(Pipeline):
    """
    Pipeline Object for running the pixel decorrelation pipeline

    :param pixfn: path to pixel file
    :type pixfn: str

    :param lcfn: path to pixel file
    :type lcfn: str

    :param tranfn: path to pixel file
    :type tranfn: str
    """
    def set_hyperparameters(self):
        """
        Sets the hyperparameters using information from the skeleton
        lightcurve
        """
        tchunk = 10 # split lightcurve in to tchunk-day long segments
        self.sigma_n = white_noise_estimate(self.kepmag)
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

    def raw_corrected(self):
        dmin = dict(self.dfaper.iloc[0])
        dmin['noisename'] = noisename
        dmin['raw'] = dmin['f_'+noisename]
        dmin['cor'] = dmin['fdt_'+noisename]
        dmin['fac'] = dmin['raw'] / dmin['cor'] *100
        outstr = "Noise Level (%(noisename)s) : Raw={raw:.1f} (ppm), Corrected={cor:.1f} (ppm); Improvement = {fac:.1f} %%".format(**dmin)
        
        return outstr

def run(pixfn, lcfn, transfn, tlimits=[-np.inf,np.inf], tex=None, 
             debug=False, ap_select_tlimits=None):
    """
    Run the pixel decorrelation on pixel file
    """

    # Small, med, and large apertures 
    DEFAULT_AP_RADII = [1.5, 3, 8] 

    pipe = PipelinePixDecor(
           pixfn, lcfn,transfn, tlimits=tlimits, tex=None
           )
    
    pipe.print_parameters()
    pipe.set_lc0('circular',10)
    pipe.set_hyperparameters()
    pipe.reject_outliers()
    apers = pipe.get_default_apertures()

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
