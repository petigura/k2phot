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

    def detrend(self, ap):
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

        # Un-normalize the data
        for k in self.unnormkeys:
            lc[k] = norm.unnorm(lc[k])

        noise = self.get_noise(lc)

        _phot = phot.Photometry(
            self.medframe, lc, ap.weights, ap.verts, noise, pixfn=self.pixfn
            )
        return _phot

    def noise_aperture(self, npix, dfaper):
        """
        Noise for different pixel sizes

        Call ala

        >>> dfaper = []
        >>> minimize(noise_aperture, npix0, dfaper)

        Returns
        -------
        noise at each value of pixel
        """
        if hasattr(npix, '__iter__'):
            npix = npix[0]
        npix = int(npix)
        if npix < 4:
            return 1e6

        d = self._get_dfaper_row()
        aper = self.get_aperture('region', npix)
        d['aper'] = aper
        d = self._detrend_dfaper_row(d)
        dfaper += [d]
        print "{aper} noise ({noisename}) = {noise:.1f} ppm".format(**d)
        return d['noise']

def run(pixfn, lcfn, transfn, tlimits=[-np.inf,np.inf], tex=None, 
             debug=False, ap_select_tlimits=None):
    """
    Run the pixel decorrelation on pixel file
    """

    pipe = PipelinePixDecor(
           pixfn, lcfn,transfn, tlimits=tlimits, tex=None
           )
    
    pipe.print_parameters()
    pipe.set_lc0('circular',10)
    pipe.set_hyperparameters()
    pipe.reject_outliers()

    dfaper = pipe.get_dfaper_default()
    dfaper = pipe.detrend_dfaper(dfaper)
    dfaper_optimize = pipe.optimize_aperture()
    dfaper = dfaper + dfaper_optimize
    dfaper = pd.DataFrame(dfaper)
    row = dfaper.loc[dfaper.noise.idxmin()].copy()
    row['to_fits'] = True
    row['fits_group'] = 'optimum'
    dfaper = dfaper.append(row, ignore_index=True)
    pipe.dfaper = dfaper
    print dfaper.sort('npix')['fits_group npix noise to_fits'.split()]
    pipe.to_fits(pipe.lcfn)

    if 0:
        from matplotlib import pylab as plt
        plt.ion()
        plt.figure()
        import pdb;pdb.set_trace()

    _phot = phot.read_fits(lcfn,'optimum')
    with pipe.FigureManager('_0-aperture.png'):
        plotting.phot.aperture(_phot)

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
