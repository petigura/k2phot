"""
The `pixel decorrelation` pipeline
"""
import numpy as np
import pandas as pd

import pixdecor
import phot
import plotting
from pipeline_core import Pipeline, white_noise_estimate
from lightcurve import Lightcurve, Normalizer
from config import bjd0, noisekey, noisename

import k2sc.k2sc
import k2sc.k2data
from astropy.io import fits
import pyfits as pf

class PipelineK2SC(Pipeline):
    """
    Pipeline Object for running the pixel decorrelation pipeline

    :param pixfn: path to pixel file
    :type pixfn: str

    :param lcfn: path to pixel file
    :type lcfn: str

    :param tranfn: path to pixel file
    :type tranfn: str
    """

    def get_K2Data(self, ap):
        """
        Returns the an instance of K2Data passed around the `k2sc` pipeline
        """
        self.set_lc0(ap.ap_type,ap.npix)
        hduL = fits.open(self.pixfn)
        quality = hduL[1].data['QUALITY']
        fluxes = self.lc0['fsap']
        time = self.lc0['t'] - bjd0
        cadence = self.lc0['cad']

        # Stand in for true photometric errors
        errors = fluxes * (60 * 30 * self.lc0['fsap'].median() )**-0.5
        pos = self.lc0['xpr ypr'.split()]
        pos -= pos.mean()
        head = pf.getheader(self.pixfn, 0)
#        head = fits.getheader(self.pixfn, 0)
        epic = head['KEPLERID']
        x = pos['xpr']
        y = pos['ypr']

        time = np.array(time)
        cadence = np.array(cadence)
        quality = np.array(quality)
        fluxes = np.array(fluxes)
        errors = np.array(errors)
        x = np.array(x)
        y = np.array(y)
        _K2Data = k2sc.k2data.K2Data(
            epic,
            time=time,
            cadence=cadence,
            quality=quality,
            fluxes=fluxes,
            errors=errors,
            x=x,
            y=y,
            sap_header=head
        )    
        
        return _K2Data

    def detrend(self, ap):
        # Create new lightcurve from skeleton

        k2data = self.get_K2Data(ap)
        splits = [2180]
        results = k2sc.k2sc.main(k2data, splits, de_niter=1)
        return results
        

#        self.im.ap = ap
#        lc['fsap'] = self.im.get_sap_flux()
#        norm = Normalizer(lc['fsap'].median()) 
#        lc['f'] = norm.norm(lc['fsap'])        
#        lc['fdtmask'] = self.fdtmask 
#        lc = pixdecor.detrend_t_roll_2D( 
#            lc, self.sigma, self.length_t, self.length_roll,self.sigma_n, 
#            reject_outliers=False
#            )
#
#        # Cast as Lightcurve object
#        lc = Lightcurve(lc)
#        noise = []
#        for key in [noisekey,'f']:
#            ses = lc.get_ses(key) 
#            ses = pd.DataFrame(ses)
#            ses['name'] = ses.index
#            ses['name'] = key +'_' + ses['name'] 
#            ses = ses[['name','value']]
#            noise.append(ses)
#
#        noise = pd.concat(noise,ignore_index=True)
#        for k in self.unnormkeys:
#            lc[k] = norm.unnorm(lc[k])
#
#
#
#        _phot = phot.Photometry(
#            self.medframe, lc, ap.weights, ap.verts, noise, pixfn=self.pixfn
#            )
#
#        return _phot


    def _detrend_dfaper_row(self, d):
        """
        Detrend dfaper_row

        dfaper_row : dictionary with the aperture defined
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


