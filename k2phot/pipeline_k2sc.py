"""
The `pixel decorrelation` pipeline
"""

import copy

import numpy as np
import pandas as pd
from astropy.io import fits

import k2sc.k2sc
import k2sc.k2data

import phot
import plotting
from pipeline_core import Pipeline, white_noise_estimate
from lightcurve import Lightcurve, Normalizer
from config import bjd0, noisekey, noisename

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
    def __init__(self, pixfn, lcfn, transfn, splits, tlimits=[-np.inf,np.inf], 
                 tex=None, plot_backend='.png',aper_custom=None, xy=None,
                 transitParams=None, transitArgs=None):
        super(PipelineK2SC,self).__init__(
            pixfn, lcfn, transfn, tlimits=tlimits,tex=tex, 
            plot_backend=plot_backend, aper_custom=aper_custom,xy=xy,
            transitParams=None, transitArgs=None
            )

        self.splits = splits
        self.debug = False

    def detrend(self, ap):
        k2sc_result  = copy.deepcopy(self.k2sc_result)

        # Recompute GP correction with new aperture size
        self.set_lc0(ap.ap_type, ap.npix)
        flux = np.array(self.lc0['fsap'])
        inputs = k2sc_result.detrender.data.inputs
        mask = k2sc_result.detrender.data.mask
        kernel = k2sc_result.detrender.kernel
        splits = self.splits
        detrender = k2sc.detrender.Detrender(
            flux, inputs, mask=mask, kernel=kernel, splits=splits
            )
        tr_time, tr_position = detrender.predict(
            k2sc_result.pv, components=True
            )

        # Build up final light curve
        lc = self.lc0.copy() 
        fsapmed = lc['fsap'].median()
        lc['ftnd_t_roll_2D'] = tr_position + tr_time - fsapmed  
        lc['fdt_t_roll_2D'] = lc['fsap'] - lc['ftnd_t_roll_2D'] + fsapmed
        fsapmed = lc['fsap'].median()
        lc['ftnd_t_rollmed'] = tr_position
        lc['fdt_t_rollmed'] = lc['fsap'] - lc['ftnd_t_rollmed'] + fsapmed
        noise = self.get_noise(lc)
        _phot = phot.Photometry(
            self.medframe, lc, ap.weights, ap.verts, noise, pixfn=self.pixfn
            )
        return _phot

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
        
        head = fits.open(self.pixfn)[0].header
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

    def k2sc(self, ap):
        """
        Run the k2sc pipeline using the full k2sc pipeline that
        optimizes over the hyperparameters.
        """
        print "Computing k2sc_data, k2sc_result"
        k2data = self.get_K2Data(ap)
        if self.debug:
            k2data, results = k2sc.k2sc.main(
                k2data, self.splits, de_niter=1, de_npop=10
                )
        else:
            k2data, results = k2sc.k2sc.main(
                k2data, self.splits, de_niter=10, de_npop=100
                )
        self.k2sc_data = k2data
        self.k2sc_result = results[0]

   
def run(pixfn, lcfn, transfn, splits, tlimits=[-np.inf,np.inf], tex=None, 
             debug=False,plot_backend='.png', aper_custom=None,xy=None,
             transitParams=None, transitArgs=None):
    """
    Run the pixel decorrelation on pixel file
    """

    pipe = PipelineK2SC(
        pixfn,lcfn,transfn,splits,tlimits=tlimits, plot_backend=plot_backend,
        tex=tex, aper_custom=aper_custom,xy=xy, transitParams=None, transitArgs=None
    )
    pipe.debug = debug

    # Perform hyper parameter optimization using the best guess aperture
    ap = pipe.get_aperture_guess()
    pipe.k2sc(ap)

    if aper_custom is None:
        # Photometry with circular apertures
        dfaper_default = pipe.get_dfaper_default()
        dfaper_default = pipe.aperture_scan(dfaper_default)

        # Photometry with region apertures
        dfaper_scan = pipe.get_dfaper_scan()
        dfaper_scan = pipe.aperture_scan(dfaper_scan)
        dfaper_scan = pipe.aperture_polish(dfaper_scan)

        # Find optimal region aperture
        dfaper = dfaper_default + dfaper_scan
        dfaper = pd.DataFrame(dfaper)
        idx = dfaper[dfaper.fits_group.str.contains('region')].noise.idxmin()
    else:
        # Photometry with circular apertures
        dfaper = pipe.get_dfaper_custom()
        dfaper = pipe.aperture_scan(dfaper)
        dfaper = pd.DataFrame(dfaper)
        idx = 0 # only include one aperture
    
    row = dfaper.loc[idx].copy()
    row['to_fits'] = True
    row['fits_group'] = 'optimum'
    dfaper = dfaper.append(row, ignore_index=True)
    # Hack to get around sit where noise was not compute
    print "{} aperture sizes".format(len(dfaper))
    dfaper = dfaper.dropna()
    print "{} aperture sizes (notnull)".format(len(dfaper))
    print dfaper.sort_values(by='npix')['fits_group npix noise to_fits'.split()]

    # Save and make diagnostic plots
    pipe.dfaper = dfaper
    pipe.to_fits(pipe.lcfn)
    pipe.plot_diagnostics()

