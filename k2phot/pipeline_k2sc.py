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



def run(pixfn, lcfn, transfn, tlimits=[-np.inf,np.inf], tex=None, 
             debug=False, ap_select_tlimits=None):
    """
    Run the pixel decorrelation on pixel file
    """

    # Small, med, and large apertures 
    DEFAULT_AP_RADII = [1.5, 3, 8] 

    pipe = PipelineK2SC(
           pixfn, lcfn, transfn, tlimits=tlimits, tex=None
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
