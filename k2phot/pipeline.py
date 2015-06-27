import pixdecor
import plotting
import numpy as np
from config import bjd0

def pipeline(pixfn,lcfn,transfn, tlimits=[-np.inf,np.inf],tex=None,debug=False):
    """
    Run the pixel decorrelation on pixel file
    """
    pixdcr = pixdecor.PixDecor(
        pixfn, lcfn,transfn, tlimits=tlimits, tex=None
        )
    if debug:
        npts = len(pixdcr.lc0)
        idx = [int(0.25*npts),int(0.50*npts)]

        tlimits = [pixdcr.lc0.iloc[i]['t'] - bjd0 for i in idx]
        pixdcr = pixdecor.PixDecor(
            pixfn, lcfn,transfn, tlimits=tlimits, tex=None
        )
        pixdcr.apertures = [3,4]


    pixdcr.set_hyperparameters()
    pixdcr.reject_outliers()
    pixdcr.scan_aperture_size()
    pixdcr.to_fits(lcfn)

    if 0:
        from matplotlib import pylab as plt
        plt.ion()
        plt.figure()
        plotting.plot_medframe(pixdcr)
        import pdb;pdb.set_trace()

    with pixdcr.FigureManager('_0-median-frame.png'):
        plotting.medframe(pixdcr)

    with pixdcr.FigureManager('_1-background.png'):
        plotting.background(pixdcr)

    with pixdcr.FigureManager('_2-noise_vs_aperture_size.png'):
        plotting.noise_vs_aperture_size(pixdcr)

    with pixdcr.FigureManager("_3-fdt_t_roll_2D.png"):
        plotting.detrend_t_roll_2D(pixdcr)

    with pixdcr.FigureManager("_4-fdt_t_roll_2D_zoom.png"):
        plotting.detrend_t_roll_2D(pixdcr,zoom=True)

    with pixdcr.FigureManager("_5-fdt_t_rollmed.png"):
        plotting.detrend_t_rollmed(pixdcr)


