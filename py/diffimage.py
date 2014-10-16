"""
Module for computing difference images in and out of transit
"""
import cPickle as pickle
from argparse import ArgumentParser

import h5py
from scipy import ndimage as nd
import pandas as pd
from astropy.io import fits
from matplotlib.pylab import *

from image_registration import register_images
from pixel_decorrelation import get_stars_pix,plot_label,subpix_reg_stack,imshow2,log10scale
from pixel_io import loadPixelFile

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

tlblkw = dict(cfrac=0.5,cpad=1)
class DiffImage(object):
    """
    
    """
    def __init__(self,pixelfile,tlimits=None):
        cube,headers = loadPixelFile(pixelfile,bjd0=0,tlimits=tlimits)        

        # Remove nans and median detrend
        flux = ma.masked_invalid(cube['flux'])
        flux = ma.masked_invalid(flux)
        flux.fill_value = 0
        flux = flux.filled()
        medflux = ma.median(flux.reshape(flux.shape[0],-1),axis=1)
        flux = flux - medflux[:,newaxis,newaxis]

        # Put images in to DataFrame
        flux = [f for f in flux]
        df = pd.DataFrame(dict(cad=cube['CADENCENO'],flux=flux))
        df['cad'] = df.cad.astype(int)

        # Only select images used in LC
#        df = pd.merge(df,lc,on='cad')
        df.index = df.cad
        self.df = df
        self.t = cube['TIME']
        self.pixelfile = pixelfile

    def register_images(self):
        print "Calculating relative offsets between frames"
        df = self.df
        flux  = np.array(df.flux.tolist())
        dx,dy = subpix_reg_stack(flux) 
        df['dc'] = dx # Handle the weird convention
        df['dr'] = dy

        def shift(x):
            dc = x['dc']
            dr = x['dr']
            flux = x['flux']
            return nd.shift(flux,[-dr,-dc],order=4)

        fluxs = []
        for i in df.index:
            fluxs+=[shift(df.ix[i])]
        df['fluxs'] = fluxs

        # Compute shifts between registered images (should be nearly 0)
        dxs,dys = subpix_reg_stack(np.array(fluxs))
        df['dcs'] = dxs # Handle the weird convention
        df['drs'] = dys
        self.df = df

    def split_inout(self,P,t0,tdur):
        """
        Split light-curve according to in/out of transit
        """
        tlbl = transLabel(self.t,P,t0,tdur,**tlblkw)
 
        self.P = P
        self.t0 = t0
        self.tdur = tdur
        self.df_in = self.df[tlbl['tRegLbl']>=0]
        self.df_out = self.df[tlbl['cRegLbl']>=0]
        self.tlbl = tlbl

        def med_fluxs(df):            
            return np.median(array(df['fluxs'].tolist()),axis=0)

        self.med_image_in = med_fluxs(self.df_in)
        self.med_image_out = med_fluxs(self.df_out)
        self.diffms = self.med_image_out - self.med_image_in

def plot_diffimage(dim,gridfile=None):
    df = dim.df

    fig,axL = subplots(ncols=2,nrows=2,figsize=(8,8))
    sca(axL[0,0])
    title('Shifts from frame 0')
    plot(df.dc,df.dr,'.',label='Raw Images')
    plot(df.dcs,df.drs,'.',
         label='Shifted Images\n4-th order poly\nshould be small')
    xlabel('$\delta$ column (pixels)')
    ylabel('$\delta$ row (pixels)')
    legend()

    if gridfile!=None:
        sca(axL[0,1])

        import h5py
        fit = h5py.File(gridfile)['dv']['fit']
        t = fit['t'][:]
        f = fit['f'][:]

        tlbl = transLabel(t,dim.P,0,dim.tdur,**tlblkw)
        btrans = tlbl['tRegLbl'] >=0
        bcont = tlbl['cRegLbl'] >=0

        plot(t[btrans],f[btrans],'.',label='In Transit')
        plot(t[bcont],f[bcont],'.',label='Out of Transit')
        xlabel('t - t0 (days)')
        ylabel('flux')
        legend()

    sca(axL[1,0])
    refimage = df.iloc[0]['flux']
    catcut, shift = get_stars_pix(dim.pixelfile,refimage)
    epic = fits.open(dim.pixelfile)[0].header['KEPLERID']
    plot_label(log10scale(refimage),catcut,epic,colorbar=False)
    title('log10( Reference Frame )')

    sca(axL[1,1])
    imshow2(dim.diffms)
    plot_label(dim.diffms,catcut,epic,colorbar=False)
    title('Difference Image\nOut of Transit - In Transit')
    gcf().set_tight_layout(True)

def t0shft(t,P,t0):
    """
    Epoch shift

    Find the constant shift in the timeseries such that t0 = 0.

    Parameters
    ----------
    t  : time series
    P  : Period of transit
    t0 : Epoch of one transit (does not need to be the first).

    Returns
    -------
    dt : Amount to shift by.
    """
    t  = t.copy()
    dt = 0

    t  -= t0 # Shifts the timeseries s.t. transits are at 0,P,2P ...
    dt -= t0

    # The first transit is at t =  nFirstTransit * P
    nFirstTrans = np.ceil(t[0]/P) 
    dt -= nFirstTrans*P 

    return dt

def transLabel(t,P,t0,tdur,cfrac=1,cpad=0):
    """
    Transit Label

    Mark cadences as:
    - transit   : in transit
    - continuum : just outside of transit (used for fitting)
    - other     : all other data

    Parameters
    ----------
    t     : time series
    P     : Period of transit
    t0    : epoch of one transit
    tdur  : transit duration
    cfrac : continuum defined as points between tdur * (0.5 + cpad)
            and tdur * (0.5 + cpad + cfrac) of transit midpoint cpad
    cpad  : how far away from the transit do we start the continuum
            region in units of tdur.


    Returns
    -------

    A record array the same length has the input. Most of the indecies
    are set to -1 as uninteresting regions. If a region is interesting
    (transit or continuum) I label it with the number of the transit
    (starting at 0).

    - tLbl      : Index closest to the center of the transit
    - tRegLbl   : Transit region
    - cRegLbl   : Continuum region
    - totRegLbl : Continuum region and everything inside

    Notes
    -----
    tLbl might not be the best way to find the mid transit index.  In
    many cases, dM[rec['tLbl']] will decrease with time, meaning there
    is a cumulative error that's building up.

    """

    t = t.copy()
    t += t0shft(t,P,t0)

    names = ['totRegLbl','tRegLbl','cRegLbl','tLbl']
    rec  = np.zeros(t.size,dtype=zip(names,[int]*len(names)) )
    for n in rec.dtype.names:
        rec[n] -= 1
    
    iTrans   = 0 # number of transit, starting at 0.
    tmdTrans = 0 # time of iTrans mid transit time.  
    while tmdTrans < t[-1]:
        # Time since mid transit in units of tdur
        t0dt = np.abs(t - tmdTrans) / tdur 
        it   = t0dt.argmin()
        bt   = t0dt < 0.5
        bc   = (t0dt > 0.5 + cpad) & (t0dt < 0.5 + cpad + cfrac)
        btot = t0dt < 0.5 + cpad + cfrac
        
        rec['tRegLbl'][bt] = iTrans
        rec['cRegLbl'][bc] = iTrans
        rec['tLbl'][it]    = iTrans
        rec['totRegLbl'][btot] = iTrans

        iTrans += 1 
        tmdTrans = iTrans * P

    return rec

if __name__ == "__main__":
    p = ArgumentParser(description='Difference Images')
    p.add_argument('pixelfile',type=str,help='*.fits file')
    p.add_argument('gridfile',type=str,help='*.grid.h5 file')
    p.add_argument('--P',type=float,help='Transit Period')
    p.add_argument('--t0',type=float,help='Transit Epoch')
    p.add_argument('--tdur',type=float,help='Transit Duration')
    p.add_argument('--tmin',type=float,default=-np.inf)
    p.add_argument('--tmax',type=float,default=np.inf)
    p.add_argument('--showplot',action='store_true')

    args  = p.parse_args()
    pixelfile = args.pixelfile
    gridfile = args.gridfile
    P = args.P
    t0 = args.t0
    tdur = args.tdur
    tmin = args.tmin
    tmax = args.tmax
    showplot = args.showplot

    diffim = DiffImage(pixelfile,tlimits=[tmin,tmax])
    diffim.register_images()
    diffim.split_inout(P,t0,tdur)
    plot_diffimage(diffim,gridfile=gridfile)
    plotfile = args.gridfile.replace('grid.h5','diffim.png')
    if showplot:
        show()
    
    gcf().savefig(plotfile)
