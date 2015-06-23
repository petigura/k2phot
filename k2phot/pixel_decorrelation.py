
import os
from cStringIO import StringIO as sio

import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pylab as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib.pylab import *
import matplotlib.gridspec as gridspec
import george
from george import kernels

from imagestack import ImageStack,read_imagestack
from channel_transform import read_channel_transform

from config import bjd0
from ses import ses_stats,total_precision_theory
from astropy.io import fits
import contextlib

os.system('echo "pixel_decorrelation modules loaded:" $(date) ')

noisekey = 'fdt_t_roll_2D' # Column to use to compute noise
noisename = 'mad_6_cad_mtd' # Noise metric to use for optimal aperture
outfmt = dict(marker='o',mew=0,mfc='r',alpha=0.5,lw=0,zorder=0,ms=8)
infmt = dict(marker='.',lw=0,zorder=5)

segments = """\
k2_camp start stop
C1 91440 92096
C1 92099 92672
C1 92677 93270
C1 93426 94182
C1 94188 94892
C1 94894 95348
"""
segments = pd.read_table(sio(segments),sep='\s*')

def plotseg():
    for i in segments.iterrows():
        plt.axvspan(i[1]['start'],i[1]['stop'])


def namemag(fitsfile):
    """
    Pull name and magnitude from fits file
    """
    with fits.open(fitsfile) as hduL:
        hdr = hduL[0].header
        return "EPIC-%i, KepMag=%.1f" % (hdr['KEPLERID'],hdr['KEPMAG'])


def split_lc(lc):
    seg = segments.copy()
    camp_seg = seg
    lc_segments = []
    for i,row in camp_seg.iterrows():
        lc_seg = lc.query('%(start)i <= cad <= %(stop)i' % row)
        lc_segments +=[lc_seg]

    assert len(lc_segments) > 0, "No segments"
    return lc_segments

def square_plots(n):
    sqn = sqrt(n)
    ncol = int(sqn) + 1
    nrow = int(np.ceil( float(n) / ncol)) 
    return nrow,ncol

def roblims(x,p,fac):
    """
    Robust Limits

    Return the robust limits for a plot

    Parameters
    ----------
    x  : input array
    p : percentile (compared to 50) to use as lower limit
    fac : add some space
    """
    ps = np.percentile(x,[p,50,100-p])
    lim = ps[0] - (ps[1]-ps[0]) * fac,ps[2]+(ps[2]-ps[1]) * fac
    return lim

@contextlib.contextmanager
def FigureManager(basename,suffix=None):
    # Executes before code block
    plt.figure() 
    
    # Now run the code block
    yield

    # Executes after code block
    if basename is not None:
        figpath = basename+suffix
        plt.savefig(figpath)
        plt.close('all')
        print "created %s " % figpath

def plot_noise_vs_aperture_size(dfaper,noisename='mad_6_cad_mtd'):
    dmin = dfaper.sort(noisename).iloc[0]
    dfaper = dfaper.sort('r')
    
    plt.semilogy()
    plt.plot(dfaper.r,dfaper[noisename],'o-')
    plt.plot(dmin['r'],dmin[noisename],'or',mfc='none',ms=15,mew=2,mec='r')

    xlab = 'Target Aperture Radius [pixels]'
    txtStr = 'Minimum: %.1f ppm at R=%.1f pixels' % \
             (dmin[noisename],dmin['r'])

    plt.xlabel(xlab)
    plt.ylabel('Noise (%s) [ppm]' % noisename)
    plt.minorticks_on()

    desc = dfaper[noisename].describe()
    factor = 1.2
    yval = desc['min']/factor,desc['max']*factor
    plt.ylim(yval)
    yticks = np.logspace(np.log10(yval[0]), np.log10(yval[1]), 8)
    plt.yticks(yticks, ['%i' % el for el in yticks])
    plt.minorticks_off()

def plot_detrend(t,f,ftnd,fdt,fses,fdtses):
    legkw = dict(frameon=False,fontsize='x-small')

    fig,axL = plt.subplots(nrows=2,figsize=(12,6),sharex=True,sharey=True)
    plt.sca(axL[0])
    plt.plot(t,f,label='Flux SES = %i' % fses)
    plt.plot(t,ftnd,label='Fit')

    plt.ylabel('Flux')
    plt.legend(**legkw)
    plt.sca(axL[1])
    plt.plot(t,fdt,label='Residuals SES = %i' % fdtses)
    plt.legend(**legkw)
    plt.xlabel('BJD - %i' % bjd0 )
    plt.ylabel('Flux')
    fig.set_tight_layout(True)

class Lightcurve(pd.DataFrame):
    def get_fm(self,col,maskcol='fmask'):
        fm = ma.masked_array(self[col],self[maskcol])
        return fm

    def get_ses(self,col,maskcol='fmask'):
        fm = self.get_fm(col,maskcol=maskcol)
        fm = ma.masked_invalid(fm)
        dfses = ses_stats(fm)
        return dfses

    def plot_detrend(self,columns,ocolumns=[]):
        """
        Parameters
        ----------
        columns : flux, fit, residuals

        """

        t = self['t']
        if min(t) > 2e6:
            t -= bjd0

        f,ftnd,fdt = [self.get_fm(col) for col in columns]
        norm = Normalizer(ma.median(f))

        self['fnorm'] = norm.norm(f)
        self['fdtnorm'] = norm.norm(fdt)
        
        fses = self.get_ses('fnorm').ix[noisename]
        fdtses = self.get_ses('fdtnorm').ix[noisename]
        plot_detrend(t,f,ftnd,fdt,fses,fdtses)
        self.drop(['fnorm','fdtnorm'],axis=1)

class Normalizer:
    def __init__(self,xmed):
        self.xmed = xmed
    def norm(self,x):
        return x / self.xmed - 1
    def unnorm(self,x):
        return (x + 1) * self.xmed


def detrend_roll_seg(lc,plot_diag=False,verbose=False):
    print "Light curve: ntotal=%i nfmask=%i nfdtmask=%i" % \
        (len(lc),lc['fmask'].sum(),lc['fdtmask'].sum())

    tkey = 'roll' # name of dependent variable
    ykey = 'fdt_t' # name of independent variable
    # Update fields
    fdtkey = 'fdt_t_roll' 
    ftndkey = 'ftnd_t_roll' 
    lc.index = lc.cad # Index by cadence

    # Split the light curve up into segments of approx 10 day
    segments = []
    np.random.seed(0)

    start = lc.iloc[0]['cad']
    while start < lc.iloc[-1]['cad']:
        step = 500 + np.random.randint(40) - 20
        stop = start+step
        segments.append( dict(start=start,stop=stop) )
        start = stop

    segments = pd.DataFrame(segments)
    laststart = segments.iloc[-1]['start']
    laststop = segments.iloc[-1]['stop']
    if laststop - laststart < 250:
        segments = segments.iloc[:-1]
    segments.loc[segments.index[-1],'stop'] = laststop

    nseg = len(segments)
    if verbose:
        print "breaking up light curve into following %i segments " % nseg
        print segments.to_string()

    if plot_diag:
        nrow,ncol = square_plots(nseg)
        fig = plt.figure(figsize=(12,8))
        gs0 = gridspec.GridSpec(nrow,ncol)
        fig.set_tight_layout(True)

    for i,row in segments.iterrows():
        if plot_diag:
            gs00 = gridspec.GridSpecFromSubplotSpec(
                5,1, subplot_spec=gs0[i],wspace=0.0, hspace=0.0
                )

            ax1 = plt.subplot(gs00[0,0])
            ax2 = plt.subplot(gs00[1:,0])
            sca(ax1) 
            title('Segment %i' % i)
            axL = [ax1,ax2]
            

        else:
            axL = None 

        # Grab the indecies to detrend against
        idx = lc.ix[row['start']:row['stop']].index
        lcseg = lc.ix[idx]
        lcseg = detrend_roll(lcseg,plot_diag=plot_diag,axL=axL)
        for k in [fdtkey,ftndkey]:
            lc.ix[idx,k] = lcseg[k]

    if plot_diag:
        axes = gcf().get_axes()
        xlim = min(lc[tkey]),max(lc[tkey])
        ylim = min(lc[ykey]),max(lc[ykey])
        residlim = min(lc[fdtkey]),max(lc[fdtkey])
        setp(axes[0::2],xlim=xlim,ylim=residlim)
        setp(axes[1::1],xlim=xlim,ylim=ylim)

        axlabel1 = axes[(nrow - 1)*ncol * 2]
        axlabel2 = axes[(nrow - 1)*ncol * 2 + 1]

        sca(axlabel1)
        ylabel('Residuals \n (fdt_t_roll)')

        sca(axlabel2)
        ylabel('Normalized Flux - 1 \n (fdt_t)')
        xlabel('Roll Angle (arcsec)')
        legend(fontsize='x-small',loc='best')

        for ax in axes:
            ax.grid()
            if ax!=axlabel2 and ax!=axlabel1:
                setp(ax,xticklabels=[],yticklabels=[]) 

    # Since we work with segments some of the fits could be nans just
    # because they weren't included in the segments. Update the mask
    # to remove them from subsequent analysis
    for maskkey in 'fmask fdtmask'.split():
        segmask = lc[[ftndkey,fdtkey]].isnull().sum(axis=1) > 0
        lc[maskkey] = lc[maskkey] | segmask
            
    return lc

def detrend_t_roll_iter(lc,f0,niter=5,plot_diag=True):
    for i in range(niter):
        lc['f'] = lc['fdt_t_roll']+lc['ftnd_t']
        lc = detrend_t(lc,plot_diag=plot_diag)
        lc['fdt_t'] = f0 - lc['ftnd_t']
        lc = detrend_roll_seg(lc,plot_diag=plot_diag)
    return lc

def detrend_t_roll_2D(lc, sigma, length_t, length_roll, sigma_n, 
                      reject_outliers=False,debug=False):
    """
    Detrend against time and roll angle. Hyperparameters are passed
    in as arguments. Option for iterative outlier rejection.

    Parameters
    ----------
    sigma : sets the scale of the GP variance
    length_t : length scale [days] of GP covariance
    length_roll : length scale [arcsec] of GP covariance
    sigma_n : amount of white noise
    reject_outliers : True, reject outliers using iterative sigma clipping

    Returns 
    -------
    """

    # Define constants
    Xkey = 't roll'.split() # name of dependent variable
    Ykey = 'f' # name of independent variable
    fdtkey = 'fdt_t_roll_2D' 
    ftndkey = 'ftnd_t_roll_2D' 
    outlier_threshold = [None,10,5,3]

    if reject_outliers:
        maxiter = len(outlier_threshold) - 1
    else:
        maxiter = 1

    print "sigma, length_t, length_roll, sigma_n"
    print sigma, length_t, length_roll, sigma_n

    iteration = 0
    while iteration < maxiter:
        if iteration==0:
            fdtmask = np.array(lc.fdtmask)
        else:
            # Clip outliers 
            fdt = lc[fdtkey]
            sig = np.median( np.abs( fdt ) ) * 1.5
            newfdtmask = np.abs( fdt / sig ) > outlier_threshold[iteration]
            lc.fdtmask = lc.fdtmask | newfdtmask
            
        print "iteration %i, %i/%i excluded from GP" % \
            (iteration,  lc.fdtmask.sum(), len(lc.fdtmask) )

        # suffix _gp means that it's used for the training
        # no suffix means it's used for the full run
        lc_gp = lc[~lc.fdtmask] 

        # Define the GP
        kernel = sigma**2 * kernels.ExpSquaredKernel(
            [length_t**2,length_roll**2],ndim=2
            ) 

        gp = george.GP(kernel)
        gp.compute(lc_gp[Xkey],sigma_n)

        # Detrend againts time and roll angle
        mu,cov = gp.predict(lc_gp[Ykey],lc[Xkey])
        lc[ftndkey] = mu
        lc[fdtkey] = lc[Ykey] - lc[ftndkey]

        # Also freeze out roll angle dependence
        medroll = np.median( lc['roll'] ) 
        X_t_rollmed = lc[Xkey].copy()
        X_t_rollmed['roll'] = medroll
        mu,cov = gp.predict(lc_gp[Ykey],X_t_rollmed)
        lc['ftnd_t_rollmed'] = mu
        lc['fdt_t_rollmed'] = lc[fdtkey] + mu
        iteration+=1

    if debug:
        lc_gp = lc[~lc.fdtmask] 
        from matplotlib.pylab import *
        ion()
        fig,axL = subplots(nrows=2,sharex=True)
        sca(axL[0])
        errorbar(lc_gp['t'],lc_gp[Ykey],yerr=sigma_n,fmt='o')
        plot(lc['t'],lc[ftndkey])
        sca(axL[1])
        fm = ma.masked_array(lc[fdtkey],lc['fmask'])
        plot(lc['t'],fm)
        fig = figure()
        plot(lc_gp['roll'],lc_gp['f'],'.')
        plot(lc_gp['roll'],lc_gp['ftnd_t_roll_2D'],'.')

        import pdb;pdb.set_trace()

    return lc

def detrend_t_roll_2D_segments(*args,**kwargs):
    """
    Simple wrapper around detrend_t_roll_2D

    Parameters
    ----------
    segment_length : approximate time for the segments [days]
    
    Returns
    -------
    lc : lightcurve after being stiched back together
    """
    lc = args[0]
    segment_length = kwargs['segment_length']
    kwargs.pop('segment_length')
    nchunks = lc['t'].ptp() / segment_length 
    nchunks = int(nchunks)
    nchunks = max(nchunks,1)
    if nchunks==1:
        args_segment = (lc,) + args[1:]
        return detrend_t_roll_2D(*args_segment,**kwargs)

    lc_segments = np.array_split(lc,nchunks)
    lc_out = []
    for i,lc in enumerate(lc_segments):
        args_segment = (lc,) + args[1:]
        lc_out+=[detrend_t_roll_2D(*args_segment,**kwargs)]

    lc_out = pd.concat(lc_out)
    return lc_out

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

def pixel_decorrelation(pixfile,lcfile,transfile,debug=False,tlimits=[-np.inf,np.inf],tex=None):
    """
    Run the pixel decorrelation on pixel file
    """

    # Set parameters 
    r0 = 3 # Identify outliers using 1D decorrelation. r0 is the aperture to use

    # Set parameters used in detrend_t_roll_iter, see that function for docs
    if debug:
        dfiter = dict(
            niter=[1], 
            sigma_clip=[2], 
            plot_diag=[True], 
            )
        s = "debug mode"
        apertures = [3,4]
        niter0 = 2

    else:
        dfiter = dict(
            niter=[5,3,3],
            sigma_clip=[5,3,2],
            plot_diag=[False,False,True]
            )

        s = "full mode"
        apertures = range(2,8)
        niter0 = 5

    dfiter = pd.DataFrame(dfiter)

    # Define skeleton light curve. This pandas DataFrame contains all
    # the columns that don't depend on which aperture is used.

    im,x,y = read_imagestack(pixfile,tlimits=tlimits,tex=tex)
    lc = im.ts # This is a skeleton light curve

    # Load up transformation information
    trans,pnts = read_channel_transform(transfile)
    trans['roll'] = trans['theta'] * 2e5

    # Merge transformation info with the rest of the light curve
    pnts = pd.DataFrame(pnts[0]['cad'.split()])
    trans = pd.concat([trans,pnts],axis=1)
    lc = pd.merge(trans,lc,on='cad')

    # Part 1 
    # Perform an initial run with a single aperture size
    # grab the outliers from this run and use it in subsequent runs
    im.set_apertures(x,y,r0)
    lc = Lightcurve(lc) # upcast to light curve object
    lc['fsap'] = im.get_sap_flux()

    # Standardize the data
    norm = Normalizer(lc['fsap'].median())
    lc['f'] = norm.norm(lc['fsap'])
    f0 = np.array(lc['f'].copy())
    lc['fmask'] = lc['f'].isnull() | lc['thrustermask'] | lc['bgmask']
    lc['fdtmask'] = lc['fmask'].copy()

    # Set the values of the GP hyper parameters
    kepmag = fits.open(pixfile)[0].header['KEPMAG']
    sigma_n = white_noise_estimate(kepmag)
    tchunk = 10 # split lightcurve in to tchunk-day long segments
    nchunks = lc['t'].ptp() / tchunk
    nchunks = int(nchunks)
    sigma = map(lambda x : std(x['f']), np.array_split(lc,nchunks))
    sigma = np.median(sigma)
    length_t = 4
    length_roll = 10
    lc = detrend_t_roll_2D_segments( 
        lc, sigma, length_t, length_roll,sigma_n, debug=False, 
        reject_outliers=True, segment_length=20
        )

    fdtmask = lc['fdtmask'].copy() # Save the mask for future runs

    # Part 2 
    # Now rerun for different apertures
    dfaper = []
    for r in apertures:
        # Set apertures and normalize light curve
        im.set_apertures(x,y,r)
        lc['fsap'] = im.get_sap_flux()
        norm = Normalizer(lc['fsap'].median()) 
        lc['f'] = norm.norm(lc['fsap'])        
        lc['fdtmask'] = fdtmask # Sub in dt mask from previous iteration
        lc = detrend_t_roll_2D( 
            lc, sigma, length_t, length_roll,sigma_n, debug=False, 
            reject_outliers=False
        )

        lc = Lightcurve(lc)
        dfses = lc.get_ses(noisekey) # Cast as Lightcurve object
        noise = dfses.ix[noisename]

        basename = os.path.splitext(lcfile)[0]
        starname = os.path.basename(basename)

        sdisp = """\
starname=%s
r=%i
mad_1_cad_mean=%.1f 
%s=%.1f\
""" % (starname,r,dfses.ix['mad_1_cad_mean'],noisename,dfses.ix[noisename])

        sdisp = " ".join(sdisp.split())
        print sdisp

        unnormkeys = [
            "f",
#            "fdt_t",
#            "ftnd_t",
#            "fdt_t_roll",
#            "ftnd_t_roll",
            "fdt_t_roll_2D",
            "ftnd_t_roll_2D",
            "fdt_t_rollmed",
            "ftnd_t_rollmed",
            ]

        for k in unnormkeys:
            lc[k] = norm.unnorm(lc[k])

        dfaper.append( 
            {'r':r, 'lc': lc.copy(), noisename:dfses.ix[noisename] } 
        )

    dfaper = pd.DataFrame(dfaper)
    dfaper = dfaper.sort(noisename)

    dmin = dfaper.iloc[0]
    lc = dmin['lc']
    dfapersave = dfaper[['r',noisename]]
    to_fits(pixfile,lcfile,lc,dfapersave)

    # Generate diagnostic plots
    if debug:
        ion()
        figure()
        fm = ma.masked_array(lc['fdt_t_rollmed'],lc['fmask'])
        plot(lc['t'],fm)

        import pdb;pdb.set_trace()


    with FigureManager(basename,suffix='_0-median-frame.png'):
        # Add title and stars
        im.radius = dmin['r']
        fr = im.get_medframe()
        fr -= np.median(lc['fbg'])

        epic = fits.open(pixfile)[0].header['KEPLERID']
        fr.plot()
        fr.plot_label(pixfile,epic)
        tit = namemag(pixfile) + " aperture radius=%.1f pixels" % (fr.r)
        title(tit)

    with FigureManager(basename,suffix='_1-noise-vs-aperture-size.png'):
        plot_noise_vs_aperture_size(dfaper,noisename=noisename)

    with FigureManager(basename,suffix="_fdt_t_roll_2D.png"):
        lc2 = Lightcurve(lc)
        lc2.plot_detrend('f ftnd_t_roll_2D fdt_t_roll_2D'.split())
        yl = roblims(lc['ftnd_t_roll_2D'],5,2)
        tit = namemag(pixfile) + " Pixel Decorrelation" 
        ylim(*yl)
        
    with FigureManager(basename,suffix="_fdt_t_roll_2D_zoom.png"):
        lc2 = Lightcurve(lc)
        lc2.plot_detrend('f ftnd_t_roll_2D fdt_t_roll_2D'.split())
        yl = roblims(lc['fdt_t_roll_2D'],5,2)
        tit = namemag(pixfile) + " Pixel Decorrelation" 
        ylim(*yl)

    return lc,dfapersave


def to_fits(pixfile,fitsfile,lc,dfaper):
    """
    Package up the light curve, SES information into a fits file
    """

    def fits_column(c,df):
        array = np.array(df[ c[0] ])
        column = fits.Column(array=array, format=c[1], name=c[0], unit=c[3])
        return column

    def BinTableHDU(df,coldefs):
        columns = [fits_column(col,df) for col in coldefs]
        hdu = fits.BinTableHDU.from_columns(columns)
        for c in coldefs:
            hdu.header[c[0]] = c[2]
        return hdu

    # Copy over primary HDU
    hduL_pixel = fits.open(pixfile)
    hdu0 = hduL_pixel[0]

    # Light curve table
    coldefs = [
        ["thrustermask","L","Thruster fire","bool"],
        ["roll","D","Roll angle","arcsec"],
        ["cad","J","Unique cadence number","int"],
        ["t","D","Time","BJD - %i" % bjd0],
        ["fbg","D","Background flux","electrons per second per pixel"],
        ["fsap","D","Simple aperture photometry","electrons per second"],
        ["fmask","L","Global mask. Observation ignored","bool"],
        ["fdtmask","L",
         "Detrending mask. Observation ignored in detrending model","bool"],
#        ["ftnd_t","D","Gaussian process model: GP(fsap; t)",
#         "electrons per second"],
#        ["fdt_t","D","Residuals (fsap - ftnd_t)","electrons per second"],
#        ["ftnd_t_roll","D","Gaussian process model: GP(fdt_t; roll)",     
#         "electrons per second"],
#        ["fdt_t_roll","D","Residuals (fdt_t - ftnd_t_roll)",
#         "electrons per second"],
        ["ftnd_t_roll_2D","D","Gaussian process model: GP(fsap; t, roll)",     
         "electrons per second"],
        ["fdt_t_roll_2D","D","Residuals (fsap - ftnd_t_roll_2D)",
         "electrons per second"],
        ["ftnd_t_rollmed","D","GP from ftnd_roll_2D using median roll",
         "electrons per second"],
        ["fdt_t_rollmed","D","ftnd_t_rollmed + fdt_t_roll_2D",
         "electrons per second"],
    ]
    
    hdu1 = BinTableHDU(lc,coldefs)

    # dfaper 
    coldefs = [
        ["r","D","Radius of aperture","pixels"],
        [noisename,"D","Noise metric used to optimize aperture size","pixels"],
    ]

    hdu2 = BinTableHDU(dfaper,coldefs)        
    hduL = fits.HDUList([hdu0,hdu1,hdu2])
    hduL.writeto(fitsfile,clobber=True)
    return hduL
