#!/usr/bin/env python

from argparse import ArgumentParser
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
import channel_centroids
from pixel_io import bjd0
from ses import ses_stats
import flatfield
from astropy.io import fits
import contextlib

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


def detrend_t(lc,plot_diag=False):
    """
    Detrend against time
    """

    tkey = 't' # name of dependent variable
    ykey = 'f' # name of independent variable
    L_t = 4  # caracteristic length-scale for time-dependent kernel
    A_t = 0.2 # Amplitude term in time-kernel
    fdtkey = 'fdt_t' 
    ftndkey = 'ftnd_t' 

    L_t = 4  # caracteristic length-scale for time-dependent kernel
    A_t = 0.2 # Amplitude term in time-kernel
    bin_width_t = 0.5 # size [days of light curve bins]

    # Light curve used in GP
    lc_gp = lc[~lc.fdtmask]

    for key in [tkey,ykey]:
        assert lc_gp[key].isnull().sum()==0,"GP array can contain no nans"
 
    # Compute bins 
    bins_t = np.arange(
        lc.iloc[0][tkey], 
        lc.iloc[-1][tkey]+bin_width_t,
        bin_width_t
        )
    nbins = len(bins_t) - 1

    g = lc.groupby(pd.cut(lc_gp.t,bins_t))
    lc_bin_t = g[[tkey,ykey]].median()
    lc_bin_t['yerr'] = g[ykey].std() / np.sqrt(g[ykey].size())
    lc_bin_t = lc_bin_t.dropna()

    t = lc_bin_t[tkey] # Independent variable
    y = lc_bin_t[ykey] # Error on dependent variable
    yerr = lc_bin_t['yerr']

    # Construct GP
    theta_t = L_t**2
    k_t = A_t * kernels.ExpSquaredKernel(theta_t)
    gp = george.GP(k_t,solver=george.HODLRSolver)
    gp.compute(t,yerr)

    mu, cov = gp.predict(y,lc[tkey])

    # Update fields
    lc[ftndkey] = mu
    lc[fdtkey] = lc[ykey] - lc[ftndkey]

    if plot_diag:
        lcplot = lc.dropna(subset=[tkey,ykey])
        lcplotout = lcplot[lcplot.fdtmask]

        fig = plt.figure(figsize=(20,8))
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
        ax2 = plt.subplot2grid((3,3), (1,0), colspan=3,rowspan=2)

        sca(ax2)
        plot(lcplot[tkey],lcplot[ykey],label='Unbinned Flux')
        errorbar(t,y,yerr=yerr,fmt='.',label='Binned Flux')

        gpfmt = dict(color='c',lw=2,zorder=10,ms=8,label=ftndkey)        
        plot(lcplot[tkey],lcplot[ftndkey],**gpfmt)
        plot(lcplotout[tkey],lcplotout[ykey],**outfmt)

        sca(ax1)
        plot(
            lcplot[tkey],lcplot[fdtkey],label='Detrended flux (after fit to t)'
            )

        plot(
            lcplotout[tkey],lcplotout[fdtkey],**outfmt)

        xlabel('Time [days]')
        ylabel('Flux')
        gcf().set_tight_layout(True)

    return lc

def detrend_roll(lc,plot_diag=False,axL=None):
    """
    Detrend against roll angle

    Parameters
    ----------
    lc : Pandas DataFrame must contain the following fields
         - roll
         - fdt_t - flux detrended against time
         - fmask - if true, flux is ignored 
         - fdtmask - if true, flux is ignored from construction of model

    """
    tkey = 'roll' # name of dependent variable
    ykey = 'fdt_t' # name of independent variable
    L_roll = 10  # caracteristic length-scale for roll-dependent kernel [arcsec]
    A_roll = 0.2 # Amplitude term in time-kernel
    fdtkey = 'fdt_t_roll' 
    ftndkey = 'ftnd_t_roll' 

    # Light curve used in GP
    lc_gp = lc[~lc.fdtmask]
    for key in [tkey,ykey]:
        assert lc_gp[key].isnull().sum()==0,"GP array can contain no nans"

    # Construct GP object
    theta_roll = L_roll**2
    k_roll = A_roll * kernels.ExpSquaredKernel(theta_roll)
    gp = george.GP(k_roll,solver=george.HODLRSolver)
    t = lc_gp[tkey] # Independent variable
    y = lc_gp[ykey] # Error on dependent variable
    yerr = 1e-3
    gp.compute(t,yerr)

    mu, cov = gp.predict(lc_gp[ykey],lc[tkey])        
    lc[ftndkey] = mu
    lc[fdtkey] = lc[ykey] - lc[ftndkey]

    if plot_diag:
        sca(axL[1])
        lcplot = lc.dropna(subset=[tkey,ykey])
        lcplot = lcplot.sort(tkey)
        lcplotout = lcplot[lcplot.fmask]
        infmt = dict(marker='.',lw=0,zorder=10,label=fdtkey)
        outfmt = dict(marker='o',mew=0,mfc='r',alpha=0.5,lw=0,zorder=0,ms=6,
                      label='Excluded from Model')
        gpfmt = dict(color='c',lw=2,zorder=10,ms=8,label=ftndkey)        

        plot(lcplot[tkey],lcplot[ykey],**infmt)
        plot(lcplotout[tkey],lcplotout[ykey],**outfmt)
        plot(lcplot[tkey],lcplot[ftndkey],**gpfmt)
        sca(axL[0])
        plot( lcplot[tkey],lcplot[ykey] - lcplot[ftndkey],**infmt )
        plot(lcplotout[tkey],lcplotout[ykey] - lcplotout[ftndkey],**outfmt)

    return lc

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

def detrend_t_roll_2D(lc):
    L_t = 4**2 # Correlation length-scale for time component
    L_roll = 10**2 # Correlation length-scale for roll component

    tkey = 't roll'.split() # name of dependent variable
    ykey = 'f' # name of independent variable
    fdtkey = 'fdt_t_roll_2D' 
    ftndkey = 'ftnd_t_roll_2D' 
    lc_gp = lc[~lc.fdtmask]
    b = np.array(~lc.fdtmask)
    x = np.array(lc_gp[tkey])
    y = lc[ykey][b]
    yerr = 1e-4
    k2d = 1.*kernels.ExpSquaredKernel([L_t,L_roll],ndim=2) 
    gp = george.GP(k2d)
    gp.compute(x,yerr)

    X_t_roll = lc[tkey]

    mu,cov = gp.predict(y,X_t_roll)
    lc[ftndkey] = mu
    lc[fdtkey] = lc[ykey] - lc[ftndkey]

    # Also freeze out roll angle dependence
    medroll = np.median( lc['roll'] ) 
    X_t_rollmed = lc[tkey].copy()
    X_t_rollmed['roll'] = medroll
    yerr = 1e-4
    mu,cov = gp.predict(y,X_t_rollmed)
    lc['ftnd_t_rollmed'] = mu
    lc['fdt_t_rollmed'] = lc[fdtkey] + mu
    return lc

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

    basename = os.path.splitext(lcfile)[0]
    starname = os.path.basename(basename)

    im,x,y = read_imagestack(pixfile,tlimits=tlimits,tex=tex)
    lc = im.ts # This is a skeleton light curve

    # Load up transformation information
    trans,pnts = channel_centroids.read_channel_centroids(transfile)
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

    lc = detrend_t(lc,plot_diag=False) 
    lc = detrend_roll_seg(lc,plot_diag=False)

    # Perform first round of iterative detrending 
    # Figure out which observations were outliers and repeat
    print "aperture size=%i" % r0 
    print "running detrend_t_roll_iter with following parameters:"
    print dfiter.to_string()
    lc = detrend_t_roll_iter(lc,f0,niter=niter0)
    for i,row in dfiter.iterrows():
        sig = np.median(np.abs(lc['fdt_t_roll']))*1.5
        boutlier = np.array(np.abs(lc['fdt_t_roll']) > row['sigma_clip']*sig)
        lc['fdtmask'] = lc['fdtmask'] | boutlier
        lc = detrend_t_roll_iter(
            lc,f0,niter=row['niter'],plot_diag=row['plot_diag']
            )

    figures = [m.canvas.figure for m in Gcf.get_all_fig_managers()]
    for i,fig in enumerate(figures):
        figpath = basename+"_fdt_t_roll_r=%i_%i.png" % (r0,i)
        fig.savefig(figpath)
    plt.close('all')

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
        f0 = np.array(lc['f'].copy())
        lc['fdtmask'] = fdtmask # Sub in dt mask from previous iteration
        lc = detrend_t_roll_2D(lc)
        dfses = lc.get_ses(noisekey) # Cast as Lightcurve object
        noise = dfses.ix[noisename]

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
            "fdt_t",
            "ftnd_t",
            "fdt_t_roll",
            "ftnd_t_roll",
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
        ["cad","J","Unique candence number","int"],
        ["t","D","Time","BJD - %i" % bjd0],
        ["fbg","D","Background flux","electrons per second per pixel"],
        ["fsap","D","Simple aperture photometry","electrons per second"],
        ["fmask","L","Global mask. Observation ignored","bool"],
        ["fdtmask","L",
         "Detrending mask. Observation ignored in detrending model","bool"],
        ["ftnd_t","D","Gaussian process model: GP(fsap; t)",
         "electrons per second"],
        ["fdt_t","D","Residuals (fsap - ftnd_t)","electrons per second"],
        ["ftnd_t_roll","D","Gaussian process model: GP(fdt_t; roll)",     
         "electrons per second"],
        ["fdt_t_roll","D","Residuals (fdt_t - ftnd_t_roll)",
         "electrons per second"],
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
        
if __name__ == "__main__":
    p = ArgumentParser(description='Pixel Decorrelation')
    p.add_argument('pixfile',type=str)
    p.add_argument('lcfile',type=str)
    p.add_argument('transfile',type=str)
    p.add_argument('--debug',action='store_true',help='run in debug mode?')
    p.add_argument(
        '--tmin', type=float, default=-np.inf,help='Minimum valid time index'
    )
    p.add_argument(
        '--tmax', type=float, default=np.inf,help='Max time'
    )

    p.add_argument(
        '--tex', type=str, default=None,help='Exclude time range'
    )

    
    args  = p.parse_args()

    tex = args.tex
    if type(args.tex)!=type(None):
        tex = eval("np.array(%s)" % tex)

    tlimits = [args.tmin,args.tmax]

    pixel_decorrelation(
        args.pixfile,args.lcfile,args.transfile,debug=args.debug,
        tlimits=tlimits, tex = tex
        )
