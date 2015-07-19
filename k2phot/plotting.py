"""
Module for generating diagnostic plots
"""
import matplotlib.pylab as plt
import numpy as np
import lightcurve
import config
import contextlib
from numpy import ma
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import MaxNLocator
import matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

pntskw = dict(marker='.',linestyle='-',alpha=0.8,mew=0,ms=5,mfc='RoyalBlue',color='RoyalBlue')
legkw = dict(frameon=False,fontsize='x-small',loc='lower left')

def medframe(pixdcr):
    """
    Plot median frame with aperture drawn 
    """
    pixdcr.set_apertures(pixdcr.dmin['r'])
    lc = pixdcr.lc

    fr = pixdcr.im.get_medframe()
    fr -= np.median(lc['fbg'])
    fr.plot()
    fr.plot_label(pixdcr.pixfn,int(pixdcr.starname))
    tit = pixdcr.name_mag() + " aperture radius=%.1f pixels" % (fr.r)
    plt.title(tit)

def noise_vs_aperture_size(pixdcr,noisename='fdt_mad_6_cad_mtd'):
    dfaper = pixdcr.dfaper
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
    tit = pixdcr.name_mag()
    plt.title(tit)

def detrend_t_roll_2D(pixdcr,**kwargs):
    with detrend_titles(pixdcr):
        lightcurve_detrend_t_roll_2D(pixdcr.lc,**kwargs)
        axL= plt.gcf().get_axes()
        plt.sca(axL[1])
        plt.title( pixdcr.raw_corrected() )

def detrend_t_rollmed(pixdcr,**kwargs):
    with detrend_titles(pixdcr):
        lightcurve_detrend_t_rollmed(pixdcr.lc,**kwargs)

def background(pixdcr):
    lc = pixdcr.lc
    t = lc['t']
    if min(t) > 2e6:
        t -= config.bjd0

    fbg = ma.masked_array(lc['fbg'],lc['bgmask'])
    plt.plot(t,fbg.data,color='RoyalBlue',label='Background Flux')
    plt.plot(t,fbg,color='Tomato',label='Outliers Removed')
    plt.legend(**legkw)
    plt.ylabel('Background Flux (electrons / s / pixel)')
    plt.xlabel(config.timelabel)
    plt.title(pixdcr.name_mag())


 #########

@contextlib.contextmanager
def detrend_titles(pixdcr):
    """
    A small context manager that pops figures and resolves the output
    filepath
    """

    yield # Now run the code block
    axL = plt.gcf().get_axes()
    plt.sca(axL[0])
    plt.title(pixdcr.name_mag())

def lightcurve_detrend_t_roll_2D(lc,zoom=False):
    keys = 'fsap ftnd_t_roll_2D fdt_t_roll_2D'.split()
    flines, ftndlines, fdtlines = lightcurve_detrend(lc,keys,zoom=zoom)
    
    # Label Axes
    axL = plt.gcf().get_axes()
    plt.xlabel(config.timelabel)
    flines.set_label('Raw SAP Flux')
    ftndlines.set_label('GP Model, Time and Roll')
    fdtlines.set_label('Detrened Flux')
    
    plt.sca(axL[0])
    lightcurve_masks(lc)
    plt.legend(**legkw)

    plt.sca(axL[1])
    plt.legend(**legkw)
    lightcurve_masks(lc)

def lightcurve_detrend_t_rollmed(lc,zoom=False):
    keys = 'fsap ftnd_t_rollmed fdt_t_rollmed'.split()
    flines, ftndlines, fdtlines = lightcurve_detrend(lc,keys)
    
    # Label Axes
    axL = plt.gcf().get_axes()
    plt.xlabel(config.timelabel)
    flines.set_label('Raw SAP Flux')

    ftndlines.set_linewidth(0)
    fdtlines.set_label('Detrened Flux')
    for ax in axL:
        plt.sca(ax)
        plt.legend(**legkw)

def find_replace_label(label0,label1):
    def myfunc(x):
        return hasattr(x, 'set_label')

    fig = plt.gcf()
    for o in fig.findobj(myfunc):
        print o.get_label()
        if o.get_label()==label0:
            o.set_label(label1)

def detrend(t,f,ftnd,fdt):
    """
    Normalize and plot
    """
    fitkw = dict(alpha=0.8,color='Tomato')
    #fig,axL = plt.subplots(nrows=2,figsize=(12,6),sharex=True,sharey=True)
    fig = plt.figure(figsize=(12,6))
    
    ax1 = host_subplot(211, axes_class=AA.Axes)
    ax2 = host_subplot(212, axes_class=AA.Axes,sharex=ax1)
    axL = [ax1,ax2]

    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    for ax in axL:
        ax.yaxis.set_major_formatter(y_formatter)
        ax.grid()

    plt.sca(axL[0])

    flines, = plt.plot(t,f,label='Raw',**pntskw)
    ftndlines, = plt.plot(t,ftnd,lw=2,label='Fit',**fitkw)
    plt.ylabel('Flux')

    plt.sca(axL[1])
    fdtlines, = plt.plot(t,fdt,label='Resid',**pntskw)

    fig.set_tight_layout(True)

        

    return flines, ftndlines, fdtlines 

def lightcurve_detrend(lc,keys,zoom=False):
    lc = lightcurve.Lightcurve(lc)
    t = lc['t']
    if min(t) > 2e6:
        t -= config.bjd0

    f,ftnd,fdt = [lc.get_col(col,norm=True,maskcol='fmask') for col in keys]

    lines = detrend(t,f,ftnd,fdt)
    yl = roblims(ftnd.compressed(),5,2)
    if zoom:
        yl = roblims(fdt.compressed(),5,2)        
    plt.ylim(*yl)

    medkeys = [keys[0],keys[2]]
    axL = plt.gcf().get_axes()
    for ax,medkey in zip(axL,medkeys):
        ax2 = ax.twin() 
        yt = ax.get_yticks()
        ax2.set_yticks(yt)

        med = ma.median(lc.get_col(medkey,maskcol='fmask'))
        ax2.set_yticklabels(["%.2e" % s for s in  yt*med])
        ax2.axis["top"].major_ticklabels.set_visible(False)
        plt.setp(ax,ylabel='Normalized Flux')
        plt.setp(ax2,ylabel='Flux (electrons/s)')

    return lines


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

def lightcurve_masks(lc):
    colors = 'Cyan Pink LimeGreen k'.split()
    maskkeys = 'thrustermask bgmask fdtmask fmask'.split()
    ax = plt.gca()
    for i,k in enumerate(maskkeys):
        lcmask = lc[lc[k]]
        y = np.zeros(len(lcmask)) + 0.95 - 0.03 * i
        x = lcmask['t']
        trans = btf(ax.transData,ax.transAxes)
        color = colors[i]
        plt.plot(x,y,'|',ms=4,mew=2,transform=trans,color=color,label=k)

def lightcurve_segments(lc0):
    nseg = 8
    nrows = 4
    fig = plt.figure(figsize=(18,10))
    gs = GridSpec(nrows,nseg)
    lc_segments = np.array_split(lc0,nseg)
    plt.rc('lines',markersize=6,markeredgewidth=0)
    def plot(*args,**kwargs):
        plt.plot(*args,alpha=0.5,**kwargs)

    for i in range(nseg):
        if i==0:
            ax0L = [fig.add_subplot(gs[j,0]) for j in range(nrows)]
            axiL = ax0L
        else:
            axiL = [
                fig.add_subplot(gs[0,i],sharey=ax0L[0]),
                fig.add_subplot(gs[1,i],sharey=ax0L[1]),
                fig.add_subplot(gs[2,i],sharey=ax0L[2]),
                fig.add_subplot(gs[3,i],sharex=ax0L[3],sharey=ax0L[3]),
            ]

            for ax in axiL:
                plt.setp(
                    ax.yaxis,
                    visible=False,
                    major_locator=MaxNLocator(3)
                )
                plt.setp(
                    ax.xaxis,
                    visible=False,
                    major_locator=MaxNLocator(3)
                )

        lc = lc_segments[i]
        lc = lightcurve.Lightcurve(lc)
        fm = lc.get_fm('fsap')
        fdt = lc.get_fm('ftnd_t_roll_2D')
        
        plt.sca(axiL[0])
        plot(lc['t'],lc['roll'])

        plt.sca(axiL[1])
        plot(lc['roll'],fm,'.')
        plot(lc['roll'],fdt,'.')

        plt.sca(axiL[2])
        plot(lc['t'],lc['roll'])

        plt.sca(axiL[3])
        
        xpr = lc.get_fm('xpr')
        ypr = lc.get_fm('ypr')
        plot(xpr,ypr,'.')

    fig.subplots_adjust(
        left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0.001, wspace=0.001
    )


def diag(dv,tpar=False):
    """
    Print a 1-page diagnostic plot of a given h5.
    
    Right now, we recompute the single transit statistics on the
    fly. By default, we show the highest SNR event. We can fold on
    arbitrary ephmeris by setting the tpar keyword.

    Parameters
    ----------
    h5   : h5 file after going through terra.dv
    tpar : Dictionary with alternate ephemeris specified by:
           Pcad - Period [cadences] (float)
           t0   - transit epoch [days] (float)
           twd  - width of transit [cadences]
           mean - depth of transit [float]
           noise 
           s2n
    """

    # Top row
    axPeriodogram  = fig.add_subplot(gs[0,0:8])
    axAutoCorr = fig.add_subplot(gs[0,8])

    # Second row
    axPF       = fig.add_subplot(gs[1,0:2])
    axPFzoom   = fig.add_subplot(gs[1,2:4],sharex=axPF,)
    axPF180    = fig.add_subplot(gs[1,4:6],sharex=axPF)
    axPFSec    = fig.add_subplot(gs[1,6:8],sharex=axPF)
    axSingSES  = fig.add_subplot(gs[1,-2])

    # Last row
    axStack        = fig.add_subplot(gs[2:8 ,0:8])
    axStackZoom    = fig.add_subplot(gs[2:8 ,8:])

    # Top row
    sca(axPeriodogram)
    periodogram(dv)

    sca(axAutoCorr)
    autocorr(dv)
    AddAnchored("ACF",prop=tprop,frameon=True,loc=2)    

