"""
Module for plotting phot object
"""

from .config import *
from ..config import bjd0, timelabel
from ..lightcurve import Lightcurve, Normalizer
import finderchart

def medframe(phot):
    """Plot median frame with aperture drawn"""
    medframe = phot.medframe
    medframe -= np.median(phot.lc['fbg'])
    imshow(phot.medframe)
    shape = phot.medframe.shape

    verts = phot.ap_verts 
    plt.plot(verts.x,verts.y,color="LimeGreen",lw=2)
    plt.xlabel('Column (pixels)')
    plt.ylabel('Row (pixels)')
    plt.xlim(-0.5,shape[1]-0.5)
    plt.ylim(-0.5,shape[0]-0.5)
    tit = phot.name_mag()
    plt.title(tit)

def aperture(phot):
    """Plot K2 aperture on Kepler pixels and DSS"""
    plt.rc('font',size=8)
    fig = plt.figure(figsize=(11.5,3.5))
    fig.subplots_adjust(
        wspace=0.4, top=0.83, bottom=0.12, left=0.06, right=0.95
         )
    ax = fig.add_subplot(131)
    plt.sca(ax)
    medframe(phot)
    finderchart.dss(phot, survey='poss1_red', subplot_args=(132,))
    finderchart.dss(phot, survey='poss2ukstu_red', subplot_args=(133,))

def flux_position_scatter(lc0,thruster=False):
    """Plot K2 aperture on Kepler pixels and DSS"""
    lc = lc0.copy()
    plt.rc('font',size=8)
    fig,axL = plt.subplots(nrows=1,ncols=2,figsize=(15,6),sharex=True,sharey=True)
    lc = Lightcurve(lc)
    cols = ['fsap','fdt_t_roll_2D']
    vals =  [lc.get_col(col,norm=True,maskcol='fmask') for col in cols]
    for i in range(2):
        col = cols[i]
        fm = vals[i]
        plt.sca(axL[i])
        lim = roblims(fm, 5, 2)

        lcmask = lc[lc.thrustermask]
        if thruster:
            plt.plot(lcmask.xpr,lcmask.ypr,'x',color='w',mew=3,ms=8)
            plt.plot(lcmask.xpr,lcmask.ypr,'x',color='m',mew=1)

        plt.scatter(
            lc.xpr, lc.ypr,c=fm, vmin=lim[0], vmax=lim[1],cmap=plt.cm.coolwarm
        )
        c = plt.colorbar()
        plt.title(col)
    fig.subplots_adjust(left=0.07,right=0.99, hspace=0.08)
    
    xl = roblims(lc.xpr, 10, 1.5)
    yl = roblims(lc.ypr, 10, 1.5)
    plt.xlim(*xl)
    plt.ylim(*yl)

def background(phot):
    lc = phot.lc
    t = lc['t']
    if min(t) > 2e6:
        t -= bjd0

    fbg = ma.masked_array(lc['fbg'],lc['bgmask'])
    plt.plot(t,fbg.data,color='RoyalBlue',label='Background Flux')
    plt.plot(t,fbg,color='Tomato',label='Outliers Removed')
    plt.legend(**legkw)
    plt.ylabel('Background Flux (electrons / s / pixel)')
    plt.xlabel(timelabel)
    tit = phot.name_mag()
    plt.title(tit)

def detrend_t_roll_2D(phot,**kwargs):
    with detrend_titles(phot):
        lightcurve_detrend_t_roll_2D(phot.lc,**kwargs)
        axL= plt.gcf().get_axes()

def detrend_t_rollmed(phot,**kwargs):
    with detrend_titles(phot):
        lightcurve_detrend_t_rollmed(phot.lc,**kwargs)

@contextlib.contextmanager
def detrend_titles(phot):
    """
    A small context manager that pops figures and resolves the output
    filepath
    """

    yield # Now run the code block
    axL = plt.gcf().get_axes()
    plt.sca(axL[0])
    plt.title(phot.name_mag())

def lightcurve_detrend_t_roll_2D(lc,zoom=False):
    keys = 'fsap ftnd_t_roll_2D fdt_t_roll_2D'.split()
    flines, ftndlines, fdtlines = lightcurve_detrend(lc,keys,zoom=zoom)
    
    # Label Axes
    axL = plt.gcf().get_axes()
    plt.xlabel(timelabel)
    flines.set_label('Raw SAP Flux')
    ftndlines.set_label('GP Model, Time and Roll')
    fdtlines.set_label('Detrended Flux')
    
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
    plt.xlabel(timelabel)
    flines.set_label('Raw SAP Flux')
    ftndlines.set_linewidth(0)
    fdtlines.set_label('Detrened Flux')
    for ax in axL:
        plt.sca(ax)
        plt.legend(**legkw)

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
        ax.grid(zorder=0)

    plt.sca(axL[0])
    flines, = plt.plot(t,f,label='Raw',**pntskw)
    ftndlines, = plt.plot(t,ftnd,lw=2,label='Fit',**fitkw)
    plt.ylabel('Flux')
    plt.sca(axL[1])
    fdtlines, = plt.plot(t,fdt,label='Resid',**pntskw)
    fig.set_tight_layout(True)
    return flines, ftndlines, fdtlines 

def lightcurve_detrend(lc,keys,zoom=False):
    lc = Lightcurve(lc)
    t = lc['t']
    if min(t) > 2e6:
        t -= bjd0

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
        yt = ax.get_yticks() # Get values of the left y-axis
        ax2.set_yticks(yt) # Copy into the right y-axis

        # Convert into electrons per-second
        med = ma.median(lc.get_col(medkey,maskcol='fmask'))
        yt2 = yt * med

        # Determine the exponent
        yt2max = yt2[-1] 
        exponent = int(np.log10(yt2max))
        yt2 /= (10**exponent)
        yt2 = ["%.2f" % s for s in  yt2]
        
        ax2.set_yticklabels(yt2) 
        ax2.axis["top"].major_ticklabels.set_visible(False)
        plt.setp(ax,ylabel='Normalized Flux')
        plt.setp(ax2,ylabel='Flux (10^%i electrons/s)' % exponent )

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
    plo,pmed,phi = np.percentile(x,[p,50,100-p])
    lim = plo - (pmed-plo) * fac, phi + (phi-pmed) * fac
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
        lc = Lightcurve(lc)

        fm = lc.get_fm('fsap')
        ftnd = lc.get_fm('ftnd_t_roll_2D')
        
        plt.sca(axiL[0])
        plot(lc['t'],lc['roll'],label='x: t, y: roll')

        plt.sca(axiL[1])
        plot(lc['roll'],fm,'.',label='x: roll, y: flux')
        plot(lc['roll'],ftnd,'.',label='x: roll, y: flux model')

        plt.sca(axiL[2])
        plot(lc['t'],lc['roll'],label='x: t, y: roll')

        plt.sca(axiL[3])
        
        xpr = lc.get_fm('xpr')
        ypr = lc.get_fm('ypr')
        plot(xpr,ypr,'.',label='x: xpr, y: yrp')

    for i in range(nrows):
        plt.sca(ax0L[i])
        plt.legend(fontsize='x-small')

    fig.subplots_adjust(
        left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0.001, wspace=0.001
    )

