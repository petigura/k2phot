"""
Module for plotting phot object
"""

from .config import *
from ..config import bjd0, timelabel
from ..lightcurve import Lightcurve, Normalizer

def medframe(phot):
    """
    Plot median frame with aperture drawn 
    """
    medframe = phot.medframe
    medframe -= np.median(phot.lc['fbg'])
    imshow(phot.medframe)
    verts = phot.ap_verts 
    plt.plot(verts.x,verts.y,color="LimeGreen",lw=2)
    plt.xlabel('Column (pixels)')
    plt.ylabel('Row (pixels)')
    tit = phot.name_mag()
    plt.title(tit)

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

