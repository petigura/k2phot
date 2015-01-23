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

from imagestack import ImageStack
from pixel_decorrelation import get_wcs
import channel_centroids
from pixel_io import bjd0
from ses import ses_stats
import flatfield


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

def split_lc(lc):
    seg = segments.copy()
    camp_seg = seg#[seg.k2_camp==k2_camp]
    lc_segments = []
    for i,row in camp_seg.iterrows():
        lc_seg = lc.query('%(start)i <= cad <= %(stop)i' % row)
        lc_segments +=[lc_seg]
    return lc_segments

def plot_position_PCs(lc):
    test = plt.scatter(
        lc.pos_pc0,lc.pos_pc1,c=lc.f,linewidths=0,alpha=0.8,s=20)

    plt.xlabel('pos_pc0')
    plt.ylabel('pos_pc1')
    plt.colorbar()

def plot_gp_pos(lc,gp_pos):
    desc = lc.describe()
    res = 50
    lim1 = desc.ix['min','pos_pc0'], desc.ix['max','pos_pc0']
    lim2 = desc.ix['min','pos_pc1'], desc.ix['max','pos_pc1']
    x1, x2 = np.meshgrid(np.linspace(lim1[0],lim1[1], res),
                         np.linspace(lim2[0],lim2[1], res))
    xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
    y_pred, MSE = gp_pos.predict(xx, eval_MSE=True)
    y_pred = y_pred.reshape((res,res))
    extent = (lim1[0],lim1[1],lim2[0],lim2[1])
    plt.imshow(
        np.flipud(y_pred), alpha=0.8, extent=extent,aspect='auto')

class Lightcurve(pd.DataFrame):
    def set_position_PCs(self):
        X = lc_to_X(self,'dx dy'.split())
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)
        for i in [0,1]:
            self['pos_pc%i' %i ] = X_r[:,i]

    def get_fm(self,col,maskcol='fmask'):
        return ma.masked_array(col,maskcol)

    def get_X(self,col,maskcol=None):
        if maskcol is None:
            return lc_to_X(self,col)
        else:
            lc2 = self[~self[maskcol]]
            return lc_to_X(lc2,col)


def plot_detrend(lc,columns):
    """
    Parameters
    ----------
    columns : flux, fit, residuals

    """
    legkw = dict(frameon=False,fontsize='x-small')
    fkey,ftndkey,fdtkey = columns
    t = lc['t']
    if min(t) > 2e6:
        t -= bjd0

    get_masked = lambda key : ma.masked_array(lc[key],lc['fmask'])
    res = map(get_masked,columns)
    f,ftnd,fdt = res

    fses = f / ma.median(f) - 1 
    fdtses = fdt / ma.median(fdt) - 1 
    fses,fdtses = map(get_ses,[fses,fdtses])

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


class Normalizer:
    def __init__(self,xmed):
        self.xmed = xmed
    def norm(self,x):
        return x / self.xmed - 1
    def unnorm(self,x):
        return (x + 1) * self.xmed


def get_ses(f):
    ses = ses_stats(f)
    ses.index = ses.name
    ses = ses.ix['mad_6-cad-mtd','value']
    return ses

def read_weight_file(h5filename,debug=False):
    with h5py.File(h5filename,'r') as h5:
        groupnames = [i[0] for i in h5.items()]

    dfweights = pd.DataFrame(groupnames,columns=['name'])
    if debug:
        dfweights = dfweights[dfweights.name.str.contains('r=3|r=4') &
                              dfweights.name.str.contains('mov=0_weight=0')]

    dfweights['im'] = ''
    for index,sweights in dfweights.iterrows():
        sweights['im'] = flatfield.read_hdf(h5filename,sweights['name'])

    return dfweights

def im_to_lc(im):
    frames = im.get_frames()
    moments = [frame.get_moments() for frame in frames]
    moments = pd.DataFrame(
        moments,columns='m10 m01 mupr20 mupr02 mupr11'.split()
        )

    lc = pd.concat([im.ts,moments],axis=1)

    # Add the thurster fire mask
    k2_camp = 'C%i' % fits.open(im.fn)[0].header['CAMPAIGN']
    lc = flatfield.add_cadmask(lc,k2_camp)

    lc['f'] = im.get_sap_flux()
    lc['dx'] = lc['m01']
    lc['dy'] = lc['m10']
    return lc

fmad = lambda x : np.median(abs(x))
fnugget = lambda x : (1.6 * fmad(x - np.median(x) ))**2

def fit_gp_sigma_clip(gp,X,y,verbose=False):
    binlier = np.ones(X.shape[0]).astype(bool)
    binlier_last = binlier.copy()
    colors='rcyrcy'
    i = 0 
    done = False
    while not done:
        if verbose:
            print binlier.sum(),i
        gp.fit(X[binlier,:],y[binlier])
        y_pred = gp.predict(X)
        mad = fmad(y - y_pred)

        binlier_last = binlier.copy()
        binlier = abs(y - y_pred) < 4*mad

        if np.all(binlier_last==binlier) or (i >= 3):
            done = True

        i+=1

    return gp

def decorrelate_position_and_time(lc,verbose=True):
    lc = Lightcurve(lc)

    lc['f'] /= np.median(lc['f'])
    lc['f'] -= 1

    lc.set_position_PCs()

    X_t = lc_to_X(lc,'t')
    X_pos = lc_to_X(lc,['pos_pc0','pos_pc1'])
    y = np.array(lc['f'])

    gpkw = dict(
        regr='linear',corr='squared_exponential',nugget=fnugget(y))

    gp_t = GaussianProcess(theta0=3,**gpkw)

    thetaU = [1,1]
    thetaL = [1e-3,1e-3]
    theta0 = [1e-2,1e-2]
    gp_pos = GaussianProcess(theta0=theta0,thetaU=thetaU,thetaL=thetaL,**gpkw)

    fdt_pos_last = y.copy()
    fdt_pos = y.copy()

    i = 0

    done = False
    while not done:
        ndiff = (fdt_pos - fdt_pos_last) / fdt_pos_last
        chi2 = np.sum(ndiff**2) / len(fdt_pos)

        gp_t = fit_gp_sigma_clip(gp_t,X_t,fdt_pos,verbose=verbose)
        ftnd_t = gp_t.predict(X_t)
        fdt_t = y - ftnd_t

        gp_pos = fit_gp_sigma_clip(gp_pos,X_pos,fdt_t,verbose=verbose)
        ftnd_pos = gp_pos.predict(X_pos)

        fdt_pos_last = fdt_pos.copy()
        fdt_pos = y - ftnd_pos

        fdt_t_pos = y - ftnd_t - ftnd_pos
#        gp_t.nugget = gp_pos.nugget = fnugget(fdt_t_pos)

        if np.allclose(fdt_pos,fdt_pos_last) or (i >= 3):
            done = True
        i+=1


    lc['fdt_pos'] = fdt_pos
    lc['ftnd_pos'] = ftnd_pos

    lc['fdt_t'] = fdt_t
    lc['ftnd_t'] = ftnd_t
    return lc,gp_t,gp_pos

def decorrelate_position_and_time_1D(lc,verbose=False):
    lc = Lightcurve(lc)
    medflux = np.median(lc['f'])
    lc['f'] /= medflux
    lc['f'] -= 1
    lc.set_position_PCs()

    gpkw = dict(
        regr='constant',corr='squared_exponential',nugget=fnugget(lc['f'])
        )

    gp_t = GaussianProcess(theta0=3,**gpkw)
    gp_pos = GaussianProcess(theta0=0.03,thetaL=0.01,thetaU=0.1,**gpkw)

    lc['fdt_pos_last'] = lc['f'].copy()
    lc['fdt_pos'] = lc['f'].copy()

    i = 0
    done = False
    while not done:
        ndiff = (lc['fdt_pos'] - lc['fdt_pos_last']) / lc['fdt_pos_last']
        chi2 = np.sum(ndiff**2) / len(lc)
        if verbose:
            print i, chi2

        # Detrend against time
        lc_gp = Lightcurve(lc[~lc.fmask])
        gp_t = fit_gp_sigma_clip(
            gp_t,lc_gp.get_X('t') ,np.array(lc_gp['fdt_pos']),verbose=verbose)

        lc['ftnd_t'] = gp_t.predict( lc.get_X(['t']) )
        lc['fdt_t'] = lc['f'] - lc['ftnd_t']
        lc_segments = split_lc(lc)
        for lc_seg in lc_segments:
            
            lc_seg.__class__ = Lightcurve
            lc_seg_gp = Lightcurve(lc_seg[~lc_seg.fmask])

            gp_pos = fit_gp_sigma_clip(
                gp_pos, lc_seg_gp.get_X(['pos_pc0']), 
                np.array(lc_seg_gp['fdt_t']),
                verbose=verbose
                )

            lc_seg['ftnd_pos'] = gp_pos.predict(lc_seg.get_X(['pos_pc0']))

        lc = pd.concat(lc_segments)
        lc['fdt_pos_last'] = lc['fdt_pos'].copy()
        lc['fdt_pos'] = lc['f'] - lc['ftnd_pos']

        lc['ftnd_t_pos'] =  lc['ftnd_t'] + lc['ftnd_pos']
        lc['fdt_t_pos'] = lc['f'] - lc['ftnd_t_pos']

        gp_pos.nugget = gp_t.nugget = fnugget(lc['fdt_t_pos'])
        if np.allclose(lc['fdt_pos'],lc['fdt_pos_last']) or (i >= 5):
            done = True

        i+=1

    lc = lc.drop(['fdt_pos_last'],axis=1)
    for col in 'f ftnd_t fdt_t ftnd_pos fdt_pos ftnd_t_pos fdt_t_pos'.split():
        lc[col] +=1
        lc[col] *= medflux

    return lc

def FigureManager(basename,suffix=None):
    # Executes before code block
    plt.figure() 
    
    # Now run the code block
    yield

    # Executes after code block
    if basename is not None:
        plt.savefig(basename+suffix)

# Here's another way of writing the context manager using a class.

#class FigureManager(object):
#    def __init__(self,basename,suffix=None):
#        self.basename = basename
#        self.suffix = suffix
#    def __enter__(self):
#        plt.figure()
#    def __exit__(self, exc_type, exc_val, exc_tb):
#        if self.basename is not None:
#            plt.savefig(self.basename+self.suffix)


def plot_ses_vs_aperture_size(dflc):
    dflc['r'] = dflc.name.apply(lambda x : x.split('r=')[1][0]).astype(float)
    dflc['method'] = dflc.name.apply(lambda x : x.split('r=')[0][:-1])
    g = dflc.groupby('method')
    slcmin = dflc.ix[dflc['ses'].argmin()]

    plt.semilogy()
    for method,idx in g.groups.iteritems():
        df = dflc.ix[idx]
        plt.plot(df.r,df.ses,'o-',label=method)

    plt.plot(slcmin['r'],slcmin['ses'],'or',mfc='none',ms=15,mew=2,mec='r')
    plt.legend()

    xlab = 'Target Aperture Radius [pixels]'
    txtStr = 'Minimum: %(ses).1f ppm at R=%(r).1f pixels' % slcmin

    plt.xlabel(xlab)
    plt.ylabel('RMS [ppm]', )
    plt.minorticks_on()

    desc = dflc.ses.describe()
    factor = 1.2
    yval = desc['min']/factor,desc['max']*factor
    plt.ylim(yval)
    yticks = np.logspace(np.log10(yval[0]), np.log10(yval[1]), 8)
    plt.yticks(yticks, ['%i' % el for el in yticks])

def plot_pixel_decorrelation(lcFile):
    dflc = read_dflc(lcFile)
    lcmin = dflc.ix[dflc['ses'].idxmin(),'lc']
#    im = dflc.ix[dflc['ses'].idxmin(),'im']

    # Handle plotting
    basename = os.path.join(
        os.path.dirname(lcFile),
        os.path.basename(lcFile).split('.')[0]
        )


#    with FigureManager(basename,suffix='_0-median-frame.png'):
#        fr = ff.get_medframe()
#        fn = ff.fn
#        epic = fits.open(fn)[0].header['KEPLERID']
#        fr.plot()
#        fr.plot_label(fn,epic)


    with FigureManager(basename,suffix='_1-ses-vs-aperture-size.png'):
        plot_ses_vs_aperture_size(dflc)

    with FigureManager(basename,suffix='_2-pos_pc0.png'):
        fdt_t = ma.masked_array(lcmin['fdt_t'],lcmin['fmask'])
        plt.plot(lcmin['pos_pc0'],fdt_t,'.',
            label='Flux (High-pass Filtered)'
            )

        plt.plot(
            lcmin['pos_pc0'],lcmin['ftnd_pos'],'.',
            label='Flux Dependence\nagainst Principle Component 0'
            )

        plt.legend(loc='best')
        plt.xlabel('Principle Component 0 [pixels]')
        plt.ylabel('Flux')
        desc = lcmin.ftnd_pos.describe()
        spread = desc['max'] - desc['min']
        plt.ylim( desc['min'] - spread, desc['max'] + spread )
        plt.gcf().set_tight_layout(True)

    with FigureManager(basename,suffix='_3-gp_t_pos.png'):
        plot_detrend(lcmin,'f ftnd_t_pos fdt_t_pos'.split())

    with FigureManager(basename,suffix='_4-gp_t_pos_zoom.png'):
        plot_detrend(lcmin,'f ftnd_t_pos fdt_t_pos'.split())
        desc = lcmin.ftnd_t_pos.describe()
        spread = desc['max'] - desc['min']
        plt.ylim( desc['min'] - spread, desc['max'] + spread )
        plt.gcf().set_tight_layout(True)

def read_dflc(path):
    with h5py.File(path) as h5:
        groupnames = [item[0] for item in h5.items()]

    if np.any(np.array([n.count('mov') for n in groupnames]) > 0):
        groupnames = [n for n in groupnames if n.count('mov') > 0]
        
    dflc = []
    for gname in groupnames:
        s = pd.read_hdf(path,'%s/header' % gname) 
        s['lc'] = pd.read_hdf(path,'%s/lc' % gname)
        dflc += [s]

    dflc = pd.DataFrame(dflc)
    return dflc



outfmt = dict(marker='o',mew=0,mfc='r',alpha=0.5,lw=0,zorder=0,ms=8)
infmt = dict(marker='.',lw=0,zorder=5)

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


#    t = lc_gp[tkey] # Independent variable
#    y = lc_gp[ykey] # Error on dependent variable
#    yerr = 1e-4

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

def square_plots(n):
    sqn = sqrt(n)
    ncol = int(sqn) + 1
    nrow = int(np.ceil(n / ncol))
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


def detrend_roll_seg(lc,plot_diag=False):
    print "Light curve: ntotal=%i nfmask=%i nfdtmask=%i" % \
        (len(lc),lc['fmask'].sum(),lc['fdtmask'].sum())

    tkey = 'roll' # name of dependent variable
    ykey = 'fdt_t' # name of independent variable
    # Update fields
    fdtkey = 'fdt_t_roll' 
    ftndkey = 'ftnd_t_roll' 
    lc.index = lc.cad # Index by cadence
    nseg = len(segments)
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


def pixel_decorrelation(pixfile,lcfile,debug=False):
    """
    Run the pixel decorrelation on pixel file
    """
    basename = os.path.splitext(lcfile)[0]
    starname = os.path.basename(basename)

    # Load up pixel file
    im = ImageStack(pixfile)
    im.ts['fbg'] = im.get_fbackground()
    im.flux -= im.ts['fbg'][:,np.newaxis,np.newaxis]
    wcs = get_wcs(im.fn)
    ra,dec = im.headers[0]['RA_OBJ'],im.headers[0]['DEC_OBJ']
    x,y = wcs.wcs_world2pix(ra,dec,0)
    x = float(x)
    y = float(y)

    lc = im.ts # This is a skeleton light curve

    # Load up transformation information
    transfile = os.environ['K2_DIR']  
    transfile = os.path.join(transfile,'JayAnderson/','pixeltrans_ch04.h5')
    trans,pnts = channel_centroids.read_channel_centroids(transfile)
    trans = pd.DataFrame(trans)
    A = trans['A']
    B = trans['B']
    C = trans['C']
    D = trans['D']
    scale = sqrt(A*D-B*C) - 1
    theta = arctan2(B-C,D+A) 
    s1 = (A-D)/2  
    s2 = (B+C)/2      
    trans['roll'] = theta * 2e5 # arcsec

    # Merge transformation info with the rest of the light curve
    pnts = pd.DataFrame(pnts[0]['cad'.split()])
    trans = pd.concat([trans,pnts],axis=1)
    lc = pd.merge(trans,lc,on='cad')

    # Perform an initial run using a r=3 pixel aperture. We'll grab
    # the outliers from this run and use it in subsequent runs

    r = 3
    im.set_apertures(x,y,r)
    lc['fsap'] = im.get_sap_flux()

    # Standardize the data

    norm = Normalizer(lc['fsap'].median())
    lc['f'] = norm.norm(lc['fsap'])
    f0 = np.array(lc['f'].copy())
    lc['fmask'] = lc['f'].isnull()
    lc['fdtmask'] = lc['fmask'].copy()
    lc = detrend_t(lc,plot_diag=True) # sets fdt_t 
    lc = detrend_roll_seg(lc,plot_diag=True)
    lc['fdtmask'] = lc['fmask'] | lc['f'].isnull()

    # Generate some diagnostic plots of initial run 
    figures = [manager.canvas.figure for manager in Gcf.get_all_fig_managers()]
    for i,fig in enumerate(figures):
        figpath = basename+"_iter=-1_%i.png" % i
        fig.savefig(figpath)
    plt.close('all')

    dfiter = pd.DataFrame(
        dict(
            niter=[5,3,3],
            sigma_clip=[5,3,2],
            plot_diag=[False,False,True]
            )
        )
    

    if debug:
        dfiter = pd.DataFrame(
            dict(
                niter=[1],sigma_clip=[5], plot_diag=[True]
                ) 
            )     
        
    # Perform first round of iterative detrending 
    lc = detrend_t_roll_iter(lc,f0)
    for i,row in dfiter.iterrows():
        # Figure out which observations were outliers and repeat
        sig = np.median(np.abs(lc['fdt_t_roll']))*1.5
        boutlier = np.array(np.abs(lc['fdt_t_roll']) > row['sigma_clip']*sig)
        lc['fdtmask'] = lc['fdtmask'] | boutlier
        lc = detrend_t_roll_iter(
            lc,f0,niter=row['niter'],plot_diag=row['plot_diag']
            )

        figures = [m.canvas.figure for m in Gcf.get_all_fig_managers()]
        for j,fig in enumerate(figures):
            figpath = basename+"_iter=%i_%i.png" % (i,j)
            fig.savefig(figpath)
        plt.close('all')

    # Save the mask for future runs
    fdtmask = lc['fdtmask'].copy()
    lc.to_hdf(basename,'%i' % r)

    # Now rerun for different apertures
    lcmin = None
    minnoise = inf
    minr = inf

    for r in range(2,8):
        im.set_apertures(x,y,r)
        lc['fsap'] = im.get_sap_flux()
        # Standardize the data
        norm = Normalizer(lc['fsap'].median())
        lc['f'] = norm.norm(lc['fsap'])        

        f0 = np.array(lc['f'].copy())

        lc['fdtmask'] = fdtmask
        lc = detrend_t(lc,plot_diag=True) # sets fdt_t 
        lc = detrend_roll_seg(lc,plot_diag=True)
        lc = detrend_t_roll_iter(lc,f0,niter=5,plot_diag=False)

        lc['f'] = f0
        lc = detrend_t_roll_2D(lc)


        noisekey = 'fdt_t_roll_2D'
        fm = ma.masked_array(lc[noisekey],lc['fmask']) 
        dfses = ses_stats(fm)
        dfses.index = dfses.name
        noise = dfses.ix['mad_6-cad-mtd','value']
        
        keys = "mad_1-cad-mean mad_6-cad-mtd".split()
        print "%s: r=%i " % (starname,r) + \
            " ".join( ["%s=%.1f" % (k,dfses.ix[k,'value'])  for k in keys] ) 

        if noise < minnoise:
            minr = r
            minnoise = noise

            for k in 'f fdt_t ftnd_t fdt_t_roll ftnd_t_roll fdt_t_roll_2D ftnd_t_roll_2D'.split():
                lc[k] = norm.unnorm(lc[k])
                lcmin = lc.copy()

    # Now set with the least noisey light curve
    lc = lcmin.copy()
    plot_detrend(lc,'f ftnd_t_roll_2D fdt_t_roll_2D'.split())
    fig = gcf()

    # Save version that shows full range 
    yl = roblims(lc['ftnd_t_roll_2D'],5,2)
    ylim(*yl)
    fig.savefig(basename+"_fdt_t_roll_2D.png")
    
    # Save version that shows full range 
    yl = roblims(lc['fdt_t_roll_2D'],5,2)
    ylim(*yl)
    fig.savefig(basename+"_fdt_t_roll_2D_zoom.png")

    outfile = basename+".h5"
    lc.to_hdf(outfile,'%i' % r)

def detrend_t_roll_2D(lc):
    tkey = 't roll'.split() # name of dependent variable
    ykey = 'f' # name of independent variable
    fdtkey = 'fdt_t_roll_2D' 
    ftndkey = 'ftnd_t_roll_2D' 
    lc_gp = lc[~lc.fdtmask]
    b = np.array(~lc.fdtmask)
    x = np.array(lc_gp[tkey])
    y = lc[ykey][b]
    yerr = 1e-4
    k2d = 1.*kernels.ExpSquaredKernel([4**2,10**2],ndim=2) 
    gp = george.GP(k2d)
    gp.compute(x,yerr)
    mu,cov = gp.predict(y,lc[tkey])
    lc[ftndkey] = mu
    lc[fdtkey] = lc[ykey] - lc[ftndkey]
    return lc

        
if __name__ == "__main__":
    p = ArgumentParser(description='Pixel Decorrelation')
    p.add_argument('pixfile',type=str)
    p.add_argument('lcfile',type=str)
    args  = p.parse_args()
    pixel_decorrelation(args.pixfile,args.lcfile,debug=False)


