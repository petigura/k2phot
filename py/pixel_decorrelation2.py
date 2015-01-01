from argparse import ArgumentParser
import contextlib
import os

import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pylab as plt
import h5py
from sklearn.gaussian_process import GaussianProcess
from sklearn.decomposition import PCA,FastICA

from ses import ses_stats
import flatfield

def lc_to_X(lc,columns):
    """
    Convenience function to convert lc to X array
    
    Parameters 
    ----------
    lc : Pandas DataFrame
    columns : keys to construct X array
    """

    ncols = len(columns)
    X = np.array(lc[columns])
    if ncols==1:
        X = X.reshape(-1,1)
    return X

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
    plt.legend(**legkw)

    plt.sca(axL[1])
    plt.plot(t,fdt,label='Residuals SES = %i' % fdtses)
    plt.legend(**legkw)
    fig.set_tight_layout(True)
    plt.xlabel('Time')

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
                              dfweights.name.str.contains('mov=1_weight=1')]

    dfweights['im'] = ''
    for index,sweights in dfweights.iterrows():
        sweights['im'] = flatfield.read_hdf(h5filename,sweights['name'])

    return dfweights

def im_to_lc(im):
    frames = im.get_frames()
    moments = [frame.get_moments() for frame in frames]
    moments = pd.DataFrame(moments)
    lc = pd.concat([im.ts,moments],axis=1)
    lc['f'] = im.get_sap_flux()
    lc = pd.merge(lc,flatfield.cadmask,left_on='cad',right_index=True)
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
        regr='constant',corr='squared_exponential',nugget=fnugget(lc['f']))

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

        lc_segments = [lc.ix[idx] for idx in np.array_split(lc.index,6)]
        for lc_seg in lc_segments:
            lc_seg.__class__ = Lightcurve
            lc_seg_gp = Lightcurve(lc_seg[~lc_seg.fmask])

            gp_pos = fit_gp_sigma_clip(
                gp_pos, lc_seg_gp.get_X(['pos_pc0']), 
                np.array(lc_seg_gp['fdt_t']),
                verbose=verbose)

#            gp_pos.fit(X_pos,lc_seg['fdt_t'])
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

@contextlib.contextmanager
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

    plt.xlabel(xlab, )
    plt.ylabel('RMS c[ppm]', )
    plt.minorticks_on()

    desc = dflc.ses.describe()
    factor = 1.2
    yval = desc['min']/factor,desc['max']*factor
    plt.ylim(yval)
    yticks = np.logspace(np.log10(yval[0]), np.log10(yval[1]), 8)
    plt.yticks(yticks, ['%i' % el for el in yticks])

def plot_frames():
    py.figure(tools.nextfig(), [14, 10])
    fig = py.gcf()

    gs = GridSpec(6,2)
    # Axes for Time Series
    axL_ts = [fig.add_subplot(gs[i,:]) for i in range(3)]
    [py.setp(ax.get_xticklabels(), visible=False) for ax in axL_ts[1:]]

    # Axes for Time Series
    axL_im = [fig.add_subplot(gs[3:,i]) for i in range(2)]

    py.sca(axL_ts[0])
    py.plot(time, input.rawFlux, '.-k', mfc='c')
    py.ylabel('Raw Flux')

    py.sca(axL_ts[1])
    py.plot(time, input.x, '.-k', mfc='r')
    py.ylabel('X motion [pix]', )

    py.sca(axL_ts[2])
    py.plot(time, input.y, '.-k', mfc='r')
    py.ylabel('Y motion [pix]', )
    py.xlabel('BJD - 2454833', )

    cat = k2_catalogs.read_cat(return_targets=False)

    if hasattr(input,'catcut'):
        py.sca(axL_im[0])
        plot_label(input.medianFrame,input.catcut,input.epic, shift=shift)
        py.title("Median Frame")
        py.sca(axL_im[1])    
        logframe = np.log10(input.medianFrame)
        logframe = ma.masked_invalid(logframe)
        logframe.fill_value=0
        logframe = logframe.filled()
        plot_label(logframe,input.catcut,input.epic, shift=shift)
        py.title("log10(Median Frame)")
        

    for i in range(2):
        py.sca(axL_im[i])
        py.contour(input.crudeApertureMask, [0.5], colors='g', linewidths=2.5)
    #if input.apertureMode[0:4]=='circ':
    #    xy = list(input.loc)[::-1] # FLIP x and y
    #    args =  [xy[0]+shift[0], xy[1]+shift[1], input.apertures[0]]
    #    for i in range(2):
    #        py.sca(axL_im[i])
    #        tools.drawCircle(*args,color='lime', fill=False, linewidth=3)


    py.gcf().text(.5, .95, titstr, fontsize='large', ha='center')
    ax = py.axis()

    args = [input.x,input.y]
    args = args[::-1] # Flip X and Y !
    py.plot(*args,color='r',marker='.')

def plot_pixel_decorrelation(lcFile):
    dflc = read_dflc(lcFile)
    lcmin = dflc.ix[dflc['ses'].idxmin(),'lc']
    
    # Handle plotting
    basename = os.path.join(
        os.path.dirname(lcFile),
        os.path.basename(lcFile).split('.')[0]
        )

    with FigureManager(basename,suffix='_0-ses-vs-aperture-size.png'):
        plot_ses_vs_aperture_size(dflc)

    with FigureManager(basename,suffix='_pos_pc0.png'):
        fdt_t = ma.masked_array(lcmin['fdt_t'],lcmin['fmask'])
        plt.plot(lcmin['pos_pc0'],fdt_t,'.')
        plt.plot(lcmin['pos_pc0'],lcmin['ftnd_pos'],'.')

        desc = lcmin.ftnd_pos.describe()
        spread = desc['max'] - desc['min']
        plt.ylim( desc['min'] - spread, desc['max'] + spread )
        plt.gcf().set_tight_layout(True)

    with FigureManager(basename,suffix='_gp_t_pos.png'):
        plot_detrend(lcmin,'f ftnd_t_pos fdt_t_pos'.split())

    with FigureManager(basename,suffix='_gp_t_pos_zoom.png'):
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

def pixel_decorrelation(weightFile,debug=True):
    # Load up DataFrame with 
    dflc = read_weight_file(weightFile,debug=debug)
    dflc['lc'] = map(im_to_lc,dflc.im.tolist())
    dflc['ses'] = None

    basename = os.path.join(
        os.path.dirname(weightFile),
        os.path.basename(weightFile).split('_')[0]
    )

    lcFile = basename+'.h5'
    for index,slc in dflc.iterrows():
        lc = slc['lc']
        lc = decorrelate_position_and_time_1D(lc)
        fdt = ma.masked_array(lc['fdt_pos'],lc['fmask'])
        fdt = fdt / ma.median(fdt) - 1
        slc['lc'] = lc
        slc['ses'] = get_ses(fdt)

        slcsave = slc['name ses'.split()]
        slcsave.to_hdf(lcFile,'%(name)s/header' % slcsave )
        slc['lc'].to_hdf(lcFile,'%(name)s/lc' % slcsave)
        print "%(name)s, mad_6-cad-mtd = %(ses)i" % slc

    plot_pixel_decorrelation(lcFile)

if __name__ == "__main__":
    p = ArgumentParser(description='Pixel Decorrelation')
    p.add_argument('weightFile',type=str)
    p.add_argument('--debug',action='store_true')
    args  = p.parse_args()
    pixel_decorrelation(args.weightFile,debug=args.debug)
