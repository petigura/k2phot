from argparse import ArgumentParser
import contextlib

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

class decorlc(object):
    def __init__(self,lc0):
        """
        """
        # Protect the input
        lc = lc0.copy()

        # Save a copy of the not normalized flux
        lc['f_not_normalized'] = lc['f']
        lc['f'] /= lc.f.median()
        lc['f'] -= 1

        self.nugget = (1.6*np.median(np.abs(lc.f)))**2
        corrlen = 10

        self.time_theta = 1./corrlen
        self.vmin,self.vmax = np.percentile(lc.f,[1,99])
        lc['gpmask'] = ~lc.f.between(self.vmin,self.vmax)

        self.lc = lc # Save array with out dropped columns

    def get_lc_gp(self):
        """
        Get light curve used in gp (uses fmask and gpmask)
        """
        lc = self.lc
        mask =  lc.fmask | lc.gpmask
        return lc[~mask]

    def set_position_PCs(self):
        lc = self.lc
        X = lc_to_X(lc,'dx dy'.split())
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)
        X_r = X_r / np.std(X_r,axis=0)
        for i in [0,1]:
            lc['pos_pc%i' %i ] = X_r[:,i]
        self.lc = lc

    def plot_detrend(self,columns):
        plot_detrend(self.lc,columns)

    def decorrelate_position(self):
        self.decorrelate_gp(
            self.gp_xy,'_xy','pos_pc0 pos_pc1'.split())

    def plot_position_PCs(self):
        lc = self.lc
        test = plt.scatter(
            lc.pos_pc0,lc.pos_pc1,c=lc.f,vmin=self.vmin,vmax=self.vmax,
            linewidths=0,alpha=0.8,s=20)

        plt.xlabel('pos_pc0')
        plt.ylabel('pos_pc1')
        plt.colorbar()
        return test

    def plot_gp_xy(self):
        lc = self.lc
        desc = lc.describe()
        res = 50
        lim1 = desc.ix['min','pos_pc0'], desc.ix['max','pos_pc0']
        lim2 = desc.ix['min','pos_pc1'], desc.ix['max','pos_pc1']
        x1, x2 = np.meshgrid(np.linspace(lim1[0],lim1[1], res),
                             np.linspace(lim2[0],lim2[1], res))
        xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
        y_pred, MSE = self.gp_xy.predict(xx, eval_MSE=True)
        y_pred = y_pred.reshape((res,res))
        extent = (lim1[0],lim1[1],lim2[0],lim2[1])
        plt.imshow(
            np.flipud(y_pred), alpha=0.8, extent=extent,aspect='auto',
            vmin=self.vmin,vmax=self.vmax)

    def decorrelate_position_and_time(self):
        self.decorrelate_gp(
            self.gp_time_xy,'_time_xy','t pos_pc0 pos_pc1'.split())

    def decorrelate_gp(self,gp,suffix,columns):
        lc_gp = self.get_lc_gp()

        # Fit the GP light curve with dropped indecies
        X = lc_to_X(lc_gp,columns)
        gp = gp.fit(X,lc_gp['f'])

        # Use the GP to predict the flux for masked and unmasked light curve

        lc = self.lc
        X = lc_to_X(lc,columns)
        gpcolumn = 'f_gp'+suffix
        lc[gpcolumn] = gp.predict(X)
        lc['fdt'+suffix] = lc['f'] - lc[gpcolumn]
        self.lc = lc

    def plot_decorrelate_position_and_time(self):
        columns = 'f f_gp_time_xy fdt_time_xy'.split()
        self.plot_detrend(columns)


class Lightcurve(pd.DataFrame):
    def set_position_PCs(self):
        X = lc_to_X(self,'dx dy'.split())
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)
        X_r = X_r / np.std(X_r,axis=0)
        for i in [0,1]:
            self['pos_pc%i' %i ] = X_r[:,i]

    def plot_position_PCs(self):
        lc = self.lc
        test = plt.scatter(
            lc.pos_pc0,lc.pos_pc1,c=lc.f,vmin=self.vmin,vmax=self.vmax,
            linewidths=0,alpha=0.8,s=20)

        plt.xlabel('pos_pc0')
        plt.ylabel('pos_pc1')
        plt.colorbar()
        return test



fmad = lambda x : np.median(abs(x))
fnugget = lambda x : (1.6*fmad(x))**2

def fit_gp_sigma_clip(gp,X,y):
    binlier = np.ones(X.shape[0]).astype(bool)
    binlier_last = binlier.copy()
    colors='rcyrcy'
    i = 0 
    while (not np.all(binlier==binlier_last)) or (i==0):
        print binlier.sum(),i
        gp.fit(X[binlier,:],y[binlier])
        y_pred = gp.predict(X)
        mad = fmad(y - y_pred)

        binlier_last = binlier.copy()
        binlier = abs(y - y_pred) < 4*mad
        i+=1

    return gp

def fit_gp_time_and_pos(lc):
     gpkw = dict(regr='constant',corr='squared_exponential',nugget=fnugget(lc['f']))

    X_t = lc_to_X(lc,'t')
    X_pos = lc_to_X(lc,['pos_pc0','pos_pc1'])
    y = np.array(lc['f'])

    gp_t = GaussianProcess(theta0=3,**gpkw)

    thetaU = [1,1]
    thetaL = [1e-3,1e-3]
    theta0 = [1e-2,1e-2]
    gp_pos = GaussianProcess(theta0=theta0,thetaU=thetaU,thetaL=thetaL,**gpkw)

    fdt_pos_last = y.copy()
    fdt_pos = y.copy()

    i = 0
    while (not np.all(fdt_pos==fdt_pos_last)) or (i==0):
        gp_t = fit_gp_sigma_clip(gp_t,X_t,fdt_pos)
        ftnd_t = gp_t.predict(X_t)
        fdt_t = y - ftnd_t

        gp_pos = fit_gp_sigma_clip(gp_pos,X_pos,fdt_t)
        ftnd_pos = gp_xy.predict(X_pos)

        fdt_pos_last = fdt_pos.copy()
        fdt_pos = y - ftnd_pos

        fdt_t_pos = y - ftnd_t - ftnd_pos
        gp_t.nugget = gp_pos.nugget = fnugget(fdt_t_pos)
        
        i+=1

    return ftnd_pos,ftnd_t



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
    fses,ftndses,fdtses = map(get_ses,res)

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

def read_weight_file(h5filename):
    with h5py.File(h5filename,'r') as h5:
        groupnames = [i[0] for i in h5.items()]

    dL = []
    for im,groupname in enumerate(groupnames):
        im = flatfield.read_hdf(h5filename,groupname)
        dL+=[dict(name=groupname,im=im)]
    dfweights = pd.DataFrame(dL)
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

def decorrelate_position_and_time(dc):
    dc.set_position_PCs()
    gpkw = dict(regr='constant',corr='squared_exponential',nugget=dc.nugget)
    thetaU = [1,1]
    thetaL = [1e-3,1e-3]
    theta0 = [1e-2,1e-2]
    dc.gp_xy = GaussianProcess(theta0=theta0,thetaU=thetaU,thetaL=thetaL,**gpkw)
    dc.decorrelate_position()

    # time and xy GP decorrelation
    theta0 = [dc.time_theta] + list(dc.gp_xy.theta_[0])
    dc.gp_time_xy = GaussianProcess(theta0=theta0,**gpkw)
    dc.decorrelate_position_and_time()
    return dc

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

def pixel_decorrelation_plots(dfdc,dcmin,basename=None):
    with FigureManager(basename,suffix='_0-ses-vs-aperture-size.png'):
        plot_ses_vs_aperture_size(dfdc)
        
    with FigureManager(basename,suffix='_xy.png'):
        dcmin.plot_position_PCs()
        dcmin.plot_gp_xy()

    with FigureManager(basename,suffix='_gp_time_xy.png'):
        dcmin.plot_decorrelate_position_and_time()

def plot_ses_vs_aperture_size(dfdc):
    dfdc['r'] = dfdc.name.apply(lambda x : x.split('r=')[1][0]).astype(float)
    dfdc['method'] = dfdc.name.apply(lambda x : x.split('r=')[0][:-1])
    g = dfdc.groupby('method')
    sdcmin = dfdc.ix[dfdc['ses'].argmin()]

    plt.semilogy()
    for method,idx in g.groups.iteritems():
        df = dfdc.ix[idx]
        plt.plot(df.r,df.ses,'o-',label=method)

    plt.plot(sdcmin['r'],sdcmin['ses'],'or',mfc='none',ms=15,mew=2,mec='r')
    plt.legend()

    xlab = 'Target Aperture Radius [pixels]'
    txtStr = 'Minimum: %(ses).1f ppm at R=%(r).1f pixels' % sdcmin

    plt.xlabel(xlab, )
    plt.ylabel('RMS [ppm]', )
    plt.minorticks_on()

    desc = dfdc.ses.describe()
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

def pixel_decorrelation(h5filename,debug=True):
    # Load up DataFrame with 
    dfdc = read_weight_file(h5filename)
    if debug:
        dfdc = dfdc[dfdc.name.str.contains('r=3|r=4')]

    dfdc['dc'] = map(im_to_dc,dfdc.im.tolist())
    dfdc['ses'] = None
    for index,sdc in dfdc.iterrows():
        dc = sdc['dc']
        dc = decorrelate_position_and_time(dc)
        fdt = ma.masked_array(dc.lc['fdt_time_xy'],dc.lc['fmask'])
        sdc['dc'] = dc
        sdc['ses'] = get_ses(fdt)
        print "%(name)s, mad_6-cad-mtd = %(ses)i" % sdc

    # Handle plotting
    basename = h5filename.split('.')[0]

    dfdc = dfdc.convert_objects()
    sdcmin = dfdc.ix[dfdc['ses'].argmin()]
    dcmin = sdcmin['dc']
    pixel_decorrelation_plots(dfdc,dcmin,basename=basename)
    return dfdc

if __name__ == "__main__":
    p = ArgumentParser(description='Pixel Decorrelation')
    p.add_argument('h5filename',type=str)
    p.add_argument('--debug',type=bool,default=False)
    args  = p.parse_args()
    pixel_decorrelation(args.h5filename,debug=args.debug)
