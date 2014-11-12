#from matplotlib.pylab import *
import matplotlib.pylab as plt
import numpy as np
from numpy import ma

from sklearn.decomposition import PCA,FastICA
import pandas as pd
from photometry import r2fm
from sklearn.gaussian_process import GaussianProcess
from scipy import optimize

from ses import ses_stats
from argparse import ArgumentParser
import os
import cPickle as pickle
import sqlite3

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
    def __init__(self,lc):
        """
        Normalize the light curve

        Compute its varience

        Two versions of light curve:
        - lc0 : complete light-curve no points masked out
        - lc : bad points removed
        """
        # Protect the input
        self.lc_inp = lc.copy()
        lc = lc['t cad f fmask dx dy'.split()]
        lc['f'] /= lc.f.median()
        lc['f'] -= 1

        self.nugget = (1.6*np.median(np.abs(lc.f)))**2
        corrlen = 40

        self.time_theta = 1./corrlen

        self.vmin,self.vmax = np.percentile(lc.f,[1,99])

        dropidx = lc[lc.fmask | (~lc.f.between(self.vmin,self.vmax)) ].index
        print "Removing %i/%i cadences (fmask)" % (len(dropidx),len(lc))
        lc0 = lc.copy()
        lc = lc.drop(dropidx)        

        self.lc0 = lc0 # Save array with out dropped columns
        self.lc = lc

    def decorrelate_time(self,plot_diag=False):
        """
        Decorelate against time
        """
        lc = self.lc
        gp_time = GaussianProcess(
            theta0=self.time_theta,regr='constant',corr='squared_exponential',
            nugget=self.nugget**2)

        # X array is just the time array
        X = lc_to_X(lc,['t'])
        y = np.array(lc['f'])
        gp_time.fit(X,y)

        # Calculate the GP values for time
        lc['f_gp_t'] = gp_time.predict(X)
        lc['fdt'] = lc['f'] - lc['f_gp_t']
        self.lc = lc

    def plot_decorrelate_time(self):
        lc = self.lc
        fig,axL = plt.subplots(nrows=2,figsize=(12,3))

        plt.sca(axL[0])
        plt.plot(lc['t'],lc['f'],label='Raw Flux')
        plt.plot(lc['t'],lc['f_gp_t'],label='GP against time')
        plt.legend()

        plt.sca(axL[1])
        plt.plot(lc['t'],lc['fdt']+0.01,label='GP removed')
        plt.legend()

        fig.set_tight_layout(True)

    def get_position_PCs(self):
        lc = self.lc0
        X = lc_to_X(lc,'dx dy'.split())
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)

#        ica = FastICA(n_components=2)
#        X_r = ica.fit_transform(X) 

        X_r = X_r / np.std(X_r,axis=0)

        for i in [0,1]:
            lc['pos_pc%i' %i ] = X_r[:,i]

        self.lc0 = lc
        self.lc = pd.merge(self.lc,self.lc0['pos_pc0 pos_pc1'.split()],
                           left_index=True,right_index=True)

    def plot_detrend(self,columns):
        plot_detrend(self.lc,columns)

    def decorrelate_position(self):
        """
        Must have previously decorrelated against time

        kwargs are passed to GP
        """
        lc = self.lc
        gp_xy = self.gp_xy
        X = lc_to_X(lc,'pos_pc0 pos_pc1'.split())
        gp_xy = gp_xy.fit(X, lc.fdt)
        lc['f_gp_xy'] = gp_xy.predict(X)
        lc['fdtxy'] = lc['fdt'] - lc['f_gp_xy']

        self.gp_xy = gp_xy
        self.lc = lc

    def plot_position_PCs(self):
        lc = self.lc
        vmin,vmax = np.percentile(lc.f,[1,99])
        test = plt.scatter(lc.pos_pc0,lc.pos_pc1,c=lc.fdt,vmin=self.vmin,
                           vmax=self.vmax,linewidths=0,alpha=0.8,s=20)

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
            np.flipud(y_pred), alpha=0.8, extent=extent,vmin=self.vmin,vmax=self.vmax,
            aspect='auto')

    def plot_decorrelate_position(self) :
        lc = self.lc
        fig,axL = plt.subplots(nrows=2,figsize=(12,6),sharex=True,sharey=True)

        plt.sca(axL[0])
        plt.plot(lc['t'],lc['fdt'],label='Flux (time-dependence removed)')
        plt.plot(lc['t'],lc['f_gp_xy'],label='GP (position)')
        plt.legend()

        plt.sca(axL[1])
        plt.plot(lc['t'],lc['fdtxy'],label='GP removed')
        plt.legend()

        fig.set_tight_layout(True)
        plt.xlabel('Time')

    def decorrelate_position_and_time(self):
        """
        Must have previously decorrelated against time
        """
        self.decorrelate_gp(
            self.gp_time_xy,'_time_xy','t pos_pc0 pos_pc1'.split())

    def decorrelate_gp(self,gp,suffix,columns):
        # Fit the GP using the masked light curv
        X = lc_to_X(self.lc,columns)
        gp = gp.fit(X, self.lc['f'])

        # Use the GP to predict the flux for masked and unmasked light curve
        for attr in 'lc lc0'.split():
            lc = getattr(self,attr)
            X = lc_to_X(lc,columns)
            gpcolumn = 'f_gp'+suffix
            lc[gpcolumn] = gp.predict(X)
            lc['fdt'+suffix] = lc['f'] - lc[gpcolumn]
            setattr(self,attr,lc)

    def plot_decorrelate_position_and_time(self):
        columns = 'f f_gp_time_xy fdt_time_xy'.split()
        self.plot_detrend(columns)



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

def fmad(x):
    """
    Convenience function to compute second moment using L1 norm
    """
    return np.median(np.abs(x - np.median(x)))

def decorrelate_position_and_time_hepler(lc):
    dc = decorlc(lc)
    dc.get_position_PCs()

    gpkw = dict(regr='constant',corr='squared_exponential',nugget=dc.nugget)

    # Determine the GP parameters
    dc.decorrelate_time()
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

def decorrelate_position_and_time(lc):
    # Run once through to clip out outliers
    dc = decorrelate_position_and_time_hepler(lc)
    #    x = dc.lc['fdt_time_xy']
    #    mad = fmad(x)
    #    outliers = np.abs(x - np.median(x)) > 10*mad
    #    plt.ion()
    #    import pdb;pdb.set_trace()
    #    lc['fmask'] = lc['fmask'] | outliers

#    lc['f'] = dc.lc['fdt_time_xy'] + 1
#    dc = decorrelate_position_and_time_hepler(lc)
    return dc


def get_ses(f):
    ses = ses_stats(f)
    ses.index = ses.name
    ses = ses.ix['mad_6-cad-mtd','value']
    return ses

def decorrelate_position_and_time_wrap(lc0):
    radiusL = range(3,4)
    def f(radius):
        lc = lc0.copy()
#        fcolumn = 'f_weighted%i' % radius
#        fcolumn = 'f_unweighted'
        fcolumn = 'f_weighted' 
        lc = lc.rename(columns={fcolumn:'f'})
        dc = decorrelate_position_and_time(lc)
        fdt = ma.masked_array(dc.lc['fdt_time_xy'],dc.lc['fmask'])
        ses = get_ses(fdt)
        print "radius = %i, mad_6-cad-mtd = %i" % (radius,ses)
        return dc,ses
        
    res = map(f,radiusL)
    dcL,sesL = map(list,zip(*res))

    idxmin = np.argmin(sesL)
    radiusmin = radiusL[idxmin]
    sesmin = sesL[idxmin]
    dc = dcL[idxmin]
    print "radius %i had minimum mad_6-cad-mtd at %i" % (radiusmin,sesmin)
    return dc

if __name__ == "__main__":


    p = ArgumentParser(description='Difference Images')
    p.add_argument('outdir',type=str)
    p.add_argument('--h5file',type=str,default='')

    args  = p.parse_args()
    outdir = args.outdir

    if args.h5file!='':
        lc = pd.read_hdf(args.h5file,'lc')
        starname = args.h5file.split('/')[-1].split('.')[0]
    else:
        df = pd.read_sql('select * from phot',sqlite3.connect('test.sqlite'))
        g = df.groupby('starname')
        lc =  df.ix[g.groups[starname]]

    dc = decorrelate_position_and_time_wrap(lc)

    basename = os.path.join(outdir,starname)

    dc.lc.to_hdf("%s.h5" % basename,'lc')
    dc.lc0.to_hdf("%s.h5" % basename,'lc0')

    dc.plot_position_PCs()
    dc.plot_gp_xy()
    plt.gcf().savefig("%s_xy.png" % basename)

    dc.plot_decorrelate_position_and_time()
    plt.gcf().savefig("%s_gp_time_xy.png" % basename)
