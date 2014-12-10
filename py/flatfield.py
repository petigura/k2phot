"""
Routines that solve for the flat-field
"""
import cPickle as pickle
from argparse import ArgumentParser
import copy
import os.path

import sqlite3
import numpy as np
from numpy import ma
from scipy import optimize
import numpy as np
from pixel_decorrelation import imshow2


import pandas as pd
from astropy.io import fits
from photutils import CircularAperture,aperture_photometry
from photutils.aperture_funcs import do_circular_photometry
from photutils.aperture_core import _sanitize_pixel_positions

from scipy import ndimage as nd

import h5plus
from pdplus import LittleEndian as LE
import photometry
from pixel_decorrelation import imshow2,get_star_pos,loadPixelFile,get_stars_pix,subpix_reg_stack
from skimage import measure
from matplotlib import pylab as plt 
from astropy import units as u    
from pixel_decorrelation2 import plot_detrend

from itertools import product

import circular_photometry



cadmaskfile = os.path.join(os.environ['K2PHOTFILES'],'C0_fmask.csv')
cadmask = pd.read_csv(cadmaskfile,index_col=0)

class ImageStack(object):
    def __init__(self,fn,tlimits=None):
        cube,headers = loadPixelFile(fn,tlimits=tlimits)
        self.fn = fn
        self.headers = headers
        self.flux = cube['FLUX'].astype(float)
        self.t = cube['TIME'].astype(float)
        self.cad = cube['CADENCENO'].astype(int)
        self.ts = pd.DataFrame(dict(t=self.t , cad=self.cad))
        self.tlimits = tlimits

        # Number of frames, rows, and columns
        self.nframe,self.nrow,self.ncol = self.flux.shape 
        self.npix = self.nrow*self.ncol

    def set_apertures(self,locx,locy,radius):
        """
        Set the apertures used to compute photometry
        """
        if hasattr(locx,'__iter__'):
            assert ((len(locx)==self.nframe) &
                    (len(locy)==self.nframe) ), "Must have same length as array"
        self.ts['locx'] = locx
        self.ts['locy'] = locy

        def get_ap_weights(locx,locy):
            positions = np.array([[locx,locy]])
            return circular_photometry.circular_photometry_weights(
                self.flux[0],positions,radius)
            
        ap_weights = map(get_ap_weights,self.ts.locx,self.ts.locy)
        ap_weights = np.array(ap_weights)
        self.ap_weights = ap_weights

    def get_sap_flux(self):
        ap_flux = self.flux * self.ap_weights # flux falling in aperture
        ap_flux = ap_flux.reshape(self.nframe,-1)
        ap_flux = np.nansum(ap_flux,axis=1)
        return ap_flux

    def get_flux(self):
        """Get flux cube
        
        For Imstack object we don't modify the flux cube, so this is a
        no-op. For derived classes, we do and get_flux protechts
        self.flux
        """
        return self.flux

    def get_frame(self,i):
        locx,locy = self.ts.iloc[i]['locx locy'.split()]
        flux = self.get_flux()
        ap_weights = self.ap_weights[i]
        frame = Frame(
            flux[i],locx=locx,locy=locy,r=self.radius,ap_weights=ap_weights)
        return frame

class FlatField(ImageStack):
    """
    Stack of images that we can used to compute Flat Field
    """
    def __init__(self,*args,**kwargs):
        """
        """
        # Instantiate from ImageStack class
        super(FlatField,self).__init__(*args,**kwargs)
        self.weight_bounds = [0.9,1.1]

        # Bin up cadences used to compute robust fom
        self.dfcad = get_dfcad(self.cad)

        # Other default values for array 
        self.fmask = np.zeros(self.nframe).astype(bool)
        self.subtract_background()

        self.aperture_mode = None
        self.radius = None

    def get_fbackground(self):
        flux = self.flux.reshape(-1,self.npix,)
        fbackground = np.median(flux,axis=1)
        return fbackground

    def subtract_background(self):
        """
        Subtracts off background value from flux
        """
        fbackground = self.get_fbackground() 
        self.flux -= fbackground[:,np.newaxis,np.newaxis]  

    def set_dxdy_by_registration(self):
        assert hasattr(self,'weights')==False,\
            "Must determine dx/dy before setting weights"
        self.dx,self.dy = subpix_reg_stack(self.flux) 

    def set_loc(self,mode='moving'):
        """
        Determine location of the star.
        self.locxy : (x,y) position of star
        self.locrc : (r,c) position of star
        self.df : DataFrame with row column of each pixel
        """

        # Determine position using WCS
        pos_mode = 'wcs'
        xcen,ycen = get_star_pos(self.fn,mode=pos_mode)
        frame0 = ma.masked_invalid(self.flux)
        frame0 = ma.median(frame0,axis=0)
        frame0.fill_value = 0
        frame0 = frame0.filled()
        catcut, shift = get_stars_pix(self.fn,frame0)
        epic = self.headers[0]['KEPLERID']
        xcen,ycen = catcut.ix[epic]['pix0 pix1'.split()]
        print "Using position mode %s, star is at pixels = [%.2f,%.2f]" % \
            (pos_mode,xcen,ycen)

        self.locxy = (xcen,ycen)
        self.locrc = (ycen,xcen) 

        if mode=='moving':
            dx,dy = subpix_reg_stack(self.flux) 
            self.locx = xcen + dx
            self.locy = ycen + dy
            self.aperture_mode = mode
        if mode=='static':
            self.locx = np.zeros(self.nframe) + xcen 
            self.locy = np.zeros(self.nframe) + ycen 
            self.aperture_mode = mode

    def get_brightest_pixels(self,npix=20):
        """
        Get the brightest npixels pixels within 3 pixels of target star
        """

        # Store pixel values as a list
        row,col = np.mgrid[:self.nrow,:self.ncol]
        flux = ma.masked_invalid(self.flux)
        df = pd.DataFrame(dict(row=row.flatten(),col=col.flatten()))
        df['flux0'] = flux[0].flatten()
        df['fluxsum'] = ma.sum(flux,axis=0).flatten()
        df['dist'] = np.sqrt( (df.row - self.locrc[0])**2 + 
                              (df.col - self.locrc[1])**2 )
        df = pd.DataFrame(LE(df.to_records()))

        cut = df.query('dist < 3')
        cut = cut.sort('fluxsum',ascending=False).iloc[:npix]
        pixels = np.array(cut['row col'.split()])
        return pixels

    def set_apertures(self,radius):
        self.radius = radius 
        super(FlatField,self).set_apertures(self.locx,self.locy,radius)

    def set_ff_parameters(self, pixels, fmask=False):
        """
        Set the pixels used to for flat-fielding

        Parameters
        ----------
        pixels : (nframe,2) array specifying the row and column values
                 of the pixels to include in the flat field
                 re-weighting
        """

        # Create a 2D mask for weighted pixels.
        # True = pixel is used in the reweighting
        mask = np.zeros((self.nrow,self.ncol)).astype(bool)
        self.nweights = len(pixels)
        for i in range(self.nweights):
            r,c = pixels[i]
            mask[r,c] = True
        self.mask = mask
        self.pixels = pixels

        # Set initial weights to unity
        self.weights0 = np.ones(self.nweights) 
        self.weights = self.weights0.copy()
        self.weight_bounds = [0.9,1.1]
        self.fmask = fmask 
        self.f_total0_sum = self.get_sap_flux().sum()

    def get_weights_frame(self):
        """
        Return an image with the current value of the weights
        """
        weights_frame = np.ones((self.nrow,self.ncol))
        for i in range(self.nweights):
            r,c = self.pixels[i]
            weights_frame[r,c] = self.weights[i]
        return weights_frame

    def get_flux(self):
        flux = super(FlatField,self).get_flux()
        weights_frame = self.get_weights_frame()
        flux_reweighted = weights_frame * flux
        return flux_reweighted

    def get_sap_flux(self):
        """
        """
        ap_flux = self.get_flux() * self.ap_weights
        ap_flux = ap_flux.reshape(self.nframe,-1)
        ap_flux = np.nansum(ap_flux,axis=1)
        ap_flux = ma.masked_array(ap_flux,self.fmask)
        return ap_flux

    def figure_of_merit(self,metric):
        fm = self.get_sap_flux()
        if metric=='mad':
            fom = ma.median(ma.abs(fm - ma.median(fm)))
        if metric=='std':
            fom = ma.std(fm)
        if metric=='bin-med-std':
            fm.fill_value = np.nan
            fm.filled()
            fom = get_bin_med_std(self.dfcad,fm.filled())
        return fom 

    def solve_weights(self,metric):
        # Set bounds
        self.bounds = [self.weight_bounds]*self.nweights

        # Construct cost and constraint functions
        def func_fom(weights):
            self.weights = weights
            return self.figure_of_merit(metric)

        def func_constraint(weights):
            self.weights = weights
            constraint = self.get_sap_flux().sum() - self.f_total0_sum
            return constraint

        x0 = self.weights.copy()
        fom_initial = func_fom(x0)
        res = optimize.fmin_slsqp( 
            func_fom, x0, eqcons=[func_constraint], bounds=self.bounds,
            epsilon=0.01, iprint=1,iter=200)

        fom_final = func_fom(res)
        self.weights = res

        print "initial fom = %f" % fom_initial
        print "final fom = %f" % fom_final
        print "initial/final fom = %f" % (fom_initial/fom_final)
        self.fom_final = fom_final
        self.fom_initial = fom_initial
        return res
    
    def get_frame(self,i):
        super(FlatField,self).get_frame(i)
        frame.pixels = self.pixels
        return frame

    def __repr__(self):
        keys = 'nframe nrow ncol aperture_mode radius'.split()
        fmtd = dict([(k,getattr(self,k)) for k in keys])

        def fmtloc(s):
            if hasattr(self,s):
                fmtd[s] = "%.2f" % getattr(self,s)[0]
            else:
                fmtd[s] = "not set"
        
        s = 'radius'
        if hasattr(self,s):
            fmtd[s] = "%.1f" % getattr(self,s)
        else:
            fmtd[s] = "not set"

        map(fmtloc,'locx locy'.split())
        s = """\
<FlatField Object> 
Shape: (nframe,nrow,ncol) = (%(nframe)i,%(nrow)i,%(ncol)i) 
Reference Aperture: (x,y,r) = (%(locx)s,%(locy)s,%(radius)s )
Aperture Mode: %(aperture_mode)s""" % fmtd
        return s
    ## IO ##

    
    def to_hdf(self,filename,group):
        """
        
        """
        weights = pd.DataFrame(self.pixels,columns='row col'.split())
        weights['weight'] = self.weights
        
        keys = 'fn tlimits radius'.split()
        header = dict([(k,getattr(self,k)) for k in keys])
        header = pd.Series(header)

        weights.to_hdf(filename,'%s/weights' % group)
        header.to_hdf(filename,'%s/header' % group)
        self.ts.to_hdf(filename,'%s/ts' % group)

        print "saveing to %s[%s]" % (filename,group)

def weights_groupname(ff_par):
    return 'mov=%(mov)i_weight=%(weight)i_r=%(radius)i' % ff_par

def read_hdf(filename,group,fn=None):
    """
    Return image stack
    """
    weights = pd.read_hdf(filename,'%s/weights' % group)
    header = pd.read_hdf(filename,'%s/header' % group)
    ts = pd.read_hdf(filename,'%s/ts' % group)

    ff = FlatField(header['fn'],tlimits=header['tlimits'])

    ff.weights = weights['weight']
    ff.pixels = np.array(weights['row col'.split()])
    ff.nweights = len(weights['weight'])

    ff.flux = ff.get_flux() 
    ff.__class__ = ImageStack # FlatField -> ImageStack Object

    im = ff # Reminder that I'm dealing with an Image stack
    im.ts = ts
    im.set_apertures(im.ts['locx'],im.ts['locy'],header['radius'])

    return im

def flatfield_wrap(pixelfile,outdir,starname,tlimits=[-np.inf,np.inf]):
    ff = FlatField(pixelfile,tlimits=tlimits)
    radii = range(2,8)
#    radii = range(4,6)
    moving = [0,1]
    weighted = [0,1]
    ff_pars = list(product(moving,weighted,radii))
    ff_pars = pd.DataFrame(ff_pars,columns='mov weight radius'.split())
    for i in ff_pars.index:
        ff_par = ff_pars.ix[i]

        if ff_par['mov']==1:
            ff.set_loc(mode='moving')
        else: 
            ff.set_loc(mode='static')
            
        ff.set_apertures(ff_par['radius'])

        # Set brightest pixels to use for weighting
        pixels = ff.get_brightest_pixels(npix=20)
        ff.set_ff_parameters(pixels)

        if ff_par['weight']==1:
            ff.solve_weights('bin-med-std')
        else:
            pass

        ff.get_sap_flux()

        print ff
        groupname = weights_groupname(ff_par)
        import pdb;pdb.set_trace()

        h5filename = os.path.join(outdir,'%s_weights.h5' % starname)
        ff.to_hdf(h5filename,groupname)

    basename = starname
    basename = os.path.join(outdir,basename)

class Frame(np.ndarray):
    def __new__(cls, input_array,locx=None,locy=None,r=None,pixels=None,ap_weights=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.locx = locx
        obj.locy = locy
        obj.r = r
        obj.pixels = pixels
        obj.ap_weights = ap_weights

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

    def plot(self):
        apertures = CircularAperture([self.locx,self.locy], r=self.r)
        logflux = np.log10(self)
        logflux = ma.masked_invalid(logflux)
        logflux.mask = logflux.mask | (logflux < 0)
        logflux.fill_value = 0
        logflux = logflux.filled()
        imshow2(logflux)

        if self.pixels is not None:
            for i,pos in enumerate(self.pixels):
                r,c = pos
                plt.text(c,r,i,va='center',ha='center',color='Orange')
        apertures.plot(color='Lime',lw=1.5,alpha=0.5)

    def get_moments(self):
        """
        Return moments from frame:
        Future work: This should be a method on a FrameObject
        """
        frame = ma.masked_invalid(self)
        frame.fill_value = 0
        frame = frame.filled()
        frame *= self.ap_weights # Multiply frame by aperture weights

        # Compute the centroid
        m = measure.moments(frame)
        moments = pd.Series(dict(m10=m[0,1],m01=m[1,0]))
        moments /= m[0,0]

        # Compute central moments (second order)
        mu = measure.moments_central(frame,moments['m10'],moments['m01'])
        c_moments = pd.Series(
            dict(mupr20=mu[2,0], mupr02=mu[0,2], mupr11=mu[1,1]) )
        c_moments/=mu[0,0]
        moments = pd.concat([moments,c_moments])
        return moments

def test_frame_moments():
    flux = np.ones((20,20))
    locx,locy = 7.5,7.5
    positions = np.array([locx,locy]).reshape(-1,2)
    radius = 3
    ap_weights = circular_photometry.circular_photometry_weights(
        flux,positions,radius)
    frame = Frame(flux,locx=locx,locy=locy,r=radius,ap_weights=ap_weights)

def test_restore():
    from matplotlib.pylab import *
    r = 3
    mov = 'moving'
    ff = FlatField('pixel/C0/ktwo202126880-c00_lpd-targ.fits',tlimits=[1960,1975])
    ff.set_loc(mode=mov)
    ff.set_apertures(r)
    ff.set_ff_parameters(ff.get_brightest_pixels())
    fm_before_fit = ff.get_sap_flux()
    ff.solve_weights('bin-med-std')
    fm_after_fit = ff.get_sap_flux()
    ff.to_hdf('weights.h5','mov=1_weight=1_r=3')
    im = read_hdf('weights.h5','mov=1_weight=1_r=3')
    fm_after_fit_resotred = im.get_sap_flux()
    clf()
    plot(fm_before_fit,label='before re-weighting')
    plot(fm_after_fit,label='after reweighting')
    plot(fm_after_fit_resotred,label='after resotre and reweighting')
    legend()


def get_dfcad(cad, n_cad_per_bin=50):
    """
    Get cadence DataFrame

    Parameters
    ----------
    cad : sequence of cadences
    
    Take an array of cadences and return a dataframe with the
    following parameters

    """

    dcad = cad[1:] - cad[:-1]
    assert np.max(dcad) < 10,'Long gap in cadences, binning no longer accurate'

    cad = pd.DataFrame(dict(cad=cad))
    cad['cadbin'] = -1
    idxL = np.array(cad.index)
    idxL = np.array_split(idxL,len(idxL)/n_cad_per_bin)
    for idx in idxL:
        cad.ix[idx,'cadbin'] = cad.ix[idx].iloc[0]['cad']
    return cad    


def get_bin_med_std(dfcad,x):
    """
    Get Median Standard Devation of Bins

    Parameters 
    ----------
    dfcad : Pandas DataFrame with cadbin
    x : array for which to compute the binned median std
    """
    
    cad = dfcad.copy()
    cad['x'] = x 
    g = cad.groupby('cadbin')
    # Compute FOM, ignoring masked cadences
    fom = np.median(g.std()['x']) 
    return fom


if __name__ == "__main__":
    np.set_printoptions(precision=4)

    p = ArgumentParser(description='Generate flat field weight')
    p.add_argument('pixelfile',type=str)
    p.add_argument('outdir',type=str)
    p.add_argument('starname',type=str)
    p.add_argument('--tmin',type=float,default=-np.inf)
    p.add_argument('--tmax',type=float,default=np.inf)

    args  = p.parse_args()
    pixelfile = args.pixelfile
    outdir = args.outdir
    tlimits=[args.tmin,args.tmax]

    flatfield_wrap(args.pixelfile,args.outdir,args.starname,tlimits=tlimits)

