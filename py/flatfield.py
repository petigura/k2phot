"""
Routines that solve for the flat-field
"""
from argparse import ArgumentParser
import os.path
from itertools import product

import numpy as np
from numpy import ma
from scipy import optimize
import pandas as pd
from astropy.io import fits

from photutils.aperture_core import _sanitize_pixel_positions
from pdplus import LittleEndian as LE

from pixel_decorrelation import get_star_pos, loadPixelFile, get_stars_pix, \
    subpix_reg_stack

from imagestack import ImageStack


def add_cadmask(lc,k2_camp):
    """
    Add Cadence Mask
    """
    
    if k2_camp=='C0':
        cadmaskfile = os.path.join(os.environ['K2PHOTFILES'],'C0_fmask.csv')
        cadmask = pd.read_csv(cadmaskfile,index_col=0)
        lc = pd.merge(lc,flatfield.cadmask,left_on='cad',right_index=True)
    elif k2_camp=='C1':


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
    im.radius = header['radius']
    im.set_apertures(im.ts['locx'],im.ts['locy'],im.radius)

    return im

def flatfield_wrap(pixelfile,outdir,starname,tlimits=[-np.inf,np.inf],
                   debug=False):
    ff = FlatField(pixelfile,tlimits=tlimits)
    if debug:
        radii = range(4,5)
        moving = [1]
        weighted = [1]
    else:
        radii = range(2,8)
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

        print ff
        groupname = weights_groupname(ff_par)
        h5filename = os.path.join(outdir,'%s_weights.h5' % starname)
        ff.to_hdf(h5filename,groupname)

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
    p.add_argument('--debug',action='store_true')

    args  = p.parse_args()
    tlimits=[args.tmin,args.tmax]

    flatfield_wrap(
        args.pixelfile,args.outdir,args.starname,tlimits=tlimits,
        debug=args.debug)

