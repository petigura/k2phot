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

import pandas as pd
from astropy.io import fits
from photutils import CircularAperture,aperture_photometry
from scipy import ndimage as nd

import h5plus
from pdplus import LittleEndian as LE
import photometry
from pixel_decorrelation import imshow2,get_star_pos,loadPixelFile,get_stars_pix,subpix_reg_stack
from skimage import measure
from matplotlib import pylab as plt 

from pixel_decorrelation2 import plot_detrend


class ImageStack(object):
    def __init__(self,fn,tlimits=None):
        pos_mode = 'wcs'
        xcen,ycen = get_star_pos(fn,mode=pos_mode)

        cube,headers = loadPixelFile(fn,tlimits=tlimits)
        frame0 = ma.masked_invalid(cube['flux'])
        frame0 = ma.median(frame0,axis=0)
        frame0.fill_value = 0
        frame0 = frame0.filled()
        catcut, shift = get_stars_pix(fn,frame0)
        epic = headers[0]['KEPLERID']
        xcen,ycen = catcut.ix[epic]['pix0 pix1'.split()]

        print "Using position mode %s, star is at pixels = [%.2f,%.2f]" % \
            (pos_mode,xcen,ycen)

        self.locxy = (xcen,ycen)
        self.locrc = (ycen,xcen) 
        self.flux = cube['FLUX'].astype(float)
        self.t = cube['TIME'].astype(float)
        self.cad = cube['CADENCENO'].astype(int)

        # Number of frames, rows, and columns
        self.nframe,self.nrow,self.ncol = self.flux.shape 
        self.npix = self.nrow*self.ncol

        # Subtract off background
        self.subtract_background()

        # Store pixel values as a list
        row,col = np.mgrid[:self.nrow,:self.ncol]
        flux = ma.masked_invalid(self.flux)
        df = pd.DataFrame(dict(row=row.flatten(),col=col.flatten()))
        df['flux0'] = flux[0].flatten()
        df['fluxsum'] = ma.sum(flux,axis=0).flatten()
        df['dist'] = np.sqrt( (df.row - self.locrc[0])**2 + 
                              (df.col - self.locrc[1])**2 )
        df = pd.DataFrame(LE(df.to_records()))
        self.df = df

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

class FlatField(ImageStack):
    """
    Stack of images that we can used to compute Flat Field
    """
    def __init__(self,fn,tlimits=None):
        """
        Parameters
        ----------
        flux : Background-subtracted flux cube
        locrc : location of flux in row,column coordinates
        radius : radius of aperture
        pixels : pixels to use in weighting scheme
        """
        self.weight_bounds = [0.9,1.1]
        self.fn = fn
        # Instantiate from ImageStack class
        super(FlatField,self).__init__(fn,tlimits=tlimits)

        # Bin up cadences used to compute robust fom
        self.dfcad = get_dfcad(self.cad)

        # Other default values for array 
        self.fmask = np.zeros(self.nframe).astype(bool)
        self.dx = np.zeros(self.nframe)
        self.dy = np.zeros(self.nframe)

    def set_dxdy_by_registration(self):
        assert hasattr(self,'weights')==False,\
            "Must determine dx/dy before setting weights"
        self.dx,self.dy = subpix_reg_stack(self.flux) 

    def set_ff_parameters(self, pixels, radius, fmask=None):
        """
        Set the pixels used to for flat-fielding

        Parameters
        ----------
        pixels : (nframe,2) array specifying the row and column values
                 of the pixels to include in the flat field
                 re-weighting
        radius : radius of the moving circulare aperture used to compute flux
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
        self.radius = radius
        self.fpix = self.get_fpix()
        if type(fmask)!=type(None):
            self.fmask = fmask 
        self._update_fixed()
        
    def _update_fixed(self):
        """
        Update the parameters that are fixed during optimization
        """
        self.f_non_weighted = self.get_f_non_weighted()
        self.f_total0 = self.get_f_total(self.weights)
        self.f_total0_sum = ma.sum(self.f_total0)

    def get_f_non_weighted(self):
        """
        Return photometry for pixels that are not re-weighted
        """
        f_non_weighted = np.zeros(self.nframe)
        for i in range(self.nframe):
            locxy = self.get_aper_locxy(i)
            frame = self.flux[i]
            apertures = CircularAperture([locxy], r=self.radius)
            fluxtable = aperture_photometry(frame, apertures, mask=self.mask)
            f_non_weighted[i] = fluxtable[0][0]
        return f_non_weighted

    def get_aper_locxy(self,i):
        """
        For frame i, return the xy location of aperture center
        Should just be a property of the frame object
        """
        locxy = np.copy(self.locxy)
        locxy[0]+=self.dx[i]
        locxy[1]+=self.dy[i]
        return locxy

    def get_fpix(self):
        """
        Return a npix by npts array with the flux from each pixel
        """
        fpix = np.zeros((self.nweights,self.nframe))
        for i in range(self.nweights):
            r,c = self.pixels[i]
            fpix[i] = self.flux[:,r,c]

        return fpix

    def get_f_weighted(self,weights):
        return np.dot(self.fpix.T,weights)

    def get_f_total(self,weights):
        f_total = self.get_f_weighted(weights) + self.f_non_weighted
        f_total = ma.masked_array(f_total,self.fmask)
        return f_total

    def figure_of_merit(self,weights,metric):
        fm = self.get_f_total(weights)
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
        fom = lambda weights : self.figure_of_merit(weights,metric)
        def f_total_constraint(weights):
            return self.get_f_total(weights).sum() - self.f_total0_sum

        x0 = self.weights.copy()
        fom_initial = fom(x0)
        res = optimize.fmin_slsqp( 
            fom, x0, eqcons=[f_total_constraint], bounds=self.bounds,
            epsilon=0.01, iprint=1,iter=200)

        fom_final = fom(res)
        self.weights = res

        print "initial fom = %f" % fom_initial
        print "final fom = %f" % fom_final
        print "initial/final fom = %f" % (fom_initial/fom_final)
        self.fom_final = fom_final
        self.fom_initial = fom_initial
        return res

    def get_weights_frame(self):
        """
        Return an image with the current value of the weights
        """

        weights_frame = np.ones((self.nrow,self.ncol))
        for i in range(self.nweights):
            r,c = self.pixels[i]
            weights_frame[r,c] = self.weights[i]
        return weights_frame

    def get_moments(self,i):
        """
        Return moments from frame:
        Future work: This should be a method on a FrameObject
        """
        upsamp = 4
        apcenter = self.get_aper_locxy(i)
        frame = ma.masked_invalid(self.flux[i])
        frame.fill_value = 0
        frame = frame.filled()

        apcenter*=upsamp
        imz = nd.zoom(frame,upsamp)

        nrowz,ncolz = imz.shape
        rowz,colz = np.mgrid[:nrowz,:ncolz]
        dist = np.sqrt((rowz - apcenter[0])**2 + 
                       (colz - apcenter[1])**2 )
        
        imz = ma.masked_array(imz,dist > self.radius * upsamp)
        imz.fill_value=0
        imz = imz.filled()

        # Compute the centroid
        m = measure.moments(imz)
        moments = pd.Series(dict(m10=m[0,1],m01=m[1,0]))
        moments /= m[0,0]

        # Compute central moments (second order)
        mu = measure.moments_central(imz,moments['m10'],moments['m01'])
        c_moments = pd.Series(
            dict(mupr20=mu[2,0], mupr02=mu[0,2], mupr11=mu[1,1]) )
        c_moments/=mu[0,0]
        moments = pd.concat([moments,c_moments])
        moments/=upsamp
        return moments

    def plot_frame(self,i):
        """
        Should be a method of frame object
        """
        aperlocxy = self.get_aper_locxy(i)
        print aperlocxy
        logflux = np.log10(self.flux[i])
        logflux = ma.masked_invalid(logflux)
        logflux.mask = logflux.mask | (logflux < 0)
        logflux.fill_value = 0
        logflux = logflux.filled()
        imshow2(logflux)
        for i in range(self.nweights):
            r,c = self.pixels[i]
            plt.text(c,r,i,va='center',ha='center',color='Orange')
        apertures = CircularAperture([aperlocxy], r=self.radius)
        apertures.plot(color='Lime',lw=1.5,alpha=0.5)

    ## IO ##

    def to_pickle(self,filename):
        with open(filename,'w') as file:
            pickle.dump(self,file)
        print "saveing to %s" % filename
    
    def to_hdf(self,filename):
        f_old = ff.get_f_total(ff.weights0)
        f_weighted = ff.get_f_total(ff.weights)
        lc = dict(f_old= f_old, f_weighted= f_weighted, fmask=self.fmask)
        lc = pd.DataFrame(lc)
        lc = np.array(lc.to_records(index=False))
        with h5plus.File(filename) as h5:
            h5['lc'] = lc
        print "saveing to %s" % filename

    def to_fits(self,filename):
        """
        Produce a fits file that can be fed into pixel_decorrelation.py. Yes,
        it's a really inefficient way of storing 10 numbers
        """
        hduL = fits.open(self.fn)
        weights_frame = self.get_weights_frame()
        hduL[1].data['FLUX'] = hduL[1].data['FLUX'] * weights_frame
        hduL.writeto(filename,clobber=True)
        print "saveing to %s" % filename

cadmaskfile = os.path.join(os.environ['K2PHOTFILES'],'C0_fmask.csv')
cadmask = pd.read_csv(cadmaskfile,index_col=0)

def dict2insert(d,table):
    names = str(tuple(d.keys())).replace("'","")
    strtup = "("+("?, "*len(d.keys()))[:-2]+")"
    sqlcmd = 'INSERT INTO %s %s VALUES %s' % (table,names,strtup)
    values = tuple(d.values())
    return sqlcmd,values

def to_sql(con,stard,lc):
    """
    Output important parameters to sql database
    """
    cur = con.cursor()
    sqlcmd,values = dict2insert(stard,'star')
    cur.execute(sqlcmd,values)
    for i in range(len(lc)):
        d = dict(lc.iloc[i])
        sqlcmd,values = dict2insert(d,'phot')
        cur.execute(sqlcmd,values)
    con.commit()
    con.close()

if __name__ == "__main__":
    np.set_printoptions(precision=4)

    p = ArgumentParser(description='Photometry By Flat-Fielding')
    p.add_argument('pixelfile',type=str)
    p.add_argument('outdir',type=str)
    p.add_argument('dbfile',type=str)
    p.add_argument('--tmin',type=float,default=-np.inf)
    p.add_argument('--tmax',type=float,default=np.inf)

    args  = p.parse_args()
    pixelfile = args.pixelfile
    outdir = args.outdir
    dbfile = args.dbfile
    tlimits=[args.tmin,args.tmax]

    imstack = ImageStack(pixelfile,tlimits=tlimits)
    cut = imstack.df.query('dist < 3')
    cut = cut.sort('fluxsum',ascending=False).iloc[:20]
    pixels = np.array(cut['row col'.split()])

    ff = FlatField(pixelfile,tlimits=tlimits)
    ff.set_dxdy_by_registration()
    fmask = np.array(cadmask.ix[ff.cad]['fmask'])

    ff.set_ff_parameters(pixels,3,fmask=fmask)
    method = 'bin-med-std'
    ff.solve_weights(method)

    moments = map(ff.get_moments,range(ff.nframe))
    lc = pd.DataFrame(moments)
    lc['f_unweighted'] = ff.get_f_total(ff.weights0)
    lc['f_weighted'] = ff.get_f_total(ff.weights)
    lc['fmask'] = ff.fmask.astype(int)
    lc['cad'] = ff.cad
    lc['t'] = ff.t
    lc['dx'] = ff.dx
    lc['dy'] = ff.dy

    starname = str(fits.open(ff.fn)[0].header['KEPLERID'])
    lc['starname'] = starname

    basename = starname
    basename = os.path.join(outdir,basename)

    stard = dict(pd.Series(ff.weights).describe()['count min max std'.split()])
    stard = dict([ ('weights_%s' % k,stard[k]) for k in stard.keys() ])
    stard['starname'] = starname
    fitsfn = basename+'.ff.%s.fits' % method
    ff.to_pickle(basename+'.ff.%s.pickle'  % method)
    ff.to_hdf(basename+'.ff.%s.h5'  % method)
    ff.to_fits(fitsfn)

    if not os.path.isfile(args.dbfile):
        for schemafile in ['phot_schema.sql','star_schema.sql']:
            schemafile = os.path.join(
                os.environ['K2PHOT_DIR'],'code/py',schemafile)
            with open(schemafile) as f:
                schema = f.read()
            con = sqlite3.connect(args.dbfile)
            cur = con.cursor()
            cur.execute(schema)
            con.commit()
            con.close()
        
    # Read in the output file and compute aperture photmetry using
    # different radii. There is a more elegant way to do this!
    ff2 = FlatField(fitsfn,tlimits=tlimits)
    pixels = [] # No pixels are reweighted here:
    for radius in range(2,8):
        ff2.set_ff_parameters(pixels,radius,fmask=lc['fmask'])
        lc['f_weighted%i' % radius] = ff2.f_total0

    con = sqlite3.connect(dbfile)
    to_sql(con,stard,lc)

    h5file = '%s.ff.h5' % basename
    lc.to_hdf(h5file,'lc')
    pd.Series(stard).to_hdf(h5file,'stard')
    
    # Plotting
    columns = 'f_unweighted f_unweighted f_weighted'.split()
    plot_detrend(lc,columns)
    plt.gcf().savefig(basename+'.ff.png')
    
    plt.figure()
    ff.plot_frame(0)
    plt.gcf().savefig(basename+'.ff-frame.png')


