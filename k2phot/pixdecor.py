import os
from cStringIO import StringIO as sio

import numpy as np
from numpy import ma
import pandas as pd
import george

from pdplus import LittleEndian
from imagestack import ImageStack, read_imagestack
from channel_transform import read_channel_transform
from config import bjd0, noisekey, noisename, rbg

from ses import total_precision_theory
from astropy.io import fits
import contextlib
from matplotlib import pylab as plt
from lightcurve import Lightcurve,Normalizer

os.system('echo "pixel_decorrelation modules loaded:" $(date) ')

class PixDecorBase(object):
    """
    Contains the io methods for PixDecor object
    """

    # Light curve table
    lightcurve_columns = [
        ["thrustermask","L","Thruster fire","bool"],
        ["roll","D","Roll angle","arcsec"],
        ["xpr","D","Column position of representative star","pixel"],
        ["ypr","D","Row position of representative star","pixel"],
        ["cad","J","Unique cadence number","int"],
        ["t","D","Time","BJD - %i" % bjd0],
        ["fbg","D","Background flux","electrons per second per pixel"],
        ["bgmask","L","Outlier in background flux","bool"],

        ["fsap","D","Simple aperture photometry","electrons per second"],
        ["fmask","L","Global mask. Observation ignored","bool"],

        ["fdtmask","L",
         "Detrending mask. Observation ignored in detrending model","bool"],

        ["fdt_t_roll_2D","D","Residuals (fsap - ftnd_t_roll_2D)",
         "electrons per second"],

        ["fdt_t_rollmed","D","ftnd_t_rollmed + fdt_t_roll_2D",
         "electrons per second"],
    ]
       
    # Noise table columns
    dfaper_columns = [
        ["r","D","Radius of aperture","pixels"],
        ["f_"+noisename,"D","Noise in raw photometry","ppm"],
        ["fdt_"+noisename,"D","Noise in detrended photometry","ppm"],
    ]

    extra_header_keys = [
        ('pixfn','target pixel file'),
        ('transfn','transformation file'),
    ]

    def name_mag(self):
        """Return formatted name and magnitdue"""
        return "EPIC-%s, KepMag=%.1f" % (self.starname,self.kepmag)

    def raw_corrected(self):
        dmin = dict(self.dfaper.iloc[0])
        dmin['noisename'] = noisename
        dmin['raw'] = dmin['f_'+noisename]
        dmin['cor'] = dmin['fdt_'+noisename]
        dmin['fac'] = dmin['raw'] / dmin['cor'] *100

        outstr = "Noise Level (%(noisename)s) : Raw=%(raw).1f (ppm), Corrected=%(cor).1f (ppm); Improvement = %(fac).1f %%" % dmin
        
        return outstr

    def to_fits(self,fitsfn):
        """
        Package up the light curve, SES information into a fits file
        """
        
        # Covenience functions to facilitate fits writing
        def fits_column(c,df):
            array = np.array(df[ c[0] ])
            column = fits.Column(array=array, format=c[1], name=c[0], unit=c[3])
            return column

        def BinTableHDU(df,coldefs):
            columns = [fits_column(col,df) for col in coldefs]
            hdu = fits.BinTableHDU.from_columns(columns)
            for c in coldefs:
                hdu.header[c[0]] = c[2]
            return hdu

        # Copy over primary HDU
        hduL_pixel = fits.open(self.pixfn)
        hdu0 = hduL_pixel[0]
        for key,description in self.extra_header_keys:
            hdu0.header[key] = ( getattr(self,key), description )

        hdu1 = BinTableHDU(self.lc,self.lightcurve_columns)
        hdu2 = BinTableHDU(self.dfaper,self.dfaper_columns)        
        hduL = fits.HDUList([hdu0,hdu1,hdu2])
        hduL.writeto(fitsfn,clobber=True)

def read_fits(lcfn):
    pixdcr = PixDecorBase()
    hduL = fits.open(lcfn)
    pixdcr.lcfn = lcfn

    for k in 'pixfn transfn kepmag'.split():
        setattr(pixdcr,k,hduL[0].header[k])

    pixdcr.basename = os.path.splitext(pixdcr.lcfn)[0]
    pixdcr.starname = os.path.basename(pixdcr.basename)

    pixdcr.dfaper = pd.DataFrame(hduL[2].data)
    lc = hduL[1].data
    lc = LittleEndian( lc )
    lc = pd.DataFrame( lc ) 

    for fdtkey in 'fdt_t_roll_2D fdt_t_rollmed'.split():
        ftndkey = fdtkey.replace('fdt','ftnd')
        lc[ftndkey] = lc['fsap'] - lc[fdtkey] + np.median(lc['fsap'])

    pixdcr.lc = lc
    return pixdcr

class PixDecor(PixDecorBase):
    """
    Pixel Decorrelation Object

    Facilitates the creation of K2 photometry and diagnostic plots
    """

    unnormkeys = [
        "f",
        "fdt_t_roll_2D",
        "ftnd_t_roll_2D",
        "fdt_t_rollmed",
        "ftnd_t_rollmed",
    ]

    def __init__(self, pixfn, lcfn, transfn, tlimits=[-np.inf,np.inf], 
                 tex=None):
        self.pixfn = pixfn
        self.lcfn = lcfn
        self.transfn = transfn
        self.kepmag = fits.open(pixfn)[0].header['KEPMAG']
        self.basename = os.path.splitext(lcfn)[0]
        self.starname = os.path.basename(self.basename)
        self.apertures = kepmag_to_apertures(self.kepmag)

        # Define skeleton light curve. This pandas DataFrame contains all
        # the columns that don't depend on which aperture is used.
        im,x,y = read_imagestack(pixfn,tlimits=tlimits,tex=tex)
        self.x = x
        self.y = y
        self.im = im 

    def set_lc0(self,r0):
        """
        Set Skeleton Lightcurve

        Parameters
        ----------
        r0 : radius of aperture used to create skeleton light curve
        """
        # Define skeleton light-curve
        # Include pixel transformation information 
        self.im.set_apertures(self.x,self.y,rbg)
        self.im.set_fbackground(rbg)
        self.r0 = r0

        lc = self.im.ts 
        trans,pnts = read_channel_transform(self.transfn)
        trans['roll'] = trans['theta'] * 2e5
        # Hack select a representative star to use for x-y position 
        sqdist = (
            (np.median(pnts['x'],axis=1) - 500)**2 +
            (np.median(pnts['y'],axis=1) - 500)**2 
            )
        pnts500 = pnts[np.argmin(sqdist)]
        pnts = pd.DataFrame(pnts[0]['cad'.split()])
        pnts['xpr'] = pnts500['xpr']
        pnts['ypr'] = pnts500['ypr']

        trans = pd.concat([trans,pnts],axis=1)
        lc['fsap'] = self.im.get_sap_flux()
        norm = Normalizer(lc['fsap'].median())
        lc['f'] = norm.norm(lc['fsap'])
        lc = pd.merge(trans,lc,on='cad')
        lc['fmask'] = lc['f'].isnull() | lc['thrustermask'] | lc['bgmask']
        lc['fdtmask'] = lc['fmask'].copy()
        self.lc0 = lc

    def set_hyperparameters(self):
        """
        Sets the hyperparameters using information from the skeleton
        lightcurve
        """

        tchunk = 10 # split lightcurve in to tchunk-day long segments

        self.sigma_n = white_noise_estimate(self.kepmag)
        nchunks = self.lc0['t'].ptp() / tchunk
        nchunks = int(nchunks)
        sigma = map(lambda x : np.std(x['f']), np.array_split(self.lc0,nchunks))
        self.sigma = np.median(sigma)
        self.length_t = 4
        self.length_roll = 10

    def reject_outliers(self):
        """
        Perform an initial run with a single aperture size
        grab the outliers from this run and use it in subsequent runs
        """
        # Part 1 
        # Standardize the data

        # Set the values of the GP hyper parameters
        lc = detrend_t_roll_2D_segments( 
            self.lc0, self.sigma, self.length_t, self.length_roll,self.sigma_n,
            reject_outliers=True, segment_length=20
            )
        self.fdtmask = lc['fdtmask'].copy() 

    def set_apertures(self,r):
        self.im.set_apertures(self.x,self.y,r)
        print "aperture size = %i" % r

    def scan_aperture_size(self):
        dfaper = []
        for r in self.apertures:
            detrend_dict = self.detrend_t_roll_2D(r)
            dfaper.append(detrend_dict)
        dfaper = pd.DataFrame(dfaper)
        dfaper = dfaper.sort('fdt_'+noisename)
        self.dmin = dfaper.iloc[0]
        self.lc = self.dmin['lc']

        # Drop off the lightcurve column
        self.dfaper = dfaper.drop(['lc'],axis=1)
        
    def detrend_t_roll_2D(self,r ):
        # Create new lightcurve from skeleton
        lc = self.lc0.copy()
        self.set_apertures(r)
        lc['fsap'] = self.im.get_sap_flux()
        norm = Normalizer(lc['fsap'].median()) 
        lc['f'] = norm.norm(lc['fsap'])        
        lc['fdtmask'] = self.fdtmask 
        lc = detrend_t_roll_2D( 
            lc, self.sigma, self.length_t, self.length_roll,self.sigma_n, 
            reject_outliers=False
            )

        # Cast as Lightcurve object
        lc = Lightcurve(lc)
        sesfdt = lc.get_ses(noisekey) 
        sesf = lc.get_ses('f')
        for k in self.unnormkeys:
            lc[k] = norm.unnorm(lc[k])

        detrend_dict = {
            'r': r, 
            'lc': lc.copy(), 
            'f_' + noisename : sesf.ix[noisename],
            'fdt_' + noisename : sesfdt.ix[noisename],
            } 

        print self.get_diagnostic_info(detrend_dict)
        return detrend_dict

    def get_diagnostic_info(self, d):
        sdisp = self.starname+" "
        sdisp += "r=%(r)i " % d

        for k in "f fdt".split():
            k = k+'_'+noisename
            sdisp += "%s=%.1f " % (k,d[k] )
        return sdisp
        
    @contextlib.contextmanager
    def FigureManager(self,suffix):
        """
        A small context manager that pops figures and resolves the output
        filepath
        """
        plt.figure() # Executes before code block
        yield # Now run the code block
        figpath = self.basename+suffix
        plt.savefig(figpath,dpi=160)
        plt.close('all')
        print "created %s " % figpath

def kepmag_to_apertures(kepmag):
    """
    Given the kepmag of given target star, what range of apertures do
    we want to search over?
    """
    if 0 <= kepmag < 10:
        apertures = range(3,8)
    elif 10 <= kepmag < 14:
        apertures = [1.0, 1.4, 2.0, 3, 4, 5, 6, 8 ]
    elif 14 <= kepmag < 25:
        apertures = [1.0, 1.4, 2.0, 3, 4]
    return apertures

def white_noise_estimate(kepmag):
    """
    Estimate White Noise
    
    The Gaussian Process noise model assumes that some of the variance
    is white. 

    """
    fac = 2 # Factor by which to inflate Poisson and read noise estimate
    noise_floor = 100e-6 # Do not allow noise estimate to fall below this amount
    
    # Estimate from Poisson and read noise.
    sigma_th =  total_precision_theory(kepmag,10)
    sigma_th *= fac
    sigma_th = max(noise_floor,sigma_th)
    return sigma_th

def detrend_t_roll_2D(lc, sigma, length_t, length_roll, sigma_n, 
                      reject_outliers=False,debug=False):
    """
    Detrend against time and roll angle. Hyperparameters are passed
    in as arguments. Option for iterative outlier rejection.

    Parameters
    ----------
    sigma : sets the scale of the GP variance
    length_t : length scale [days] of GP covariance
    length_roll : length scale [arcsec] of GP covariance
    sigma_n : amount of white noise
    reject_outliers : True, reject outliers using iterative sigma clipping

    Returns 
    -------
    """

    # Define constants
    Xkey = 't roll'.split() # name of dependent variable
    Ykey = 'f' # name of independent variable
    fdtkey = 'fdt_t_roll_2D' 
    ftndkey = 'ftnd_t_roll_2D' 
    outlier_threshold = [None,10,5,3]

    if reject_outliers:
        maxiter = len(outlier_threshold) - 1
    else:
        maxiter = 1

    print "sigma, length_t, length_roll, sigma_n"
    print sigma, length_t, length_roll, sigma_n

    iteration = 0
    while iteration < maxiter:
        if iteration==0:
            fdtmask = np.array(lc.fdtmask)
        else:
            # Clip outliers 
            fdt = lc[fdtkey]
            sig = np.median( np.abs( fdt ) ) * 1.5
            newfdtmask = np.abs( fdt / sig ) > outlier_threshold[iteration]
            lc.fdtmask = lc.fdtmask | newfdtmask
            
        print "iteration %i, %i/%i excluded from GP" % \
            (iteration,  lc.fdtmask.sum(), len(lc.fdtmask) )

        # suffix _gp means that it's used for the training
        # no suffix means it's used for the full run
        lc_gp = lc[~lc.fdtmask] 

        # Define the GP
        kernel = sigma**2 * george.kernels.ExpSquaredKernel(
            [length_t**2,length_roll**2],ndim=2
            ) 

        gp = george.GP(kernel)
        gp.compute(lc_gp[Xkey],sigma_n)

        # Detrend againts time and roll angle
        mu,cov = gp.predict(lc_gp[Ykey],lc[Xkey])
        lc[ftndkey] = mu
        lc[fdtkey] = lc[Ykey] - lc[ftndkey]

        # Also freeze out roll angle dependence
        medroll = np.median( lc['roll'] ) 
        X_t_rollmed = lc[Xkey].copy()
        X_t_rollmed['roll'] = medroll
        mu,cov = gp.predict(lc_gp[Ykey],X_t_rollmed)
        lc['ftnd_t_rollmed'] = mu
        lc['fdt_t_rollmed'] = lc[fdtkey] + mu
        iteration+=1

    if debug:
        lc_gp = lc[~lc.fdtmask] 
        from matplotlib.pylab import *
        ion()
        fig,axL = subplots(nrows=2,sharex=True)
        sca(axL[0])
        errorbar(lc_gp['t'],lc_gp[Ykey],yerr=sigma_n,fmt='o')
        plot(lc['t'],lc[ftndkey])
        sca(axL[1])
        fm = ma.masked_array(lc[fdtkey],lc['fmask'])
        plot(lc['t'],fm)
        fig = figure()
        plot(lc_gp['roll'],lc_gp['f'],'.')
        plot(lc_gp['roll'],lc_gp['ftnd_t_roll_2D'],'.')

        import pdb;pdb.set_trace()

    return lc

def detrend_t_roll_2D_segments(*args,**kwargs):
    """
    Simple wrapper around detrend_t_roll_2D

    Parameters
    ----------
    segment_length : approximate time for the segments [days]
    
    Returns
    -------
    lc : lightcurve after being stiched back together
    """
    lc = args[0]
    segment_length = kwargs['segment_length']
    kwargs.pop('segment_length')
    nchunks = lc['t'].ptp() / segment_length 
    nchunks = int(nchunks)
    nchunks = max(nchunks,1)
    if nchunks==1:
        args_segment = (lc,) + args[1:]
        return detrend_t_roll_2D(*args_segment,**kwargs)

    lc_segments = np.array_split(lc,nchunks)
    lc_out = []
    for i,lc in enumerate(lc_segments):
        args_segment = (lc,) + args[1:]
        lc_out+=[detrend_t_roll_2D(*args_segment,**kwargs)]

    lc_out = pd.concat(lc_out)
    return lc_out




