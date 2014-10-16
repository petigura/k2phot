"""
Routines that solve for the flat-field
"""
import numpy as np
from numpy import ma

import photutils
from matplotlib import pylab as plt
from pixel_decorrelation import imshow2
from scipy import optimize

class star(object):
    def __init__(self,flux,locrc,radius,pixels):
        """
        Parameters
        ----------
        flux : Background-subtracted flux cube
        locrc : location of flux in row,column coordinates
        radius : radius of aperture
        pixels : pixels to use in weighting scheme
        """

        npix = len(pixels)
        mask = np.zeros(flux[0].shape).astype(bool)
        for i in range(npix):
            r,c = pixels[i]
            mask[r,c] = True
        
        self.flux = flux
        self.locrc = locrc # Position of star in row/column.
                           # Center of star is at image[locrc]

        self.locxy = locrc[::-1] # Position of star in x,y
                               # plot(x,y)

        self.pixels = pixels
        self.npix = npix
        self.npts = flux.shape[0] # Number of flux measurements
        self.aperture = ('circular', radius)  

        self.mask = mask
        self.weights0 = np.ones(npix) 
        self.weights = self.weights0.copy()

        self.f_non_weighted = self.get_f_non_weighted()

        self.fpix = self.get_fpix()
        self.f_weighted0 = self.get_f_weighted(self.weights)
        self.f_weighted0_sum = np.sum(self.f_weighted0)
        self.f_total0 = self.get_f_total(self.weights)
        self.fluxerr = np.sqrt(self.f_total0)
        # Construct constraint list
        cons = []

        bounds = []
        for i in range(npix):
            bounds.append( (0.90,1.1))
        self.bounds = bounds

    def fcons_weigthted_sum(self,weights):
        return np.sum(self.get_f_weighted(weights)) - self.f_weighted0_sum
    
    def get_f_non_weighted(self):
        """
        Return photometry for pixels that are not re-weighted
        """
        f_non_weighted = np.zeros(self.npts)
        for i in range(self.npts):
            fluxtable, aux_dict = photutils.aperture_photometry(
                self.flux[i],self.locxy,self.aperture,mask=self.mask)
            f_non_weighted[i] = fluxtable[0][0]
        return f_non_weighted

    def plot_weighted(self):
        logflux = np.log10(self.flux[0])
        logflux = ma.masked_invalid(logflux)
        logflux.mask = logflux.mask | (logflux < 0)
        logflux.fill_value = 0
        logflux = logflux.filled()
        imshow2(logflux)
        for i in range(self.npix):
            r,c = self.pixels[i]
            plt.text(c,r,i,va='center',ha='center',color='Orange')
        ap = photutils.CircularAperture(self.locxy,r=self.aperture[1])
        ap.plot(color='Lime',lw=1.5,alpha=0.5)

    def get_fpix(self):
        """
        Return a npix by npts array with the flux from each pixel
        """
        fpix = np.zeros((self.npix,self.npts))
        for i in range(self.npix):
            r,c = self.pixels[i]
            fpix[i] = self.flux[:,r,c]

        return fpix

    def get_f_weighted(self,weights):
        return np.dot(self.fpix.T,weights)

    def get_f_total(self,weights):
        return self.get_f_weighted(weights) + self.f_non_weighted
    
    def fom(self,weights):
        """
        Return figure of merit given the current value of the weights
        """
        flux = self.get_f_weighted(weights)
        fom = np.median(np.abs(flux - np.mean(flux)))
        res = (flux - np.mean(flux))/self.fluxerr
        fom = np.sum(res**2)
        fom = np.std(flux)
        # print weights
        # print fom
        return fom

    def solve_weights(self):
        x0 = self.weights.copy()
        print "initial fom = %f" % self.fom(x0)
        res = optimize.fmin_slsqp(
            self.fom,x0,eqcons=[self.fcons_weigthted_sum],bounds=self.bounds,
            epsilon=0.01,iprint=1,iter=2000)

        print "final fom = %f" % self.fom(res)
        self.weights = res
        return res

    def get_flat(self):
        flat = np.ones(self.flux[0].shape)
        for i in range(self.npix):
            r,c = self.pixels[i]
            flat[r,c] = self.weights[i]
        return flat
        
    def get_flux(self):
        """
        Return flux cube with current values of weights
        """
        return self.flux / self.get_flat()[np.newaxis,:,:]

    def get_f_total2(self):
        """
        Like get f_total, but using using 
        """
        flux = self.get_flux()
        f_total2 = np.zeros(self.npts)
        for i in range(self.npts):
            fluxtable, aux_dict = photutils.aperture_photometry(
                flux[i],self.locxy,self.aperture)
            f_total2[i] = fluxtable[0][0]
        return f_total2

    def fom2(self,weights):
        """
        Return figure of merit given the current value of the weights
        """
        self.weights = weights
        flux = self.get_f_total2()
        fom = np.std(flux)
        print weights,fom
        return fom

    def solve_weights2(self):
        x0 = self.weights.copy()
        print "initial fom = %f" % self.fom(x0)
        res = optimize.fmin_slsqp(
            self.fom2,x0,eqcons=[self.fcons_weigthted_sum],bounds=self.bounds,
            epsilon=0.01,iprint=1,iter=2000)

        print "final fom = %f" % self.fom(res)
        self.weights = res
        return res
