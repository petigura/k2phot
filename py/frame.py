import numpy as np
from numpy import ma
from skimage import measure
from photutils import CircularAperture
import pandas as pd
from pixel_decorrelation import imshow2

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
