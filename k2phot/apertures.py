from scipy import ndimage as nd
import numpy as np
import matplotlib
from scipy.misc import imresize
from matplotlib import pylab as plt
import pandas as pd
import astropy.wcs

class Aperture(object):
    """
    Class for denfining apertures
    
    :param im: 2D image array. Used to generate aperture weights
    :type locx: N x M array 

    :param locx: x (column) coordinate of center
    :type locx: float

    :param locy: y (row) coordinate of center
    :type locy: float

    :param ap_type: type of aperture used
    :param type: str eithr `circular` or `region`

    :param npix: number of pixels used in aperture
    :param type: float
    """
    def __init__(self, *args, **kwargs):
        self.verts = None
        self.weights = None

        if len(args)==0:
            return None
        
        assert len(args)==6, "args = im, im_header, locx, locy, ap_type, npix"

        im, im_header, locx, locy, ap_type, npix = args
    
        if ap_type=='circular':
            radius = np.sqrt( npix / np.pi )
            _ap = circular_aperture(im, locx, locy, radius)
        elif ap_type=='region':
            radius = np.sqrt( npix / np.pi )
            _ap = region_aperture(im, locx, locy, npix)
        else:
            assert False, "ap_type must be circular or aperture"

        verts = _ap.verts
        verts = pd.DataFrame(verts, columns=['x','y']) 
        
        # Convert x, y to sky coordinates
        wcs = astropy.wcs.find_all_wcs(im_header,keysel=['binary'])[0]
        verts['ra'], verts['dec'] = wcs.all_pix2world(verts.x,verts.y,0)

        self.verts = verts
        self.weights = _ap.weights
        self.npix = npix
        self.ap_type = ap_type
        self.name = "{}-{:.1f}".format(ap_type, npix)

    def __repr__(self):
        outstring = "<Aperture type={} npix={:.1f}>".format(
            self.ap_type, self.npix )
        return outstring

    def plot(self):
        plt.plot(self.verts[:,0],self.verts[:,1],color='LimeGreen')

def circular_aperture(im, locx, locy, radius):
    verts = _circular_aperture_verts(locx, locy, radius)
    aper = Aperture()
    aper.verts = verts
    aper.weights = verts_to_weights(verts, im.shape, supersamp=10)
    return aper

def region_aperture(im, locx, locy, npix):
    order = connected_pixels_order(im, locx, locy)
    weights = ((npix > order) & (order >=0 )).astype(float)
    aper = Aperture()
    aper.verts = mask_to_verts(weights)
    aper.weights = weights
    return aper

def mask_to_verts(mask,supersamp=10):
    mask_ss = mask.repeat(supersamp,axis=0).repeat(supersamp,axis=1)
    x_ss = ( np.arange(mask_ss.shape[1]) + 0.5 ) / supersamp - 0.5
    y_ss = ( np.arange(mask_ss.shape[0]) + 0.5 ) / supersamp - 0.5
    x_ss,y_ss = np.meshgrid(x_ss, y_ss)
    cs = plt.contour(x_ss,y_ss,mask_ss,levels=[0.5],hold=False)
    verts = cs.allsegs[0][0]
    return verts

def verts_to_weights(verts, shape, supersamp=10):
    ss_shape = ( shape[0]*supersamp, shape[1]*supersamp )
    extentx = [-0.5 , shape[1] - 0.5]
    extenty = [-0.5 , shape[0] - 0.5]
    nsampx = shape[1]*supersamp
    nsampy = shape[0]*supersamp
    x, y = np.mgrid[ extentx[0] : extentx[1] - 1.0/supersamp : nsampx*1j,
                     extenty[0] : extenty[1] - 1.0/supersamp : nsampy*1j ]

    points = np.vstack([x.flatten(),y.flatten()]).T
    path = matplotlib.path.Path(verts)
    weights = path.contains_points( points )
    points_in_ap = points[weights]

    # Note the x,y are flipped because histogram2d expects rows to
    # be first argument
    bins = [np.arange(extenty[0],extenty[1]+1,1),
            np.arange(extentx[0],extentx[1]+1,1)]
    weights, _, _ = np.histogram2d(
        points_in_ap[:,1], points_in_ap[:,0], bins=bins
        )

    weights /= supersamp**2
    return weights

def _circular_aperture_verts(locx, locy, radius):
    theta = np.linspace(0,2*np.pi,100)
    vertx = radius * np.cos(theta)
    verty = radius * np.sin(theta)
    vertx += locx
    verty += locy
    verts = np.vstack([vertx,verty]).T
    return verts

def connected_pixels_order(im0, locx, locy):
    im = im0.copy()

    # Constants
    weights = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    weights = weights.astype(float)
    weights /= np.sum(weights)
    print locx, locy 
    locr, locc = np.round(locy), np.round(locx)

    # Mask is the aperture mask
    mask = np.zeros(im.shape).astype(float)
    mask[locr,locc] = 1

    # Order stores the order with which the pixel is added
    order = np.zeros(mask.shape) - 1 
    order[locr,locc] = 0

    i = 1
    mask_old = mask.copy()
    npix = im.size
    while i < npix:
        mask_new = nd.correlate(mask_old, weights)
        mask_new = (mask_new > 0).astype(int)
        mask_delta = mask_new - mask_old
        mask_delta[0] = 0 
        mask_delta[-1] = 0 
        mask_delta[:,0] = 0 
        mask_delta[:,-1] = 0 


        im_delta = im * mask_delta 
        idx = np.unravel_index(np.argmax(im_delta),mask.shape)
        order[idx] = i
        mask_old[idx] = 1
        i+=1

    return order
