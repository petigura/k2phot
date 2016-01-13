from scipy import ndimage as nd
import numpy as np
import matplotlib
from scipy.misc import imresize
from matplotlib import pylab as plt

class Aperture(object):
    """
    Class for denfining apertures
    
    :param ap_type: what type of aperture  `circular` or `region`
    :type ap_type: str

    :param ap_params: additional parameters for construction of aperture
    :type ap_params: dict
    """
    def __init__(self):
        self.verts = None
        self.weights = None

    def plot(self):
        plt.plot(self.verts[:,0],self.verts[:,1],color='LimeGreen')

def circular_aperture(im, locx, locy, radius):
    """
    :param locx: x (column) coordinate of center
    :type locx: float

    :param locy: y (row) coordinate of center
    :type locy: float

    :param radius: radius of circle (pixels)
    :type radius: float
    """
    verts = _circular_aperture_verts(locx, locy, radius)
    aper = Aperture()
    aper.verts = verts
    aper.weights = verts_to_weights(verts, im.shape, supersamp=10)
    return aper

def region_aperture(im, locx, locy, npix):
    order = connected_pixels_order(im, locx, locy)
    #zoom = 10
    weights = ((npix > order) & (order >=0 )).astype(float)
    weights_zoom = weights #nd.zoom(weights, zoom, order=0)
    CS = plt.contour(weights_zoom,[0.5],colors=['Green'], hold=False)

    aper = Aperture()
    aper.verts = CS.allsegs[0][0] #/ zoom
    aper.weights = weights
    return aper

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
