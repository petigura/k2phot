"""
Canibalized the important bits of code from photutils
"""

import numpy as np
from photutils.geometry import circular_overlap_grid

def get_phot_extents(data, positions, extents):
    """
    Get the photometry extents and check if the apertures is fully out of data.
    Parameters
    ----------
    data : array_like
        The 2-d array on which to perform photometry.
    Returns
    -------
    extents : dict
        The ``extents`` dictionary contains 3 elements:
        * ``'ood_filter'``
            A boolean array with `True` elements where the aperture is
            falling out of the data region.
        * ``'pixel_extent'``
            x_min, x_max, y_min, y_max : Refined extent of apertures with
            data coverage.
        * ``'phot_extent'``
            x_pmin, x_pmax, y_pmin, y_pmax: Extent centered to the 0, 0
            positions as required by the `~photutils.geometry` functions.
    """

    # Check if an aperture is fully out of data
    ood_filter = np.logical_or(extents[:, 0] >= data.shape[1],
                               extents[:, 1] <= 0)
    np.logical_or(ood_filter, extents[:, 2] >= data.shape[0],
                  out=ood_filter)
    np.logical_or(ood_filter, extents[:, 3] <= 0, out=ood_filter)

    # TODO check whether it makes sense to have negative pixel
    # coordinate, one could imagine a stackes image where the reference
    # was a bit offset from some of the images? Or in those cases just
    # give Skycoord to the Aperture and it should deal with the
    # conversion for the actual case?
    x_min = np.maximum(extents[:, 0], 0)
    x_max = np.minimum(extents[:, 1], data.shape[1])
    y_min = np.maximum(extents[:, 2], 0)
    y_max = np.minimum(extents[:, 3], data.shape[0])

    x_pmin = x_min - positions[:, 0] - 0.5
    x_pmax = x_max - positions[:, 0] - 0.5
    y_pmin = y_min - positions[:, 1] - 0.5
    y_pmax = y_max - positions[:, 1] - 0.5

    # TODO: check whether any pixel is nan in data[y_min[i]:y_max[i],
    # x_min[i]:x_max[i])), if yes return something valid rather than nan

    pixel_extent = [x_min, x_max, y_min, y_max]
    phot_extent = [x_pmin, x_pmax, y_pmin, y_pmax]

    return ood_filter, pixel_extent, phot_extent

def do_circular_photometry(data, positions, radius, effective_gain,
                           method, subpixels, r_in=None):


    extents = np.zeros((len(positions), 4), dtype=int)
    extents[:,0] = positions[:,0] - radius + 0.5
    extents[:,1] = positions[:,0] + radius + 1.5
    extents[:,2] = positions[:,1] - radius + 0.5
    extents[:,3] = positions[:,1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions, extents)

    flux = np.zeros(len(positions), dtype=np.float)

    # TODO: flag these objects
    if np.sum(ood_filter):
        flux[ood_filter] = np.nan
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return (flux, )

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    if method == 'center':
        use_exact = 0
        subpixels = 1
    elif method == 'subpixel':
        use_exact = 0
    else:
        use_exact = 1
        subpixels = 1

    for i in range(len(flux)):
        if not np.isnan(flux[i]):
            fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                             y_pmin[i], y_pmax[i],
                                             x_max[i] - x_min[i],
                                             y_max[i] - y_min[i],
                                             radius, use_exact, subpixels)

            if r_in is not None:
                fraction -= circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                  y_pmin[i], y_pmax[i],
                                                  x_max[i] - x_min[i],
                                                  y_max[i] - y_min[i],
                                                  r_in, use_exact, subpixels)

            flux[i] = np.sum(data[y_min[i]:y_max[i],
                                  x_min[i]:x_max[i]] * fraction)

    return (flux, )

def circular_photometry_weights(data, positions, radius):
    """
    Return an array of weights corresponding to a circular aperture.

    Parameters 
    ----------
    data : Not used in computation. Just for shape info
    positions : (x,y) position of flux center recal x = column, y = rows
    """

    extents = np.zeros((len(positions), 4), dtype=int)
    extents[:,0] = positions[:,0] - radius + 0.5
    extents[:,1] = positions[:,0] + radius + 1.5
    extents[:,2] = positions[:,1] - radius + 0.5
    extents[:,3] = positions[:,1] + radius + 1.5

    ood_filter, extent, phot_extent = get_phot_extents(data, positions, extents)
    weights = np.zeros(data.shape)

    # TODO: flag these objects
    if np.sum(ood_filter):
        flux[ood_filter] = np.nan
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[ood_filter]),
                      AstropyUserWarning)
        if np.sum(ood_filter) == len(positions):
            return (flux, )

    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent

    use_exact = 1
    subpixels = 1

    fraction = circular_overlap_grid(
        x_pmin, x_pmax, y_pmin, y_pmax, x_max - x_min, y_max - y_min, radius, 
        use_exact, subpixels)

    yslice = slice(y_min[0],y_max[0])
    xslice = slice(x_min[0],x_max[0])
    weights[yslice,xslice] = fraction
    return weights

def test_sum():
    radius = 3
    data = np.ones((21,20))

    positions = np.array([10,10]).reshape(-1,2)
    weights = circular_photometry_weights(data, positions, radius)
    
    test1 = np.allclose(np.sum(weights), np.pi*radius**2)
    print "test 1" ,test1
 
    positions = np.array([-0.5,10]).reshape(-1,2)
    weights = circular_photometry_weights(data, positions, radius)
    
    test2 = np.allclose(np.sum(weights), np.pi*radius**2 / 2)
    print "test 2" ,test2
 
