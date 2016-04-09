"""
Module for downloading dss images and displaying apertures
"""
from astropy.io import fits
import astropy
import k2phot.phot
import xml.etree.ElementTree as etree
import urllib2
from matplotlib import pylab as plt
import numpy as np
import sys
import textwrap

def read_fits_dss(phot, survey='poss1_red'):
    survey_to_idx = {
        'poss1_red':8,
        'poss2ukstu_red':10
        }
        

    # Construct IPAC finder chart URL
    header_pri = phot.header
    ra, dec = [header_pri[k] for k in 'RA_OBJ DEC_OBJ'.split()] 
    url = 'http://irsa.ipac.caltech.edu/applications/finderchart/servlet/api?locstr={:f}+{:+f}&survey=DSS'.format(
        ra,dec
        )
    print "Accessing IPAC finderchart XML from"
    print url
    

    
    # Parse XML to get fitsurl
    response = urllib2.urlopen(url)
    tree = etree.parse(response)
    finderchart = tree.getroot()
    finderchart.findall('fitsurl')

    idx = survey_to_idx[survey]
    surveyname = finderchart[1][idx][0].text
    band = finderchart[1][idx][1].text
    obsdate = finderchart[1][idx][2].text
    fitsurl = finderchart[1][idx][3].text

    print "Downloading FITS file from"
    print fitsurl
    hduL = fits.open(fitsurl, chache=True)
    return hduL, surveyname, band, obsdate

def set_frame_size(extent_arcsec, dss, wcs):
    """
    Determine Limits of the DSS images
    """
    pixel_scale = np.abs(np.diag(wcs.pixel_scale_matrix) * 60.0 * 60.0)
    npix = np.array(list(dss.shape))
    center_pix = 0.5 * npix # central pixel value
    offset = np.array([ -0.5, 0.5 ]) * extent_arcsec * pixel_scale
    limits = center_pix + offset.reshape(-1,1) # compute limits

    # set limits
    xl = limits[:,0]
    yl = limits[:,1]
    plt.xlim(*xl)
    plt.ylim(*yl)

def dss(phot, survey='poss1_red',subplot_args=()):
    """
    Grab DSS finder chart and overplot axis
    """

    # Create Figure
    fig = plt.gcf()
    if subplot_args==():
        subplot_args = (111,)

    try:
        # Download fitsurl
        hduL_dss,  surveyname, band, obsdate = read_fits_dss(phot, survey=survey)
        header_dss = hduL_dss[0].header
        wcs = astropy.wcs.find_all_wcs(header_dss)[0]
        print wcs.pixel_scale_matrix
        dss = hduL_dss[0].data
        ax = fig.add_subplot(*subplot_args, projection=wcs)
        tr_fk5 = ax.get_transform('fk5')
        overlay = ax.get_coords_overlay('fk5')
        overlay['ra'].set_ticks(color='white')
        overlay['dec'].set_ticks(color='white')
        overlay.grid(color='white', linestyle='solid', alpha=0.5)
        plt.plot(
            phot.ap_verts['ra'], phot.ap_verts['dec'], transform=tr_fk5, 
            color='LimeGreen', lw=2
            )
        im = plt.imshow(dss, cmap='gray')

        # Size of finder chart should be approx K2 size
        nx, ny = phot.medframe.shape
        npix_max = max(nx,ny)
        set_frame_size(npix_max * 4, dss, wcs)
        title = '{}, {}\n'.format(band,obsdate[:4])
        plt.title(title)
    except:
        ax = fig.add_subplot(*subplot_args)
        error = str(sys.exc_info())
        error = textwrap.fill(error, 50)
        ax.text(0,1,error,transform=ax.transAxes, va='top')

    plt.xlabel('RA')
    plt.ylabel('Dec')

