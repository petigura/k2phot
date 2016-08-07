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

    image = None
    for _image in tree.iterfind('.//image'):
        if _image.find('band').text==survey:
            image = _image

    surveyname  = image.find('surveyname').text
    band = image.find('band').text
    obsdate = image.find('obsdate').text
    fitsurl = image.find('fitsurl').text

    print "Downloading FITS file from"
    print fitsurl
    hduL = fits.open(fitsurl, chache=True)
    return hduL, surveyname, band, obsdate

def dss(phot, survey='poss1_red',subplot_args=()):
    """
    Grab DSS finder chart and overplot axis
    """

    # Create Figure
    fig = plt.gcf()
    if subplot_args==():
        subplot_args = (111,)

    try:
        # Download DSS image
        hduL_dss,  surveyname, band, obsdate = read_fits_dss(
            phot, survey=survey
        )
        header_dss = hduL_dss[0].header
        wcs_dss = astropy.wcs.find_all_wcs(header_dss)[0]
        dss = hduL_dss[0].data


        # Compute verticies of the medframe image used to set limits
        ny, nx = phot.medframe.shape # row is y, col is x
        wcs_medframe = astropy.wcs.find_all_wcs(
            phot.header_medframe, keysel=['binary']
        )
        wcs_medframe = wcs_medframe[0] 
        verts_medframe_pix = np.array(
            [[ 0, 0 ],
             [ nx, 0 ],
             [ nx , ny ],
             [ 0 , ny ],
             [ 0, 0 ]]
        )
        verts_medframe_pix = verts_medframe_pix - 0.5 
        verts_medframe_world = wcs_medframe.all_pix2world(
            verts_medframe_pix, 0
        )
        verts_dss_pix = wcs_dss.all_world2pix(verts_medframe_world, 0)
        mm = lambda x : (np.min(x), np.max(x))
        xl = mm(verts_dss_pix[:,0])
        yl = mm(verts_dss_pix[:,1])

        # Make the plot
        ax = fig.add_subplot(*subplot_args, projection=wcs_dss)
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
        plt.plot(
            verts_medframe_world[:,0], verts_medframe_world[:,1], 
            transform=tr_fk5, color='LightSalmon', lw=2
        )

        title = '{}, {}\n'.format(band,obsdate[:4])
        plt.setp(ax, xlim=xl, ylim=yl, title=title, xlabel='RA', ylabel='Dec')

    except:
        ax = fig.add_subplot(*subplot_args)
        error = str(sys.exc_info())
        error = textwrap.fill(error, 50)
        ax.text(0,1,error,transform=ax.transAxes, va='top')


