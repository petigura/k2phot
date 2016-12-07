"""
Module for downloading dss images and displaying apertures
"""
from astropy.io import fits
import astropy
import astropy.wcs
import k2phot.phot

import xml.etree.ElementTree as etree
from matplotlib import pylab as plt
import numpy as np
import sys
import textwrap
import os

FINDERCHARTDIR = os.path.join( os.environ['K2PHOTFILES'], 'finderchart') 

def read_fits_dss(phot, survey='poss1_red'):
    # Construct IPAC finder chart URL
    header_pri = phot.header
    ra, dec = [header_pri[k] for k in 'RA_OBJ DEC_OBJ'.split()] 
    starname = header_pri['KEPLERID']
    fitsfn = "{}_{}.fits" .format(starname, survey)
    fitsfn  = os.path.join(FINDERCHARTDIR,fitsfn)

    def download_dss():
        os.system('mkdir -p {}'.format(FINDERCHARTDIR))
        cmd = 'wget --output-document={} "http://irsa.ipac.caltech.edu/applications/finderchart/servlet/api?mode=getImage&RA={:f}&DEC={:+f}&subsetsize=5.0&thumbnail_size=medium&survey=DSS&dss_bands={}&type=fitsurl"'.format(fitsfn, ra, dec, survey)
        print "Accessing IPAC finderchart"
        print cmd
        os.system(cmd)

    if os.path.exists(fitsfn) is False:
        download_dss()

    hduL = fits.open(fitsfn)
    return hduL

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
        hduL_dss = read_fits_dss(phot, survey=survey)
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

        title = '{}, {}\n'.format(survey, header_dss['DATE-OBS'][:4])
        plt.setp(ax, xlim=xl, ylim=yl, title=title, xlabel='RA', ylabel='Dec')

    except:
        ax = fig.add_subplot(*subplot_args)
        error = str(sys.exc_info())
        error = textwrap.fill(error, 50)
        ax.text(0,1,error,transform=ax.transAxes, va='top')


