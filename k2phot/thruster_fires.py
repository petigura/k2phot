"""
Determine when the thruster fires are taking place.
"""


# Generate list of 12 mag stars
from k2_catalogs import read_diag
from pixel_io import loadPixelFile
import pandas as pd
from argparse import ArgumentParser
import flatfield
import os
from matplotlib.pylab import *
import pixel_decorrelation
import imagestack

def thruster_fires(pixfiles,csvfile,tlimits=None):
    """

    """
    basename = csvfile.split('.')[0]
    figpath = basename + '.png'
    df = pd.DataFrame(pixfiles,columns=['pixfile'])
    lcL = []

    # Load up images and compute displacements 
    for i in range(len(df)):
        ff = flatfield.FlatField(df.ix[i,'pixfile'],tlimits=tlimits)
        ff.subtract_background()
        ff.set_loc(mode='static')
        ff.set_apertures(4)

        ff.__class__ = imagestack.ImageStack
        im = ff
        frame = im.get_frame(0)
        dx,dy = pixel_decorrelation.subpix_reg_stack(im.flux)

        sap_flux = im.get_sap_flux()
        lc = pd.DataFrame(
            vstack([im.cad,dx,dy,sap_flux]).T,
            columns='cad dx dy sap_flux'.split() 
            )

        lcL +=[lc]

    # Calculate displacements for each cadence
    for lc in lcL:
        dx = np.array(lc['dx'])
        dy = np.array(lc['dy'])
        ddx = dx[1:] - dx[:-1]
        ddy = dy[1:] - dy[:-1]

        # this adding the means that at cadence i, ddx is dx[i] -
        # dx[i-1]. The candence coming after the fastest slope will be tagged.
        lc['dds'] = np.hstack([[0],np.sqrt(ddx**2 + ddy**2)])


    # Join all the different light curves together
    lc2 = lcL[0]
    for i in range(1,len(lcL)):
        lc2 = pd.merge(lc2,lcL[i],on='cad',how='outer',suffixes=['','_%i' % i])
    lc2 = lc2.sort('cad')

    # Determine which cadences to mask out
    dds = ma.masked_invalid(lc2[[c for c in lc2.columns if c.count('dds')>0]])
    dds_sum = ma.sum(dds,axis=1)
    fmask = dds_sum > 3*median(dds_sum) 

    cols = [c for c in lc2.columns if c.count('sap_flux')>0]
    sap_flux = ma.masked_invalid(lc2[cols])
    sap_flux = sap_flux / ma.median(sap_flux,axis=0)
    sap_flux = sap_flux+arange(sap_flux.shape[1])*1e-2

    cadmask = lc2[['cad']]
    cadmask['fmask'] = fmask
    cadmask.to_csv(csvfile)
    
    fig,axL = subplots(nrows=3,figsize=(20,12),sharex=True)

    sca(axL[0])
    plot(lc2['cad'],dds)
    ylabel('Image Motion Since Previous Frame [pixels]')
    title('Identification of Thruster Fires')

    sca(axL[1])
    plot(lc2['cad'],dds_sum)
    plot(lc2['cad'][fmask],dds_sum[fmask],'rx',label='Masked Cadences')
    ylabel('Sum of Previous Plot')
    legend()

    sca(axL[2])

    plot(lc2['cad'],sap_flux)
    plot(lc2['cad'][fmask],sap_flux[fmask],'rx',label='Masked Cadences')
    xlabel('Cadence Number')
    ylabel('Normalized Flux\n(Plus Offset)')
    fig.set_tight_layout(True)

    gcf().savefig(figpath)

if __name__=='__main__':
    p = ArgumentParser(
        description='Identify thurster fires from light curve ensemble'
        )
    p.add_argument('csvfile',type=str,help='output file')
    p.add_argument(
        'pixfiles',type=str,nargs='+',help='list of fits files to use'
         )    
    p.add_argument('--debug',action='store_true')
    args = p.parse_args()

    if args.debug:
        tlimits = [2050,np.inf]
    else:
        tlimits = None

    thruster_fires(args.pixfiles,args.csvfile,tlimits=tlimits)
