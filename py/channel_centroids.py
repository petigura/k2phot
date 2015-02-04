import h5plus
import h5py
from argparse import ArgumentParser
import centroid
import os
import sqlite3
import pandas as pd
import k2_catalogs
import numpy as np
from astropy.io import fits
from matplotlib import mlab
from matplotlib import pylab as plt
from scipy import ndimage as nd
K2_ARCHIVE = os.environ['K2_ARCHIVE']

def get_channel_headers(headerdb,k2_camp,channel):
    con = sqlite3.connect(headerdb)
    sql = 'select * from headers where channel=%i' % channel
    headers = pd.read_sql(sql,con)
    headers.index = headers.KEPLERID
    cat =  k2_catalogs.read_cat(k2_camp)
    headers = pd.merge(headers,cat,right_index=True,left_index=True)
    return headers

def channel_centroids(headerdb,k2_camp,channel,h5file,iref=None,
                      kepmaglim=[11,14]):
    df = get_channel_headers(headerdb,k2_camp,channel)
    df = df[df.kepmag.between(*kepmaglim)].sort('kepmag')
    df.index = range(len(df))
    nstars = len(df)

    def fits_to_chip_centroid(fitsfile):
        fitsfile = os.path.join(K2_ARCHIVE,'pixel/%s/' % k2_camp,fitsfile)
        r = centroid.fits_to_chip_centroid(fitsfile)
        return r 

    # Pull the first file to get length and data type
    fitsfile0 = df.iloc[0]['fitsfile']
    cent0 = fits_to_chip_centroid(fitsfile0)

    # Determine the refence frame
    if iref==None:
        dfcent0 = pd.DataFrame(LE(cent0))
        ncad = len(dfcent0)
        med = dfcent0.median()
        dfcent0['dist'] = (
            (dfcent0['centx'] - med['centx'])**2 +
            (dfcent0['centy'] - med['centy'])**2
            )
        dfcent0 = dfcent0.iloc[ncad/4:-ncad/4]
        dfcent0 = dfcent0.dropna(subset=['centx','centy'])
        iref = dfcent0['dist'].idxmin()
    
    print "using reference frame %i" % iref
    assert np.isnan(cent0['centx'][iref])==False,\
        "Must select a valid reference cadence. No nans"

    cent = np.zeros((nstars,cent0.shape[0]), cent0.dtype)
    for i,row in df.iterrows():
        if (i%10)==0:
            print i
        cent[i] = fits_to_chip_centroid(row['fitsfile'])

    trans,pnts = centroid.linear_transform(cent['centx'],cent['centy'],iref)
    keys = cent.dtype.names
    pnts = mlab.rec_append_fields(pnts,keys,[cent[k] for k in keys])

    if h5file!=None:
        with h5plus.File(h5file) as h5:
            h5['trans'] = trans
            h5['pnts'] = pnts


    trans,pnts = read_channel_centroids(h5file)
    plot_trans(trans)
    figpath = h5file[:-3] + '.png'
    plt.gcf().savefig(figpath)
    print "saving %s " % figpath
    return cent

def plot_trans(trans):
    """
    Diagnostic plot that shows different transformation parameters
    """
    keys = 'scale theta skew1 skew2'.split()
    nrows = 5
    fig,axL = plt.subplots(nrows=nrows,figsize=(20,8),sharex=True)
    fig.set_tight_layout(True)
    for i,key in enumerate(keys):
        plt.sca(axL[i])
        plt.plot(trans[key])
        plt.ylabel(key)
        
    plt.sca(axL[4])
    dtheta = np.array(trans['dtheta'])
    thrustermask = np.array(trans['thrustermask'])
    plt.plot(dtheta,'-',mew=0)
    plt.ylabel('$\Delta$ theta')
    plt.xlabel('Observation')
    i = np.arange(len(dtheta))
    plt.plot(i[thrustermask],dtheta[thrustermask],'.')
    
def get_thrustermask(dtheta):
    """
    Identify thruster fire events.

    Parameters 
    ----------
    dtheta : roll angle for contiguous observations

    Returns
    -------
    thrustermask : boolean array where True means thruster fire
    """
    

    medfiltwid = 10 # Filter width to identify discontinuities in theta
    sigfiltwid = 100 # Filter width to establish local scatter in dtheta
    thresh = 10 # Threshold to call something a thruster fire (units of sigma)

    medtheta = nd.median_filter(dtheta,medfiltwid)
    diffdtheta = np.abs(dtheta - medtheta)
    sigma = nd.median_filter(diffdtheta,sigfiltwid) * 1.5
    thrustermask = np.array(diffdtheta > thresh*sigma)
    return thrustermask

def trans_add_columns(trans):
    """
    Add useful columns derived from trans array
    """
    trans = pd.DataFrame(trans)
    trans['scale'] = np.sqrt(trans.eval('A*D-B*C'))-1 
    trans['scale'][abs(trans['scale']) > 0.1] = None
    trans['skew1'] = trans.eval('(A-D)/2')
    trans['skew2'] = trans.eval('(B+C)/2')
    trans['theta'] = trans.eval('(B-C)/(A+D)')
    obs = np.arange(len(trans))
    bi = np.array(np.isnan(trans.theta))
    trans.loc[bi,'theta'] = np.interp(obs[bi],obs[~bi],trans[~bi].theta)
    theta = np.array(trans['theta'])
    dtheta = theta[1:] - theta[:-1]
    dtheta = np.hstack([[0],dtheta])
    trans['dtheta'] = dtheta
    return trans

def LE(arr):
    names = arr.dtype.names
    arrL = []
    for n in names:
        if arr.dtype[n].byteorder==">":
            arrnew = arr[n].byteswap().newbyteorder('=')
        else:
            arrnew = arr[n]
        arrL.append(arrnew)
    arrL = np.rec.fromarrays(arrL,names=names)
    return arrL

def read_channel_centroids(h5file):
    with h5py.File(h5file) as h5:
        trans = h5['trans'][:] 
        pnts = h5['pnts'][:]

    trans = LE(trans)
    pnts = LE(pnts)

    trans = trans_add_columns(trans)
    trans['thrustermask'] = get_thrustermask(trans['dtheta'])
    return trans,pnts

if __name__=='__main__':
    p = ArgumentParser(description='Read in fitsfiles and compute centroids')
    p.add_argument('channel',type=int)
    p.add_argument('k2_camp',type=str)
    p.add_argument('headerdb',type=str)
    p.add_argument('h5file',type=str)
    p.add_argument('--kepmagmin',type=float,default=11)
    p.add_argument('--kepmagmax',type=float,default=14)

    args = p.parse_args()
    cent = channel_centroids(
        args.headerdb,args.k2_camp,args.channel,h5file=args.h5file,
        kepmaglim=[args.kepmagmin,args.kepmagmax]
        )
    print "wrote centroid info to %s" % args.h5file

