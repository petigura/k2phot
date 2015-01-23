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

K2_ARCHIVE = os.environ['K2_ARCHIVE']

def get_channel_headers(headerdb,k2_camp,channel):
    con = sqlite3.connect(headerdb)
    sql = 'select * from headers where channel=%i' % channel
    headers = pd.read_sql(sql,con)
    headers.index = headers.KEPLERID
    cat =  k2_catalogs.read_cat(k2_camp)
    headers = pd.merge(headers,cat,right_index=True,left_index=True)
    return headers

def channel_centroids(headerdb,k2_camp,channel,iref,h5file=None,kepmaglim=[11,14]):
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

    return cent

def read_channel_centroids(h5file):
    with h5py.File(h5file) as h5:
        trans = h5['trans'][:] 
        pnts = h5['pnts'][:]
        return trans,pnts

if __name__=='__main__':
    p = ArgumentParser(description='Read in fitsfiles and compute centroids')
    p.add_argument('channel',type=int)
    p.add_argument('k2_camp',type=str)
    p.add_argument('headerdb',type=str)
    p.add_argument('iref',type=int)
    p.add_argument('h5file',type=str)
    p.add_argument('--kepmagmin',type=float,default=11)
    p.add_argument('--kepmagmax',type=float,default=14)

    args = p.parse_args()
    cent = channel_centroids(
        args.headerdb,args.k2_camp,args.channel,args.iref,
        h5file=args.h5file,kepmaglim=[args.kepmagmin,args.kepmagmax]
        )
    print "wrote centroid info to %s" % args.h5file

