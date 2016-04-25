"""
K2 Catalog

Module for reading K2 catalogs and target lists.

Target catalogues must be in $K2PHOT_DIR/target_lists/ -- e.g.,
  the file 'K2Campaign0targets.csv'

Target Catalogs must be in ${K2_DIR}/catalogs/

Example catalogues are found at
  http://archive.stsci.edu/missions/k2/catalogs/

"""

import os
from cStringIO import StringIO as sio
import sqlite3

import pandas as pd
from astropy import units as u
from astropy.coordinates import Longitude,Latitude
from astropy.io import fits
import h5py
import re

import numpy as np
from ..config import K2PHOTFILES,K2PHOT_DIR

k2cat_sqlfile = os.path.join(K2PHOTFILES,'catalogs/k2_catalogs.sqlite')
k2cat_h5file = os.path.join(K2PHOTFILES,'catalogs/k2_catalogs.h5')
MAST_CATALOGS = os.path.join(K2PHOTFILES,'mast_catalogs/')
TARGET_LISTS = os.path.join(K2PHOTFILES,'target_lists/')

def read_mast_cat(k2_camp,debug=False):
    """
    Read catalogs using the formats specified in the MAST
    """
    targets = read_target_list(k2_camp)
    targets['target'] = True        
    cat = read_epic(k2_camp, debug=debug)
    cat = pd.merge(cat, targets, how='left', on='epic')
    cat['target'] = cat.target==True
    cat.index = cat.epic
    return cat

def read_target_list(k2_camp):
    """
    Read Target Lists

    http://keplerscience.arc.nasa.gov/K2/GuestInvestigationsC03.shtml

    Reads the csv files with the target lists from each campagin
    """

    targetsfn = 'K2Campaign%itargets.csv' % int(k2_camp[1:])
    targetsfn = os.path.join(TARGET_LISTS,targetsfn)

    if re.compile('C0|C1|C3|C4|C5|C6|C7|C8|C9|C10|C11|C12|C13').match(k2_camp):
        targets = pd.read_csv(targetsfn,usecols=[0])
        targets = targets.rename(columns={'EPIC ID':'epic'})
    elif re.compile('C2').match(k2_camp):
        targets = pd.read_csv(targetsfn,usecols=[0],names=['epic'])
    else:
        assert False,"read_target_list: include k2_campaign"


    return targets

s = """
k2camp readmefn catalogfn
C0 README_d14108_01_epic_c0_dmc d14108_01_epic_c0_dmc.mrg.gz
C1 README_epic_field1_dmc d1435_02_epic_field1_dmc.mrg.gz
C2 README_d1497_01_epic_c23_dmc d1497_01_epic_c23_dmc.mrg.gz
C3 README_d1497_01_epic_c23_dmc d1497_01_epic_c23_dmc.mrg.gz
C4 README_d14184_01_epic_c245_dmc d14184_01_epic_c245_dmc.mrg.gz
C5 README_d14184_01_epic_c245_dmc d14184_01_epic_c245_dmc.mrg.gz
C6 README_d14260_01_epic_c6_dmc d14260_01_epic_c6_dmc.mrg.gz
C7 README_d14260_03_epic_c7_dmc d14260_03_epic_c7_dmc.mrg.gz
C8 README_d15042_02_epic_c8_dmc d15042_02_epic_c8_dmc.mrg.gz
C9 README_d15042_04_epic_c9_dmc d15042_04_epic_c9_dmc.mrg.gz
C10 README_d15076_02_epic_c10_dmc d15076_02_epic_c10_dmc.mrg.gz
C11 README_d15332_01_epic_c11_dmc d15332_01_epic_c11_dmc.mrg.gz
C12 README_d15332_02_epic_c12_dmc d15332_02_epic_c12_dmc.mrg.gz
C13 README_d15332_03_epic_c13_dmc d15332_03_epic_c13_dmc.mrg.gz
"""
filenames = pd.read_table(sio(s),sep='\s',index_col=0, engine='python')


class Reader(object):
    def __init__(self,k2_camp):
        assert k2_camp in filenames.index, \
            "readmefn and/or catalogfn not defined for %s " % k2_camp
        readmefn = filenames.ix[k2_camp,'readmefn']
        readmefn = os.path.join( MAST_CATALOGS , readmefn )
        catalogfn = filenames.ix[k2_camp,'catalogfn']
        catalogfn = os.path.join(MAST_CATALOGS,catalogfn)

        # Read in the column descriptors
        readme = pd.read_table(readmefn ,header=None, names=['line'])
        readme = readme[readme.line.str.contains('^#\d{1}')==True]
        readme['col'] = readme.line.apply(
            lambda x : x.split()[0][1:]).astype(int)
        readme['name'] = readme.line.apply(lambda x : x.split()[1])

        self.readme = readme
        self.catalogfn = catalogfn
        self.readmefn = readmefn

def query_epic(k2_camp,epic):
    """
    Read all the epic columns from a specific file.
    """
    reader = Reader(k2_camp)
    kw = dict(compression='gzip',sep='|',header=None,)
    print "reading in gzipped catalog, takes ~20s"
    df = pd.read_table(reader.catalogfn, usecols=[0], **kw)
    idx = np.where(df[0]==epic)[0][0]
    df = pd.read_table(reader.catalogfn, skiprows=idx,nrows=1, **kw)
    df = df.T
    df['col'] = np.arange(1,len(df)+1)
    df = pd.merge(df,reader.readme['col name'.split()],on='col')
    return df

def read_epic(k2_camp,debug=False):
    hdfpath = os.path.join(MAST_CATALOGS,k2_camp+'.hdf')
    if os.path.exists(hdfpath):
        cat = pd.read_hdf(hdfpath,k2_camp)
        return cat

    print "creating {}".format(hdfpath)
    reader = Reader(k2_camp)
    
    # List of columns to include
    # namemap = {'ID':'epic','RA':'ra','DEC':'dec','Kp':'kepmag'}
    namemap = {
        'ID':'epic','RA':'ra','DEC':'dec','Kp':'kepmag',
        'Jmag':'jmag','Hmag':'hmag','Kmag':'kmag'
    }    

    # Read in the actual calatog
    readme = reader.readme
    readme.index = readme.name
    cut = readme.ix[namemap.keys()]
    cut['newname'] = namemap.values()
    cut = cut.sort('col')
    usecols = cut.col-1

    print "reading gzipped catalog (may take some time)"
    if debug:
        nrows = 1000
    else:
        nrows = None

    cat = pd.read_table(
        reader.catalogfn, sep='|', names=cut.newname, header=None, 
        usecols=usecols, compression='gzip', nrows=nrows
        )
    if debug:
        return cat

    cat.to_hdf(hdfpath,k2_camp,complevel=1,complib='blosc')
    return cat

def read_epic_composite(k2_camp):
    k2_camp_int = int(k2_camp[1:])
    if k2_camp_int < 11:
        return read_epic(k2_camp)
    if k2_camp=='C11':
        catalogs = 'C2 C9 C11'.split()
    if k2_camp=='C12':
        catalogs = 'C3 C12'.split()
    if k2_camp=='C13':
        catalogs='C4 C13'.split()

    cat = []
    for catalog in catalogs:

        print "reading catalog {}".format(catalog)
        cat+=[read_epic(catalog)]
    cat = pd.concat(cat)
    return cat
        

def read_cat(k2_camp, **kwargs):
    """
    Read catalog
    
    Parameters
    ----------
    k2_camp : K2 Campaign (e.g. 'C1') or 'all'

    """
    
    reader = lambda x : read_cat_campaign(x, **kwargs)
    if k2_camp=='all':
        with h5py.File(k2cat_h5file,'r') as h5:
            k2_campaigns = h5.keys()
            k2_campaigns = [s for s in k2_campaigns if s[0]=='C']
        cat = map(reader, k2_campaigns)
        cat = pd.concat(cat)
    else:
        cat = reader(k2_camp)

    return cat 

def read_cat_campaign(k2_camp,return_targets=True):
    """
    Read catalog for a single campaign

    Reads in pandas DataFrame from pytables database.
    
    Parameters
    ----------
    k2_camp : K2 Campaign 
    """

    print "reading in catalog for %s from %s " % (k2_camp, k2cat_h5file)
    cat = pd.read_hdf(k2cat_h5file,k2_camp)
    cat['k2_camp'] = k2_camp
    if return_targets:
        cat = cat[cat.target]
    return cat


def read_diag(k2_camp,nbin=20):
    """
    Read Diagnostic Stars

    Query the catalog for nbin stars with different magnitude ranges

    Parameters
    ----------
    k2_camp : string with campaign name
    nbin : number of stars to return for a given magnitude range

    Returns
    -------
    dfdiag : DataFrame with subset of catalog used for diagnostics

    Usage
    -----
    
    """
    np.random.seed(0)
    cat = read_cat(k2_camp)

    dfdiag = []
    kepmagbin = range(6,16)
    for kepmag in kepmagbin:
        if kepmag < 8:
            binwidth = 0.5
        else:
            binwidth = 0.1

        cut = cat[np.abs(cat.kepmag - kepmag) < binwidth]
        ids = np.array(cut.index).copy()
        np.random.shuffle(ids)
        cut = cut.ix[ids[:nbin]]
        cut['kepmagbin'] = kepmag
        dfdiag+=[cut]

    dfdiag = pd.concat(dfdiag)
    return dfdiag

def make_k2_catalog(k2_camp):
    """
    Make Catalog

    Reads in EPIC catalog and target lists to create databases for
    quick access to K2 catalogs.

    Parameters
    ----------
    k2_camp : K2 Campaign 
    """

    if k2_camp=='all':
        df = read_cat(k2_camp)
        k2_camp = 'STAR'
    else:
        df = read_mast_cat(k2_camp)

    print "Dumping whole catalog to %s, %s" % (k2cat_h5file,k2_camp)
    print df.info()
    df.to_hdf(k2cat_h5file,k2_camp)

    print "Dumping convenience database to %s, %s" % (k2cat_sqlfile,k2_camp)
    con = sqlite3.connect(k2cat_sqlfile)
    df = df[df.target]
    df.to_sql(k2_camp,con,if_exists='replace',index=False)
    print df.info()

def read_diag_paper(k2_camp):
    """
    Read Diagnostic Stars

    For bins in the range of kepmag=[i,i+1] select 
    
    """
    np.random.seed(0)
    nbin = 100
    cat = read_cat(k2_camp)

    dfdiag = []
    kepmagbin = range(8,16)

    for kepmag in kepmagbin:
        cut = cat[cat.kepmag.between(kepmag,kepmag+1)]
        ids = np.array(cut.index).copy()
        np.random.shuffle(ids)
        if len(cut) > nbin:
            cut = cut.ix[ids[:nbin]]

        cut['kepmagbin'] = kepmag
        dfdiag+=[cut]

    dfdiag = pd.concat(dfdiag)
    return dfdiag


def makePixelFileURL(epic, cycle, mode='K2'):
    """Generate the URL for a particular target. 

    :INPUTS:
      epic : int
        Target ID (analagous to "KIC" for Kepler Prime)

      cycle : int
        Cycle/Field number (analogous to Kepler's 'quarters')

      mode : str
        For now, only works in K2 mode.
        """
    # 2014-10-03 07:43 IJMC: Created

    fmtstr = 'http://archive.stsci.edu/missions/k2/target_pixel_files/c%i/%i/%05i/ktwo%i-c%02i_lpd-targ.fits.gz' 
    return fmtstr % (cycle, 1e5*np.floor(epic/1e5), np.floor((epic - 1e5*np.floor(epic/1e5))/1e3)*1e3, epic, cycle)
