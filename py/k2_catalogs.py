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
import pandas as pd
from astropy import units as u
from astropy.coordinates import Longitude,Latitude
from astropy.io import fits

import numpy as np
import sqlite3

K2PHOTFILES = os.environ['K2PHOTFILES']
K2PHOT_DIR = os.environ['K2PHOT_DIR']

k2cat_sqlfile = os.path.join(K2PHOTFILES,'catalogs/k2_catalogs.sqlite')
k2cat_h5file = os.path.join(K2PHOTFILES,'catalogs/k2_catalogs.h5')

MAST_CATALOGS = os.path.join(K2PHOT_DIR,'mast_catalogs/')


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
    if k2_camp=='C0':
        targetsfn = 'K2Campaign0targets.csv'
    elif k2_camp=='C1':
        targetsfn = 'K2Campaign1targets.csv'
    elif k2_camp=='C2':
        targetsfn = 'K2Campaign2targets.csv'

    targetsfn = os.path.join(K2PHOT_DIR,'target_lists/',targetsfn)

    if (k2_camp=='C0') or (k2_camp=='C1'):
        targets = pd.read_csv(targetsfn,usecols=[0])
        targets = targets.rename(columns={'EPIC ID':'epic'})
    elif (k2_camp=='C2'):
        targets = pd.read_csv(targetsfn,usecols=[0],names=['epic'])

    return targets

def read_epic(k2_camp,debug=False):
    if k2_camp=='C2':
        readmefn = 'README_d1497_01_epic_c23_dmc'
        catalogfn = 'd1497_01_epic_c23_dmc.mrg.gz' 
    if k2_camp=='C6':
        readmefn = "README_d14260_01_epic_c6_dmc"
        catalogfn = "d14260_01_epic_c6_dmc.mrg.gz"
    if k2_camp=='C7':
        readmefn = "README_d14260_03_epic_c7_dmc"
        catalogfn = "d14260_03_epic_c7_dmc.mrg.gz"

    readmefn = os.path.join(MAST_CATALOGS,readmefn)
    catalogfn = os.path.join(MAST_CATALOGS,catalogfn)

    # Read in the column descriptors
    readme = pd.read_table(readmefn ,header=None, names=['line'])
    readme = readme[readme.line.str.contains('^#\d{1}')==True]
    readme['col'] = readme.line.apply(
        lambda x : x.split()[0][1:]).astype(int)
    readme['name'] = readme.line.apply(lambda x : x.split()[1])

    # List of columns to include
    namemap = {'ID':'epic','RA':'ra','DEC':'dec','Kp':'kepmag'}

    # Read in the actual calatog
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
        catalogfn, sep='|', names=cut.newname, header=None, usecols=usecols,
        compression='gzip', nrows=nrows
        )

    return cat

    
def read_cat(k2_camp,return_targets=True):
    """
    Read catalog

    Reads in pandas DataFrame from pytables database.
    
    Parameters
    ----------
    k2_camp : K2 Campaign 
    """
    cat = pd.read_hdf(k2cat_h5file,k2_camp)
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
    kepmagbin = range(10,16)
    for kepmag in kepmagbin:
        cut = cat[cat.kepmag.between(kepmag-0.1,kepmag+0.1)]
        ids = np.array(cut.index).copy()
        np.random.shuffle(ids)
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
