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
k2_dir = os.environ['K2_DIR']
from astropy import units as u
from astropy.coordinates import Longitude,Latitude
import numpy as np
import sqlite3

k2cat_sqlfile = '%(K2PHOTFILES)s/catalogs/k2_catalogs.sqlite' % os.environ
k2cat_h5file = '%(K2PHOTFILES)s/catalogs/k2_catalogs.h5' % os.environ

K2PHOT_DIR = os.environ['K2PHOT_DIR']
MAST_CATALOGS = os.path.join(K2PHOT_DIR,'mast_catalogs/')

def read_mast_cat(k2_camp):
    """
    Read catalogs using the formats specified in the MAST
    """

    if k2_camp=='Ceng':
        df = pd.read_csv('%s/catalogs/K2_E2_targets_lc.csv' % MAST_CATALOGS )
        df = df.dropna()
        namemap = dict([(c,c.strip()) for c in df.columns])
        df = df.rename(columns=namemap)
        df = df.rename(columns={
            '#EPIC':'epic','Kp':'kepmag','list':'prog'})
        df['prog'] = df.prog.str.slice(start=1)
        ra = Longitude(df.ra*u.deg,180*u.deg)
        ra.wrap_angle=180*u.deg
        df['ra'] = ra.deg

    else:
        if k2_camp=='C0':
            targetsfn = 'K2Campaign0targets.csv'
            readmefn = 'README_d14108_01_epic_c0_dmc'
            catalogfn = 'd14108_01_epic_c0_dmc.mrg' 
        elif k2_camp=='C1':
            targetsfn = 'K2Campaign1targets.csv'
            readmefn = 'README_epic_field1_dmc'
            catalogfn = 'd1435_02_epic_field1_dmc.mrg' 
        
        targetsfn = os.path.join(K2PHOT_DIR,'target_lists/',targetsfn)
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

        cat = pd.read_table(
            catalogfn, sep='|', names=cut.newname, header=None, usecols=usecols
            )

        targets = pd.read_csv(targetsfn,usecols=[0])
        targets = targets.rename(columns={'EPIC ID':'epic'})
        targets['target'] = True
        cat = pd.merge(cat, targets, how='left', on='epic')
        cat['target'] = cat.target==True
        cat.index = cat.epic

    return cat

def read_cat(k2_camp='C0',return_targets=True):
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
    cat = read_cat(k2_camp=k2_camp)

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
