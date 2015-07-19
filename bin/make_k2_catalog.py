#!/usr/bin/env python

"""
Make Catalog

Reads in EPIC catalog and target lists to create databases for
quick access to K2 catalogs.

Parameters
----------
k2_camp : K2 Campaign 
"""

from k2phot.io_utils.k2_catalogs import make_k2_catalog

import sqlite3
from argparse import ArgumentParser
if __name__=='__main__':
    p = ArgumentParser(
        description='Parse K2 catalog and dump into sqlite3 and h5 databases'
        )
    p.add_argument('k2_camp',type=str,help='K2 Campaign. e.g. C1')
    args = p.parse_args()
    k2_camp = args.k2_camp
    make_k2_catalog(k2_camp)
