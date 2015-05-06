"""
Pulls in relavent info from k2 pixel files and shoves it in to a
sqlite3 database
"""
import sys
import sqlite3
import glob
import os

from astropy.io import fits

K2PHOT_DIR = os.environ['K2PHOT_DIR']
SCHEMAFILE = os.path.join(K2PHOT_DIR,'sql/headers_schema.sql')

def scrape_header_to_db(fitsfile,dbfile):
    if not os.path.isfile(dbfile):
        with open(SCHEMAFILE) as f:
            schema = f.read()

        con = sqlite3.connect(dbfile)
        with con:
            cur = con.cursor()
            cur.execute(schema)            
        con.close() 

    with fits.open(fitsfile) as hduL:
        cols_h0 = 'KEPLERID OBJECT CHANNEL MODULE OUTPUT CAMPAIGN'.split() 
        cols_h1 = 'NAXIS NAXIS1 NAXIS2 TDIM4 1CRV4P 2CRV4P'.split()       
        outd_h0 = dict( [ (c,hduL[0].header[c]) for c in cols_h0 ] )

        outd_h1 = {}
        for c in cols_h1:
            if c.count('1CRV4P') > 0:
                k = 'REF_COL'
            elif c.count('2CRV4P') > 0:
                k = 'REF_ROW'
            else:
                k = c
            
            outd_h1[k] = hduL[1].header[c]

    outd = dict(outd_h0,**outd_h1)
    outd['fitsfile'] = os.path.basename(fitsfile)
    

    columns = ', '.join(outd.keys())
    placeholders = ':'+', :'.join(outd.keys())
    query = 'INSERT INTO headers (%s) VALUES (%s)' % (columns, placeholders)

    con = sqlite3.connect(dbfile)
    with con:
        cur = con.cursor()
        cur.execute(query,outd)            
    con.close() 

def scrape_headers_to_db(fitsfiles,dbfile):
    """
    Same as scrape_header_to_db but can handle multiple files
    """

    print "Scraping header info from the following files:" 
    print fitsfiles
    print "into"
    print dbfile

    for i,fitsfile in enumerate(fitsfiles):
        try:
            scrape_header_to_db(fitsfile,dbfile)
        except:
            print sys.exc_info()
            pass

        if i%10==0:
            print i

