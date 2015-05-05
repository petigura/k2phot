"""
Pulls in relavent info from SpecMatch h5 files and shoves them into a database.
"""

# If table doesn't exist yet, create it.
import sqlite3
import glob
from argparse import ArgumentParser
import os
from astropy.io import fits

K2PHOT_DIR=os.environ['K2PHOT_DIR']

def scrape_headers_to_db(fitsfile,dbfile):
    if not os.path.isfile(dbfile):
        schemafile = os.path.join(K2PHOT_DIR,'code/sql/headers_schema.sql')
        with open(schemafile) as f:
            schema = f.read()

        con = sqlite3.connect(args.dbfile)
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

if __name__=="__main__":
    p = ArgumentParser(description="Scrape info from fits headers")
    p.add_argument('fitsfile',type=str,nargs='+',help='files to process')
    p.add_argument('dbfile',type=str,help='sqlite3 database')

    args  = p.parse_args()
    
    for i,fitsfile in enumerate(args.fitsfile):
        try:
            scrape_headers_to_db(fitsfile,args.dbfile)
        except:
            pass

        if i%10==0:
            print i
