#!/usr/bin/env bash 

OLDPWD=${PWD}
cd ${K2_ARCHIVE}/pixel/C1/
DBFILE=${K2PHOTFILES}/test_C1_headers.db
scrape_fits_headers ktwo201222515-c01_lpd-targ.fits ktwo201223635-c01_lpd-targ.fits ktwo201226808-c01_lpd-targ.fits ktwo201227422-c01_lpd-targ.fits ktwo201228585-c01_lpd-targ.fits ktwo201230523-c01_lpd-targ.fits ktwo201233437-c01_lpd-targ.fits ktwo201233864-c01_lpd-targ.fits ktwo201234630-c01_lpd-targ.fits ktwo201234711-c01_lpd-targ.fits ktwo201235266-c01_lpd-targ.fits ${DBFILE}

#sqlite3 ${DBFILE} "select * from headers"
python -c "from k2phot.tests.test_channel_transform import *; test_channel_transform()"
python -c "from k2phot.tests.test_pixel_decorrelation import *; test_pixel_decorrelation()"
cd ${OLDPWD}
