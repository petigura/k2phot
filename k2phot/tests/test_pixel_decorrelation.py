from ..pixel_decorrelation import pixel_decorrelation

import os

K2_ARCHIVE = os.environ['K2_ARCHIVE']
K2PHOTFILES = os.environ['K2PHOTFILES']
K2_DIR = os.environ['K2_DIR']

fitsfiles = """ktwo201222515-c01_lpd-targ.fits ktwo201223635-c01_lpd-targ.fits
ktwo201226808-c01_lpd-targ.fits ktwo201227422-c01_lpd-targ.fits
ktwo201228585-c01_lpd-targ.fits ktwo201230523-c01_lpd-targ.fits
ktwo201233437-c01_lpd-targ.fits ktwo201233864-c01_lpd-targ.fits
ktwo201234630-c01_lpd-targ.fits ktwo201234711-c01_lpd-targ.fits
ktwo201235266-c01_lpd-targ.fits""".split()

pixfile = 'ktwo201367065-c01_lpd-targ.fits'
pixfile = os.path.join(K2_ARCHIVE, 'pixel/C1/', pixfile)
lcfile = os.path.join(K2_DIR, 'test_201367065.fits')
transfile = os.path.join(K2_DIR, 'test_channel_trans_C1_ch04.h5')

print transfile

def test_pixel_decorrelation():
    pixel_decorrelation(pixfile, lcfile, transfile, debug=True)
    
