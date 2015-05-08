import os
from ..pixel_decorrelation import pixel_decorrelation
from ..config import K2_ARCHIVE, K2PHOTFILES, K2_DIR

pixfile = 'ktwo201367065-c01_lpd-targ.fits'
pixfile = os.path.join(K2_ARCHIVE, 'pixel/C1/', pixfile)
lcfile = os.path.join(K2_DIR, 'test_201367065.fits')
transfile = os.path.join(K2_DIR, 'test_channel_trans_C1_ch04.h5')

print transfile

def test_pixel_decorrelation():
    pixel_decorrelation(pixfile, lcfile, transfile, debug=True)
    
