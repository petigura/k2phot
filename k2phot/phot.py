from astropy.io import fits
from config import bjd0
import pandas as pd
import numpy as np
from pdplus import LittleEndian
import os.path

_COLDEFS_LC_SHARED = [
    ["thrustermask","L","Thruster fire","bool"],
    ["roll","D","Roll angle","arcsec"],
    ["xpr","D","Column position of representative star","pixel"],
    ["ypr","D","Row position of representative star","pixel"],
    ["cad","J","Unique cadence number","int"],
    ["t","D","Time","BJD - %i" % bjd0],
]

# Column definitions that are specific to a given aperture
_COLDEFS_AP_LC = [
    ["fbg","D","Background flux","electrons per second per pixel"],
    ["bgmask","L","Outlier in background flux","bool"],
    ["fsap","D","Simple aperture photometry","electrons per second"],
    ["fmask","L","Global mask. Observation ignored","bool"],
    ["fdtmask","L",
     "Detrending mask. Observation ignored in detrending model","bool"],
    ["fdt_t_roll_2D","D","Residuals (fsap - ftnd_t_roll_2D)",
     "electrons per second"],
    ["fdt_t_rollmed","D","ftnd_t_rollmed + fdt_t_roll_2D",
     "electrons per second"],
]

# Column definitions that are specific to a given aperture
_COLDEFS_AP_NOISE = [
    ["name","20A","Noise statistic",""],
    ["value","D","Value","ppm"],
]

# Column definitions that are specific to a given aperture
_COLDEFS_VERTS = [
    ["x","D","column of aperture vertex","pixels"],
    ["y","D","row of aperture vertex","pixels"],
    ["ra","D","RA of aperture vertex","degrees"],
    ["dec","D","Dec of aperture vertex","degrees"],
]

# Keys to add to the primary header
_EXTRA_HEADER_KEYS = [
]
_VERSION = (1.0 ,'Version of the phot module')

class Photometry(object):
    """Photomery class

    Class that contains K2 photometry information. In addition to the
    light curve, it also contains information about the aperture as
    well as a median frame image for easy plotting

    :param medframe: Median image
    :type medframe: NxM array 

    :param lc: Light curve. Every cadence has a measurement
    :type lc: Pandas DataFrame
    
    :param ap_weights: Mask used to compute static aperture
    :type ap_weights: NxM array 

    :param ap_verts: Verticies of apertures (used for plotting)
    :type ap_verts: Pandas DataFrame

    :param noise: Noise statistics for given aperture
    :type noise: Pandas DataFrame

    :param pixfn: Path to pixel file. Needed in order to copy over
                   header information
    :type pixfn: str

    :param extra_header: Extra keywords to be passed into primary header
    :type extra_header: 
    """

    # Column definitions for the shared columns in a light curve object

    def __init__(self, medframe, lc, ap_weights, ap_verts, ap_noise, pixfn=None, 
                 extra_header=[]):
        self.medframe = medframe
        self.lc = lc
        self.ap_weights = ap_weights
        self.ap_verts = ap_verts
        self.ap_noise = ap_noise
        self.pixfn = pixfn 
        self.extra_header = extra_header
        self.header = fits.open(self.pixfn)[0].header

    def name_mag(self):
        """Return formatted name and magnitdue"""
        return "{OBJECT},KEPMAG={KEPMAG:.1f}".format(**self.header)

    def to_fits(self, fitsfn, group):
        """
        Package up photometry object as a fits file
        """
        # Construct primary header
        hduL_pixel = fits.open(self.pixfn)
        hdu_primary = hduL_pixel[0]
        hdu_primary.header['VERSION'] = _VERSION
        hdu_primary.header['EXTNAME'] = 'primary'
        hdu_primary.header['EXTEND'] = 'T'

        # Construct median frame HDU
        hdu_medframe = fits.ImageHDU(
            data=self.medframe,header=hduL_pixel[1].header
            )
        hdu_medframe.header['EXTNAME'] = 'medframe'
        
        # HDU that holds aperture weights 
        hdu_ap_weights = fits.ImageHDU(data=self.ap_weights)
        hdu_ap_weights.header['EXTNAME'] = extname( group, 'ap-weights')

        # HDU that holds aperture verticies
        hdu_ap_verts = _DataFrame_to_BinTableHDU(self.ap_verts, _COLDEFS_VERTS)
        hdu_ap_verts.header['EXTNAME'] = extname( group, 'ap-verts')

        # HDU that holds noise properties
        hdu_ap_noise = _DataFrame_to_BinTableHDU(self.ap_noise, _COLDEFS_AP_NOISE)
        hdu_ap_noise.header['EXTNAME'] = extname( group, 'ap-noise')

        # HDU that holds light curve info common to all apertures
        lc_shared_keys = [ _name for _name, _, _, _ in _COLDEFS_LC_SHARED]
        lc_shared = self.lc[lc_shared_keys]
        hdu_lc_shared = _DataFrame_to_BinTableHDU(self.lc, _COLDEFS_LC_SHARED)
        hdu_lc_shared.header['EXTNAME'] = 'lc-shared'

        # HDU that holds light curve info for specific apertures
        ap_lc_keys = [ _name for _name, _, _, _ in _COLDEFS_AP_LC]
        ap_lc = self.lc[ap_lc_keys]
        hdu_ap_lc = _DataFrame_to_BinTableHDU(
            self.lc, _COLDEFS_AP_LC
            )
        hdu_ap_lc.header['EXTNAME'] = extname( group, 'lc')

        hduL = [
            hdu_primary,
            hdu_medframe,
            hdu_ap_weights,
            hdu_ap_verts,
            hdu_lc_shared,
            hdu_ap_lc,
            hdu_ap_noise,
        ]
        write_hduL(fitsfn,hduL)
def write_hduL(fitsfn,hduL):
    # If file doesn't exist, add the primary header
    hdu_primary = hduL[0]
    if os.path.exists(fitsfn) is False:
        fits.append(fitsfn, hdu_primary.data, header=hdu_primary.header)

    hduL = hduL[1:]
    for hdu in hduL:
        data = hdu.data
        extname = hdu.header['EXTNAME']
        header = hdu.header
        try:
            fits.update(fitsfn, data, extname, header=header)
        except KeyError:
            fits.append(fitsfn, data, header=header)


def extname(group, suffix):
    return "{}_{}".format(group,suffix)

def hdu_to_DataFrame(hdu):
    """Convert BinTableHDU into DataFrame taking care of endian
    """
    return pd.DataFrame( LittleEndian( hdu.data) )    

def read_fits(*args):
    """Read in phot object from fits file

    :param fitsfn: Path to fits function
    :type fitsfn: string

    :param group: Group that's read in. Omitting argument lists available groups
    :type group: string
    """

    assert len(args) >=1,"read_fits(fitsfn, [group]) "
    if len(args)==1:
        fitsfn, = args
        hduL = fits.open(fitsfn)
        groups = [hdu.header['EXTNAME'] for hdu in hduL]
        groups = [n.split('_')[0] for n in groups if n.count('_ap-noise')>0]
        print "available groups"
        print groups
        return groups

    if len(args)==2:
        fitsfn, group = args
        hduL = fits.open(fitsfn)

    hdu_primary = hduL['primary']
    hdu_lc_shared = hduL['lc-shared']
    hdu_medframe = hduL['medframe']
    
    # Aperture-specific HDU
    hdu_ap_weights = hduL[extname( group, 'ap-weights')]
    hdu_ap_verts = hduL[extname( group, 'ap-verts')]
    hdu_ap_noise = hduL[extname( group, 'ap-noise')]
    hdu_ap_lc = hduL[extname( group, 'lc')]

    medframe = hdu_medframe.data
    ap_weights = hdu_ap_weights.data
    ap_verts = hdu_to_DataFrame( hdu_ap_verts )
    lc_shared = hdu_to_DataFrame( hdu_lc_shared )
    ap_lc = hdu_to_DataFrame( hdu_ap_lc )
    lc = pd.concat([lc_shared,ap_lc],axis=1)

    
    fmed = lc['fsap'].median()  
    lc['ftnd_t_roll_2D'] = lc['fsap'] - lc['fdt_t_roll_2D'] + fmed
    lc['ftnd_t_rollmed'] = lc['fsap'] - lc['fdt_t_rollmed'] + fmed

    ap_noise = hdu_to_DataFrame( hdu_ap_noise )

    phot = Photometry(medframe, lc, ap_weights, ap_verts, ap_noise,)
    phot.header = hdu_primary.header
    phot.header_medframe = hdu_medframe.header
    return phot

# Covenience functions to facilitate fits writing
def _DataFrame_to_Column(df,coldef):
    """
    Convert column from a Pandas DataFrame to Fits Column

    :param coldef: column definition list. Elements are 
                    [name, format, description, unit]
    :type coldef: list
    
    :param df: dataframe
    :type df: Pandas DataFrame

    """
    _name, _format, _description, _unit = coldef
    _array = np.array( df[_name] ) 
    column = fits.Column(
        array=_array, format=_format, name=_name, unit=_unit)
    return column

def _DataFrame_to_BinTableHDU(df, coldefs):
    """
    Covert DataFrame into fits Binary Table HDU

    :param df: DataFrame
    :type df: Pandas DataFrame
    
    :param coldefs: List of column definitions
    :type coldefs: List of 4 element lists
    """ 
    columns = []
    for coldef in coldefs:
        columns.append( _DataFrame_to_Column(df, coldef) )

    hdu = fits.BinTableHDU.from_columns(columns)
    for coldef in coldefs:
        _name, _format, _description, _unit = coldef
        hdu.header[_name] = _description

    return hdu

    
        
