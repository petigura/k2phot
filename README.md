# k2phot 

Extract photometry from K2 images

Current Version: v1.0

## Dependencies ##

Tested with Python v2.7.3, 2.7.6, 2.7.8

- NumPy (tested with 1.6.2, 1.8.1)
- SciPy (tested with 0.7.0, 0.10.1, 0.14.0)
- AstroPy (for io.fits; tested with 0.4, 0.4.1)
- pyfits (needed to run k2sc).
- Pandas (0.14.1)
- skimage (0.10.0)
- matplotlib / pylab (tested with 1.1.0, 1.3.1)
- photutils
- emcee
- george (requires eigen) complicated install, consult evernote

## Installation Instructions

## Tests

# scrape_fits_headers

A small test to verify that the `scrape_fits_headers` code is working

DBFILE=${K2PHOTFILES}/test_C1_headers.db
scrape_fits_headers ktwo201222515-c01_lpd-targ.fits ktwo201223635-c01_lpd-targ.fits ktwo201226808-c01_lpd-targ.fits ktwo201227422-c01_lpd-targ.fits ktwo201228585-c01_lpd-targ.fits ktwo201230523-c01_lpd-targ.fits ktwo201233437-c01_lpd-targ.fits ktwo201233864-c01_lpd-targ.fits ktwo201234630-c01_lpd-targ.fits ktwo201234711-c01_lpd-targ.fits ktwo201235266-c01_lpd-targ.fits $DBFILE
sqlite3 $DBFILE "select * from headers"

# test_channel_transform

 python -c "from k2phot.tests.test_channel_transform import *; test_channel_transform()"

7. Test that everything works with the following command

  ```
  pixdecor.sh -d -c C1 -r C1_02-03 -s 201367065 -t ${K2PHOTFILES}/pixeltrans_C1_ch04.h5 
  ```

## Examples

We make available two decorrelation algorithms `k2sc` and `pixdecor`

### k2sc

```python
import 
k2phot.pipeline_k2sc
k2phot.pipeline_k2sc(
    pixfile,lcfile,transfile,splits, debug=debug, tlimits=tlimits, tex=tex,
    plot_backend=plot_backend, aper_custom=aper_custom, xy=xy,
    transitParams=transitParams, transitArgs=transitArgs
)
```


### Pixdecor

```python
import 
k2phot.pipeline_k2sc
k2phot.pipeline_k2sc(
    pixfile,lcfile,transfile,splits, debug=debug, tlimits=tlimits, tex=tex,
    plot_backend=plot_backend, aper_custom=aper_custom, xy=xy,
    transitParams=transitParams, transitArgs=transitArgs
)
```


             


