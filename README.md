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


## Example

Running the pipeline on C5 data

1. Download pixel data

1. Collect the fits file meta data
   ```bash
   $ cd $K2_ARCHIVE/pixel/
   $ scrape_fits_headers $(find C5 -name "*.fits") C5_headers.db
   ```

1. Select a output channel to use. Channels 4 and 33 are a good bet.

1. Make the transformation files
   ````
   $ cd C5 # Must be in this directory to run code
   $ make_channel_transform.py --help
   $ make_channel_transform.py 4 C5 ../C5_headers.db ${K2PHOTFILES}/pixeltrans_C5_ch04.h5
   ```

1. Inspect the transformation files
   ```python 
   from k2phot import channel_transform as ct
   trans,pnts =  ct.read_channel_transform(args.transfn)
   ct.plot_trans(trans,pnts)

   phot = k2phot.phot.read_fits(args.fitsfn,'optimum')
   k2phot.plotting.phot.lightcurve_segments(phot.lc)   
   ```

1. If statisfied, run the photometric pipeline with. There are two decorrelation algorithms to use, we recommend k2sc
   ```python 

   import 
   k2phot.pipeline_k2sc
   k2phot.pipeline_k2sc(
       pixfile,lcfile,transfile,splits, debug=debug, tlimits=tlimits, tex=tex,
       plot_backend=plot_backend, aper_custom=aper_custom, xy=xy,
       transitParams=transitParams, transitArgs=transitArgs
   )
   ```
