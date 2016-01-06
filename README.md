# k2phot #

Routines for extracting lightcurves from K2 images

Current Version: v0.5

## Dependencies ##

Tested with Python v2.7.3, 2.7.6, 2.7.8

### Public Python modules ###
```
- matplotlib / pylab (tested with v1.1.0, 1.3.1)
- NumPy (tested with v1.6.2, 1.8.1)
- SciPy (tested with 0.7.0, 0.10.1, 0.14.0)
- AstroPy (for io.fits; tested with v0.4, 0.4.1)
- Pandas (v0.14.1)
- skimage (0.10.0)
```

### Private Python modules ###

```
- k2utils
- terra
```

## Installation Instructions ##

1. Make sure necessary dependencies are installed
1. `git clone git@github.com:petigura/k2phot.git`
1. `git clone git@github.com:petigura/k2utils.git`
1. `git clone git@github.com:petigura/terra.git`

1. In order for the code to run, the following environment variables need to be set

  ```
  K2_DIR $projdir/K2/ # Working directory for K2 project
  K2PHOT_DIR $home/code_carver/k2phot/ # Where the k2phot code lives
  K2PHOTFILES $projdir/www/k2photfiles/ # Path to ancilary k2files
  K2_ARCHIVE $projdir/www/K2/ # Where all pixel, and output files are 
  K2_TERRA_DIR $home/code_carver/terra/ # Path to the terra code
  ```

  and the following modifications need to be made to the PATH and PYTHONPATH
  
  ```
  prepend-path PYTHONPATH $K2PHOT_DIR/k2phot/
  prepend-path PATH $K2PHOT_DIR/code/bin/
  prepend-path PATH $K2_TERRA_DIR/k2_webapp/
  prepend-path PATH $K2_TERRA_DIR/bin/
  ```
  
  At NERSC, I manage these environment variables with the modules package so I can simultaniously keep a development and a deployed version




1. Run the following script `k2utils/code/bin/get_k2photfiles.sh` to download K2 catalogs (~100 MB) from Erik's NERSC website...
1. To properly identify target stars based on WCS data, the
  environment variable K2_DIR must be set. Target Catalogs must be in
  ${K2_DIR}/catalogs/ .  Example catalogues are found at
  http://archive.stsci.edu/missions/k2/catalogs/
1. The environment variable K2PHOT_DIR must also be set.  Target
  catalogues must be in $K2PHOT_DIR/target_lists/ -- e.g., the file
  http://keplerscience.arc.nasa.gov/K2/docs/Campaigns/C0/K2Campaign0targets.csv



## Tests ##

# scrape_fits_headers #

A small test to verify that the `scrape_fits_headers` code is working

DBFILE=${K2PHOTFILES}/test_C1_headers.db
scrape_fits_headers ktwo201222515-c01_lpd-targ.fits ktwo201223635-c01_lpd-targ.fits ktwo201226808-c01_lpd-targ.fits ktwo201227422-c01_lpd-targ.fits ktwo201228585-c01_lpd-targ.fits ktwo201230523-c01_lpd-targ.fits ktwo201233437-c01_lpd-targ.fits ktwo201233864-c01_lpd-targ.fits ktwo201234630-c01_lpd-targ.fits ktwo201234711-c01_lpd-targ.fits ktwo201235266-c01_lpd-targ.fits $DBFILE
sqlite3 $DBFILE "select * from headers"

# test_channel_transform # 

 python -c "from k2phot.tests.test_channel_transform import *; test_channel_transform()"

7. Test that everything works with the following command

  ```
  pixdecor.sh -d -c C1 -r C1_02-03 -s 201367065 -t ${K2PHOTFILES}/pixeltrans_C1_ch04.h5 
  ```

## Examples ##

### Running v0.5 on K2 C1 ###

Use the `pixdecor.sh` function to run the complete photometric pipeline and transit search.

  ```
  pixdecor.sh -c C1 -r C1_02-03 -s 201367065 -t ${K2PHOTFILES}/pixeltrans_C1_ch04.h5 
  ```

flags

- `-d` run in debug mode (shorter versions of all the code)
- `-c` which K2 campaign are we working with? 
- `-s` what is the starname?
- `-t` with pixel-transformation file do we use?

Using the web interface:

The deployed version of the code runs here:

`https://portal-auth.nersc.gov/kp2/vetting/C1/C1_01-26/201338508`

The development version runs here (note http as opposed to https)

`http://portal-auth.nersc.gov:25001/vetting/C1/C1_01-26/201338508`



## Change Log ##

### v0.3 -> v0.4 ###

Changes to current pipeline
- New method of solving for orientation of spacecraft based on many
  stars. Implemented in the channel_centroids.py and centroid.py modules

- 2D Gaussian process-based detrending with iterative identification
  of outliers provided a major advance in noise reduction. This is
  implemented in pixel_decorrelation4.py.

### v0.2 -> v0.3 ###

Changes to current pipeline:
- flatfield.py solves for flatfield
- pixel_decorrelation2.py decorrelates against 2D position and time

New code (still experimental):
- diffimage.py (code to compute difference images)
- pointing.py (code to incorporate Tom B. pointing info)


### v0.1 -> v0.2 ###
- Location of the star is acquired by using the WCS headers to achieve an initial guess. Then we generate a synthetic image and register that image to the reference image to determine the correction to the WCS solution
- Diagnostic plots show the positions of the stars according to the epic catalogs
- dx,dy between frames is computed using subpixel registration

## Old Examples ##
Examples from previous runs of the code. Calls may be deprecated. Checkout the older versions to recover.


### Running v0.3 on K2 C0 ###

```
python flatfield.py --tmin 1940 --tmax 1972 <path to pixel .fits file> -d <output directory> <sqlite3 database file>
python pixel_decorrelation2.py --h5file <path to flatfield .ff.h5 file> <output directory>
```


### Running v0.2 on K2 C0 ###
```
. /global/homes/p/petigura/k2_setup.sh
python pixel_decorrelation.py -f <path to pixel .fits file> --wcs=1 -r 4 --minrad=2 --maxrad=8 --verbose=1 --gausscen=0 --plotmode=gs --tmin=1940 --output=pobj,fits --xymeth=xcorr2D -s <output directory>
```
Notes:
- `--xymeth=xcorr2D` keyword forces use of 2D image registration
- `--output=pobj,fits` dumps both fits and pickle versions of the photometry obhect
- `--wcs=1` Determines aperture center by using coordinates (after refinement)


### Running v0.1 on K2 C0 ###

```
python pixel_decorrelation.py -f <path to pixel .fits file> --wcs=1 -r 4 --minrad=2 --maxrad=8 --verbose=1 --gausscen=0 --plotmode=gs --tmin=1940 --output=2
```
