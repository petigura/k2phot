# Change Log

Probably irrelevant now

### Running v0.5 on K2 C1

Use the `pixdecor.sh` function to run the complete photometric pipeline and transit search.

  ```
  pixdecor.sh -c C1 -r C1_02-03 -s 201367065 -t ${K2PHOTFILES}/pixeltrans_C1_ch04.h5 
  ```

flags

- `-d` run in debug mode (shorter versions of all the code)
- `-c` which K2 campaign are we working with? 
- `-s` what is the starname?
- `-t` with pixel-transformation file do we use?


### v0.3 -> v0.4

Changes to current pipeline
- New method of solving for orientation of spacecraft based on many
  stars. Implemented in the channel_centroids.py and centroid.py modules

- 2D Gaussian process-based detrending with iterative identification
  of outliers provided a major advance in noise reduction. This is
  implemented in pixel_decorrelation4.py.

### v0.2 -> v0.3

Changes to current pipeline:
- flatfield.py solves for flatfield
- pixel_decorrelation2.py decorrelates against 2D position and time

New code (still experimental):
- diffimage.py (code to compute difference images)
- pointing.py (code to incorporate Tom B. pointing info)


### v0.1 -> v0.2
- Location of the star is acquired by using the WCS headers to achieve an initial guess. Then we generate a synthetic image and register that image to the reference image to determine the correction to the WCS solution
- Diagnostic plots show the positions of the stars according to the epic catalogs
- dx,dy between frames is computed using subpixel registration

## Old Examples
Examples from previous runs of the code. Calls may be deprecated. Checkout the older versions to recover.


### Running v0.3 on K2 C0

```
python flatfield.py --tmin 1940 --tmax 1972 <path to pixel .fits file> -d <output directory> <sqlite3 database file>
python pixel_decorrelation2.py --h5file <path to flatfield .ff.h5 file> <output directory>
```


### Running v0.2 on K2 C0
```
. /global/homes/p/petigura/k2_setup.sh
python pixel_decorrelation.py -f <path to pixel .fits file> --wcs=1 -r 4 --minrad=2 --maxrad=8 --verbose=1 --gausscen=0 --plotmode=gs --tmin=1940 --output=pobj,fits --xymeth=xcorr2D -s <output directory>
```
Notes:
- `--xymeth=xcorr2D` keyword forces use of 2D image registration
- `--output=pobj,fits` dumps both fits and pickle versions of the photometry obhect
- `--wcs=1` Determines aperture center by using coordinates (after refinement)


### Running v0.1 on K2 C0

```
python pixel_decorrelation.py -f <path to pixel .fits file> --wcs=1 -r 4 --minrad=2 --maxrad=8 --verbose=1 --gausscen=0 --plotmode=gs --tmin=1940 --output=2
```
