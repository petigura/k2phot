from astropy.io import fits
from pixel_decorrelation import get_wcs,loadPixelFile
import numpy as np
from numpy import ma
from matplotlib import mlab
def centroid(flux):
    """
    Centroid
    
    Parameters
    ----------
    flux : flux cube (already should be masked and background subtracted)
    
    Returns
    -------
    centcol : Centroid of along column axis. 0 corresponds to origin
    centrow : Centroid of along row axis. 0 corresponds to origin
    """
    nframe,nrow,ncol = flux.shape

    irow = np.arange(nrow)
    icol = np.arange(ncol)

    # Compute row centriod
    fluxrow = np.sum(flux, axis=2) # colflux.shape = (nframe,nrow)
    centrow = np.sum( (fluxrow*irow), axis=1) / np.sum(fluxrow,axis=1)

    # Compute column centriod
    fluxcol = np.sum(flux,axis=1) # colflux.shape = (nframe,ncol)
    centcol = np.sum( (fluxcol*icol), axis=1) / np.sum(fluxcol,axis=1)
    return centcol,centrow

def fits_to_chip_centroid(fitsfile):
    """
    Grab centroids from fits file

    Parameters
    ----------
    fitsfile : path to pixel file

    Returns
    -------
    centx : centroid in the x (column) axis
    centy : centroid in the y (row) axis
    """
    apsize = 7

    hdu0,hdu1,hdu2 = fits.open(fitsfile)
    cube = hdu1.data
    flux = cube['FLUX']
    t = cube['TIME']
    cad = cube['CADENCENO']

    nframe,nrow,ncol = flux.shape

    # Define rectangular aperture
    wcs = get_wcs(fitsfile)
    ra,dec = hdu0.header['RA_OBJ'],hdu0.header['DEC_OBJ']
    x,y = wcs.wcs_world2pix(ra,dec,0)
    scentx,scenty = np.round([x,y]) 
    nrings = (apsize-1)/2

    x0 = scentx - nrings
    x1 = scentx + nrings
    y0 = scenty - nrings
    y1 = scenty + nrings
    mask = np.zeros((nrow,ncol))
    mask[y0:y1+1,x0:x1+1] = 1 # 1 means use in aperture

    # Compute background flux
    # mask = True aperture, don't use to compute bg
    flux_sky = flux.copy()
    flux_sky_mask = np.zeros(flux.shape)
    flux_sky_mask += mask[np.newaxis,:,:].astype(bool)
    flux_sky = ma.masked_array(flux_sky, flux_sky_mask)
    fbg = ma.median(flux_sky.reshape(flux.shape[0],-1),axis=1)

    # Subtract off background
    flux = flux - fbg[:,np.newaxis,np.newaxis]
    flux = ma.masked_invalid(flux)
    flux.fill_value = 0 
    flux = flux.filled()

    # Compute aperture photometry
    fsap = flux * mask
    fsap = np.sum(fsap.reshape(fsap.shape[0],-1),axis=1)

    # Compute centroids
    centx,centy = centroid(flux * mask)

    # table column physical WCS ax 1 ref value       
    # hdu1.header['1CRV4P'] corresponds to column of flux[:,0,0]
    # starting counting at 1. 
    centx += hdu1.header['1CRV4P'] - 1
    centy += hdu1.header['2CRV4P'] - 1

    r = np.rec.fromarrays(
        [t,cad,centx,centy,fsap,fbg],
        names='t,cad,centx,centy,fsap,fbg'
        )

    r = mlab.rec_append_fields(r,'starname',hdu0.header['KEPLERID'])
    return r

def fit_6_par_trans(x1,y1,x2,y2):
    """
    Fit Six Parameter Transformation

     Here is a FORTRAN routine that takes a list of stellar
     positions in two different frames and generates a 
     least-squares linear transformation between them.

     It doesn't do any rejection of stars at all, so it's
     important to make sure that the stars all have "consistent"
     positions.  This means you should look at the residuals
     and make sure that none of the stars transform badly.
     In other words, you want to make sure that input x2 position 
     and the transformed x2 position (based on the x1,y1 position) 
     are not in disagreement. 

     One way to do this is to do a transformation with all
     the stars, then look at all the residuals.  Reject the
     stars that have large residuals and regenerate the
     transformations.  You may have to do this several times
     to get reasonable residuals.  Of course you do not want to
     reject *too* many stars.  Often the residuals will have
     some dependence on magnitude; the bright stars can be better
     measured than the faint ones.  

     I do have versions of this that iteratively reject 
     inconsisent stars.

     All of these considerations should help you choose 
     the best stars to use in the transformations.

     Again, don't hesitate to ask if you have any questions...

       Jay


     PS: The inverse transformations are:

          x1 = AA*(x2-x2o) + BB*(y2-y2o) + x1o
          y1 = CC*(x2-x2o) + DD*(y2-y2o) + y1o

          where:  

           AA =  D/(A*D-B*C)
           BB = -B/(A*D-B*C)
           CC = -C/(A*D-B*C)
           DD =  A/(A*D-B*C)


    --------------------------------------------------------------

     this routine does a least squares solution (with no
     data rejection or weighting) for the 6-param linear
     fit for: 


        x2 = A*(x1-x1o) + B*(y1-y1o) + x2o  
        y2 = C*(x1-x1o) + D*(y1-y1o) + y2o  

     it may look like there are 8 params, but two of the
     offsets are arbitrary and are just set to the centroid
     of the distribution.

    
    """

    # Compute centroids of the points
    x1o = np.mean(x1)
    x2o = np.mean(x2)
    y1o = np.mean(y1)
    y2o = np.mean(y2)

    # compute displacements of points to center.
    dx1 = x1 - x1o
    dx2 = x2 - x2o
    dy1 = y1 - y1o
    dy2 = y2 - y2o

    # Solve for the best-fit linear parameters
    a = np.vstack([dx1,dy1]).T
    b = dx2
    x,resid,rank,s = np.linalg.lstsq(a,b)
    A,B = x

    a = np.vstack([dx1,dy1]).T
    b = dy2
    x,resid,rank,s = np.linalg.lstsq(a,b)
    C,D = x

    # Transformation matrix
    TM = np.array([[A,B],
                   [C,D]])

    return TM,x1o,y1o,x2o,y2o

def fit_6_par_trans_iter(x1,y1,x2,y2,verbose=True):
    nuse0 = len(x1)
    iuse = np.arange(nuse0)
    threshL = [5,3]

    for thresh in threshL:
        TM,x1o,y1o,x2o,y2o = fit_6_par_trans(
            x1[iuse],y1[iuse],x2[iuse],y2[iuse]
            )
        
        x2pr,y2pr = ref_to_targ(x1[iuse],y1[iuse],TM,x1o,y1o,x2o,y2o)
        dist = np.sqrt( (x2pr -x2[iuse] )**2 + (y2pr-y2[iuse])**2 )
        sig = np.median(dist)*1.5
        iuse = np.where(dist < (sig * thresh))[0]
        if verbose:
            print "nout=%i, nin=%i, sig=%.1f millipix" % \
                (nuse0,len(iuse),1e3*sig)
    
    TM,x1o,y1o,x2o,y2o = fit_6_par_trans(
        x1[iuse],y1[iuse],x2[iuse],y2[iuse]
        )

    return TM,x1o,y1o,x2o,y2o



def ref_to_targ(x1,y1,TM,x1o,y1o,x2o,y2o):
    """
    Reference frame to taget frame
    """
    dx1,dy1 = x1-x1o,y1-y1o
    dxy1 = np.vstack([dx1,dy1])
    xy2o = np.vstack([x2o,y2o])


    xy2pr = np.dot(TM,dxy1) + xy2o # Perform the transformation
    x2pr = xy2pr[0] # predicted x value, given transformation
    y2pr = xy2pr[1] # predicted y value, given transformation
    return x2pr,y2pr

def targ_to_ref(x2,y2,TM,x1o,y1o,x2o,y2o):
    """
    Reference frame to taget frame
    """
    dx2 = x2-x2o
    dy2 = y2-y2o
    dxy2 = np.vstack([dx2,dy2])
    xy1o = np.vstack([x1o,y1o])

    TMinv = np.linalg.inv(TM)

    xy1pr = np.dot(TMinv,dxy2) + xy1o # Perform the inverse transformation
    x1pr = xy1pr[0] # predicted x value
    y1pr = xy1pr[1] # predicted y value
    return x1pr,y1pr

def linear_transform(x,y,irf):
    """Linear transformation

    Parameters
    ----------
    x : `x` (column) position of points 
    y : `y` (row) position of points 
    irf : index of the frame to use as reference

    Returns
    -------
    trans : a record array describing the transformation. Has the following keys
            - A,B,C,D: Values in the rotation/scaling matrix
            - x1o,y1o: (unweighted) centroid of points in reference frame 
            - x2o,y2o: (unweighted) centroid of points in target frame 

    pnts : record array describing what happens to the points
            - x,y : original points
            - xpr,ypr : x,y points transformed into target frame given
              the best fit transformation
    """

    assert x.shape==y.shape,"x and y must have the same dimensions"
    nstar,ncad = x.shape

    # Define matrix describing the linear transformation
    transkeys = 'A,B,C,D,x1o,y1o,x2o,y2o,centxpr,centypr'
    ntranskeys = len(transkeys.split(','))
    trans = np.rec.fromarrays([np.zeros(ncad)]*ntranskeys,names=transkeys)

    # Define matrix describing the points
    pntskeys = 'x,y,xpr,ypr'
    npntskeys = len(pntskeys.split(','))
    pnts = np.rec.fromarrays([np.zeros((nstar,ncad))]*npntskeys,names=pntskeys)
    
    x1 = x[:,irf]  # column centroids of stars in reference frame
    y1 = y[:,irf]  # row centroids of stars in reference frame

    # itr is index of the target frame
    for itr in range(ncad):
        x2 = x[:,itr] # column centroids of stars in target frame
        y2 = y[:,itr] # row centroids of stars in target frame
        rp = pnts[:,itr]
        rt = trans[itr]

        if (~np.isnan(x2)).sum()!=0:
            TM,x1o,y1o,x2o,y2o = fit_6_par_trans(x1,y1,x2,y2)
            
            # Shove the transformation parameters into array
            rt['A'] = TM[0,0]
            rt['B'] = TM[0,1]
            rt['C'] = TM[1,0]
            rt['D'] = TM[1,1]
            rt['x1o'] = x1o
            rt['y1o'] = y1o
            rt['x2o'] = x2o
            rt['y2o'] = y2o

            # Shove the transformed points into record array 
            x2pr,y2pr = ref_to_targ(x1,y1,TM,x1o,y1o,x2o,y2o)
            rp['x'] = x2        
            rp['y'] = y2
            rp['xpr'] = x2pr
            rp['ypr'] = y2pr

        if itr%1000==0:
            print itr

    return trans,pnts

