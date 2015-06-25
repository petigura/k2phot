"""
Module containing code to compute the linear transformation
between sets of points
"""
import numpy as np

def fit_6_par_trans(x1,y1,x2,y2):
    """Fit Six Parameter Transformation



    this routine does a least squares solution (with no
    data rejection or weighting) for the 6-param linear
    fit for: 


        x2 = A*(x1-x1o) + B*(y1-y1o) + x2o  
        y2 = C*(x1-x1o) + D*(y1-y1o) + y2o  

    it may look like there are 8 params, but two of the
    offsets are arbitrary and are just set to the centroid
    of the distribution.


    It doesn't do any rejection of stars at all, so it's important to
    make sure that the stars all have "consistent" positions.  This
    means you should look at the residuals and make sure that none of
    the stars transform badly.  In other words, you want to make sure
    that input x2 position and the transformed x2 position (based on
    the x1,y1 position) are not in disagreement.


     PS: The inverse transformations are:

          x1 = AA*(x2-x2o) + BB*(y2-y2o) + x1o
          y1 = CC*(x2-x2o) + DD*(y2-y2o) + y1o

          where:  

           AA =  D/(A*D-B*C)
           BB = -B/(A*D-B*C)
           CC = -C/(A*D-B*C)
           DD =  A/(A*D-B*C)

    Notes
    -----
    From Jay Anderson
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
    Target frame to reference frame
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
    """
    Linear transformation

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
