import numpy as np
from scipy import signal
import pylab as py
import sys

import analysis as an
import tools
import pixel_decorrelation
from astropy.io import fits as pyfits

def computeCentroids(data, edata, mask):
    """Compute centroid time series.

    :INPUTS:
      data : 3D NumPy array
        Input target pixel file, of shape (Nobs x Nrow x Ncol).
        This should *NOT* be background-subtracted!

      edata : 3D NumPy array
        Uncertainties on input pixel data.

      mask : 2D NumPy array
        Binary mask, True for pixels used, of shape (Nrow x Ncol).

    :EXAMPLE:
      ::


        import lightcurve_diagnostics as ld
        from astropy.io import fits as pyfits

        field = 0
        epic = 202126884
        _pix = '/Users/ianc/proj/transit/kepler/data/'
        _proc = '/Users/ianc/proj/transit/kepler/proc/'
        pixfn = _pix + 'ktwo%i-c%02i_lpd-targ.fits' % (epic, field)
        procfn = _proc + '%i.fits' % epic

        dat = pyfits.getdata(procfn)
        pickledat = tools.loadpickle(procfn.replace('.fits', '.pickle'))
        cube, headers = pixel_decorrelation.loadPixelFile(pixfn, tlimits=[dat.time.min() - 2454833 - 1e-6, np.inf])
        time, data, edata = cube['time'],cube['flux'],cube['flux_err']

        tt, per, t14 = 2456768.6558394236, 5.3156579762, 0.26
        inTransit = np.abs((time-tt + per/2.) % per - per/2) < (t14/2.)
        mask = pickledat.crudeApertureMask
        c1, c2, ec1, ec2 = ld.computeCentroids(data, edata, mask)
        ld.cloudPlot(c1, c2, dat['cleanFlux'], inTransit, e_cen1=ec1, e_cen2=ec2, time=time)

    :RETURNS:
      c_row, c_col, err_c_row, err_c_col

    :NOTES:
      Follow Eqs. 1 & 2 of Bryson et al. (2013).
        """

    # 2014-10-02 15:33 IJMC: Created

    nobs, ncol, nrow = data.shape
    row = np.arange(nrow)
    col = np.arange(ncol)
    ccol, rrow = np.meshgrid(row, col)

    # Eq. 1 -- compute centroids:
    denom = (mask * data).reshape(nobs, -1).sum(-1)
    rowtemp = (mask * data * rrow).reshape(nobs, -1).sum(-1)
    coltemp = (mask * data * ccol).reshape(nobs, -1).sum(-1)

    C_row = rowtemp / denom
    C_col = coltemp / denom

    # Eq. 2 -- compute uncertainties:
    e_C_row =np.sqrt(((mask*(edata*rrow)**2)).reshape(nobs, -1).sum(-1) / denom**2 + \
         (rowtemp / denom**2)**2 * (mask*edata**2).reshape(nobs, -1).sum(-1))
    e_C_col =np.sqrt(((mask*(edata*ccol)**2)).reshape(nobs, -1).sum(-1) / denom**2 + \
         (coltemp / denom**2)**2 * (mask*edata**2).reshape(nobs, -1).sum(-1))

    return C_row, C_col, e_C_row, e_C_col

def computeMeanCentroids(cen_detrend, ecen, transitMask):
    """Compute centroid means: global, in-transit, out-of-transit.

    Returns: 
      mean_cen, emean_cen, mean_cen_out, emean_cen_out, mean_cen_in, emean_cen_in, sdom_mean_cen_in
    """
    # 2014-10-03 17:25 IJMC: Created

    mean_cen, e_mean_cen = an.wmean((cen_detrend), 1./ecen**2, reterr=True)
    mean_cen_in, e_mean_cen_in = an.wmean((cen_detrend)[transitMask], 1./ecen[transitMask]**2, reterr=True)
    mean_cen_out, e_mean_cen_out = an.wmean((cen_detrend)[True-transitMask], 1./ecen[True-transitMask]**2, reterr=True)
    sdom_mean_cen_in = (cen_detrend)[transitMask].std() / np.sqrt((transitMask).sum())

    return  mean_cen, e_mean_cen, mean_cen_out, e_mean_cen_out, mean_cen_in, e_mean_cen_in, sdom_mean_cen_in

def cloudPlot(cen1, cen2, flux, transitMask, e_cen1=None, e_cen2=None, medFiltWid=47, time=None, cen1model=None, cen2model=None, unitLab='Pix', fontsize=16, grid=True, title='', fig=None, figpos=None):
    """Make a 'cloud plot' to visually check for blended binaries.
    
    :INPUTS:
      cen1, cen2 : 1D NumPy Arrays
        Centroid time series, e.g. computed via :func:`computeCentroids`

      flux : 1D NumPy Array
        Observed light curve, already detrended for centroid motion.

      transitMask : 1D NumPy Array
        Boolean mask: True in transit, False out of transit.

      e_cen1, e_cen2 : 1D NumPy Arrays
        Uncertainty on cen1, cen2

      medFiltWid : positive, ODD, int (or None)
        Size of sliding median filter to remove slow drifts, variations, etc.

      cen1model, cen2model : 1D NumPy Arrays
        Optional *models* of cen1 & cen2 inputs -- e.g., from
        :func:`pixel_decorrelation.getArcLengths` (with
        retmodXY=True). If not passed in, then we call that function
        to generate such a model,  in which case you must pass in
        'time', too.

      time : 1D NumPy Array
        Time at which 'cen1,' etc. were measured. You must pass this
        in *unless* you pass in 'cen1model' & 'cen2model'. 

      unitLab : string
        Units of cen1, cen2 for plotting (e.g., arcsec, mas, Pix, etc.)

    :EXAMPLE:
      ::

        import lightcurve_diagnostics as ld
        from astropy.io import fits as pyfits

        field = 0
        epic = 202126884
        _pix = '/Users/ianc/proj/transit/kepler/data/'
        _proc = '/Users/ianc/proj/transit/kepler/proc/'
        pixfn = _pix + 'ktwo%i-c%02i_lpd-targ.fits' % (epic, field)
        procfn = _proc + '%i.fits' % epic

        dat = pyfits.getdata(procfn)
        pickledat = tools.loadpickle(procfn.replace('.fits', '.pickle'))
        cube, headers = pixel_decorrelation.loadPixelFile(pixfn, tlimits=[dat.time.min() - 2454833 - 1e-6, np.inf])
        time, data, edata = cube['time'],cube['flux'],cube['flux_err']

        tt, per, t14 = 2456768.6558394236, 5.3156579762, 0.26
        inTransit = np.abs((time-tt + per/2.) % per - per/2) < (t14/2.)
        mask = pickledat.crudeApertureMask
        c1, c2, ec1, ec2 = ld.computeCentroids(data, edata, mask)
        ld.cloudPlot(c1, c2, dat['cleanFlux'], inTransit, e_cen1=ec1, e_cen2=ec2, time=time)
      

    :NOTES:
      For K2 this is a bit more complicated than for Kepler Prime's
      cloud plots, because our pointing is so much worse.  My fix is
      to use models of the cen1/cen2 motions to remove the worst
      effects of spacecraft pointing.

    :TO_DO:
      Think about replacing :func:`pixel_decorrelation.getArcLengths`
      with a real 2D map, instead of Vanderburg & Johnson's crude (but
      often sufficient) assumption of 1D-motion.
    """
    # 2014-10-02 15:36 IJMC: Created




    if cen1model is None or cen2model is None:
        mostlyJunk = pixel_decorrelation.getArcLengths(time, [(cen1, cen2)], retmodXY=True)
        cen1model, cen2model = mostlyJunk[-1][0]
    cen1corrected = cen1 - cen1model
    cen2corrected = cen2 - cen2model

    if medFiltWid is None:
        medFiltWid = 1

    filtFlux = signal.medfilt(flux, medFiltWid)
    filtcen1 = signal.medfilt(cen1corrected, medFiltWid)
    filtcen2 = signal.medfilt(cen2corrected, medFiltWid)
    meanIn  = (flux / filtFlux)[transitMask].mean()
    meanOut = (flux / filtFlux)[True-transitMask].mean()
    
    mean_cen1, e_mean_cen1, mean_cen1_out, e_mean_cen1_out, mean_cen1_in, e_mean_cen1_in, sdom_mean_cen1_in = \
        computeMeanCentroids(cen1corrected - filtcen1, e_cen1, transitMask)
    mean_cen2, e_mean_cen2, mean_cen2_out, e_mean_cen2_out, mean_cen2_in, e_mean_cen2_in, sdom_mean_cen2_in = \
        computeMeanCentroids(cen2corrected - filtcen2, e_cen2, transitMask)


    # Compute statistics:
    val1a, eval1a = (mean_cen1_in - mean_cen1 ) , e_mean_cen1_in
    val1b, eval1b = (mean_cen1_in - mean_cen1 ) , sdom_mean_cen1_in
    val2a, eval2a = (mean_cen2_in - mean_cen2 ) , e_mean_cen2_in
    val2b, eval2b = (mean_cen2_in - mean_cen2 ) , sdom_mean_cen2_in


    #  Set up for plotting:
    if figpos is None:
        figpos = [0, 0, 1, 1]
    x0, y0, dx, dy = figpos
    pos1 = [x0+0.15*dx, y0+0.15*dy, 0.6*dx,  0.75*dy]
    pos2 = [x0+0.78*dx, y0+0.15*dy, 0.18*dx, 0.75*dy]

    # Invoke Zeus (cloud-compeller):
    if fig is None:
        fig = py.figure()
    ax = fig.add_subplot(111, position=pos1)

    ax.plot(cen1corrected - filtcen1 - mean_cen1, (flux/filtFlux - 1.) * 1e6, '+b', mew=1.5, alpha=0.5)
    ax.plot(cen2corrected - filtcen2 - mean_cen2, (flux/filtFlux - 1.) * 1e6, 'o', mfc='None', mew=1.5, mec='r', alpha=0.5)
    ax.set_ylabel('$\Delta$ Normalized Flux (ppm)', fontsize=fontsize)
    ax.set_xlabel('$\Delta$ Detrended Centroid (%s)' % unitLab, fontsize=fontsize)
    ax.minorticks_on()
    if grid:   ax.grid()
    
    hilit = fontsize*0.4
    ax.text(0.05, 0.95, 'Axis 1: %1.5f +/- %1.5f (stat)' % (val1a, eval1a), weight='bold', fontsize=0.8*fontsize+hilit*(np.abs(val1a/eval1a) > 3), color='c', horizontalalignment='left', transform=ax.transAxes)
    ax.text(0.05, 0.867, 'Axis 1: %1.5f +/- %1.5f (SDOM)' % (val1b, eval1b), weight='bold', fontsize=0.8*fontsize+hilit*(np.abs(val1b/eval1b) > 3), color='c', horizontalalignment='left', transform=ax.transAxes)
    ax.text(0.05, 0.783, 'Axis 2: %1.5f +/- %1.5f (stat)' % (val2a, eval2a), weight='bold', fontsize=0.8*fontsize+hilit*(np.abs(val2a/eval2a) > 3), color='orange', horizontalalignment='left', transform=ax.transAxes)
    ax.text(0.05, 0.7, 'Axis 2: %1.5f +/- %1.5f (SDOM)' % (val2b, eval2b), weight='bold', fontsize=0.8*fontsize+hilit*(np.abs(val2b/eval2b) > 3), color='orange', horizontalalignment='left', transform=ax.transAxes)
    ax.set_title(title)

    ax2 = fig.add_subplot(111, position=pos2)
    ax2.plot([0,0], [0, min(ax.get_ylim())], ':k', linewidth=2)
    ax2.plot([0, mean_cen1_in - mean_cen1], (np.array([meanOut, meanIn])-1.)*1e6, '--b', linewidth=1.5)
    ax2.plot([0, mean_cen2_in - mean_cen2], (np.array([meanOut, meanIn])-1.)*1e6, '--r', linewidth=1.5)
    ax2.errorbar(mean_cen1_in - mean_cen1, (meanIn-1.)*1e6, xerr=e_mean_cen1_in, fmt='xb', mfc='b', mec='b', mew=1, ms=fontsize)
    ax2.errorbar(mean_cen2_in - mean_cen2, (meanIn-1.)*1e6, xerr=e_mean_cen2_in, fmt='xr', mfc='r', mec='r', mew=1, ms=fontsize)
    ax2.errorbar(mean_cen1_in - mean_cen1, (meanIn-1.)*1e6, xerr=sdom_mean_cen1_in, fmt=',b', mfc='b', mec='b', mew=2, ms=fontsize)
    ax2.errorbar(mean_cen2_in - mean_cen2, (meanIn-1.)*1e6, xerr=sdom_mean_cen2_in, fmt=',r', mfc='r', mec='r', mew=2, ms=fontsize)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels([])
    ax2.set_xlim(np.array([-1,1]) * max(np.abs(ax2.get_xlim())))
    [tt.set_rotation(90) for tt in ax2.get_xaxis().get_ticklabels()]
    ax2.minorticks_on()
    if grid:   ax2.grid()

    all_tick_labels = ax.get_xticklabels() + ax.get_yticklabels() + \
        ax2.get_xticklabels() + ax2.get_yticklabels()
    [lab.set_fontsize(fontsize*0.8) for lab in all_tick_labels]
    
    return fig, (ax, ax2)
    
    
    
def plotFrame(input, fontsize=16, fig=None, figpos=None, shift=[0,0]):
    #  Set up for plotting:
    if figpos is None:
        figpos = [0, 0, 1, 1]
    x0, y0, dx, dy = figpos
    pos = [x0+0.1*dx, y0+0.1*dy, 0.8*dx,  0.8*dy]
    if fig is None:
        fig = py.figure()

    ax = fig.add_subplot(111, position=pos)
    py.sca(ax)
    py.title("Median Frame")
    logframe = np.log10(input.medianFrame)
    logframe = np.ma.masked_invalid(logframe)
    logframe.fill_value=0
    logframe = logframe.filled()
    pixel_decorrelation.plot_label(logframe,input.catcut,input.epic, shift=shift, colorbar=False)
    py.contour(input.crudeApertureMask, [0.5], linewidths=2, colors='g')
    py.title("log10(Median Frame)")
    return fig, ax

def plotLightCurves(time, flux, inTransit=None, fontsize=16, title='', fig=None, figpos=None, medFiltWid=1, per=None):
    if per is None:
        per = 9e99

    if figpos is None:
        figpos = [0, 0, 1, 1]
    x0, y0, dx, dy = figpos
    pos = [x0+0.1*dx, y0+0.1*dy, 0.8*dx,  0.8*dy]

    if medFiltWid==1:
        filtFlux = 1.
    else:
        filtFlux = signal.medfilt(flux, medFiltWid)

    if fig is None:
        fig = py.figure()

    ax = fig.add_subplot(111, position=pos)
    ax.plot(time % per, 1e6*(flux/filtFlux - 1.), '.b')
    if inTransit is not None and inTransit.sum()>0:
        ax.plot(time[inTransit] % per, 1e6*(flux/filtFlux - 1.)[inTransit], '.r')
        ax.set_xlabel('Phase-Folded Time (days)', fontsize=fontsize)
    else:
        ax.set_xlabel('Time (days)', fontsize=fontsize)

    timespan = time.max() - time.min()
    ax.set_xlim([time.min() - timespan*0.01, time.max() + timespan*0.01])

    [lab.set_fontsize(fontsize*0.8) for lab in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.set_ylabel('$\Delta$ Flux (ppm)', fontsize=fontsize)
    ax.minorticks_on()
    return fig, ax

def plotDiagnostics(lcFits, lcPickle, pixFile, fontsize=14, medFiltWid=1, tt=None, per=None, t14=None, depth=None, edepth=None):
    """An evolving construction -- combine all diagnostic plots in one.

    :INPUTS:
      lcObj - output FITS file from photometry analysis

      lcPickle - output Pickle file from photometry analysis

      pixFile - target pixel file from K2 website

    :EXAMPLE:
      ::


        import lightcurve_diagnostics as ld
        from astropy.io import fits as pyfits
        import atpy

        cand = atpy.Table('/Users/ianc/proj/mdwarfs/kepler2/cycle0/C0-cand_v0.1_status.csv', type='ascii')
        field = 0
        _pix = '/Users/ianc/proj/transit/kepler/data/'
        _proc = '/Users/ianc/proj/transit/kepler/proc/'

        for ii in range(22):
            epic = cand.epic[ii]
            tt, per = 2454833+cand.t0[ii], cand.P[ii]
            t14 = cand.t14_hr[ii]/24.
            pixfn = _pix + 'ktwo%i-c%02i_lpd-targ.fits' % (epic, field)
            procfn = _proc + '%i.fits' % epic
            procpickle = _proc + '%i.pickle' % epic

            fig, axs = ld.plotDiagnostics(procfn, procpickle, pixfn, medFiltWid=47, tt=tt, per=per, t14=t14)
            axs[2].set_title(epic)

    """
    # 2014-10-03 13:29 IJMC: Created

    # Load & massage data files:
    dat = pyfits.getdata(lcFits)
    input = tools.loadpickle(lcPickle)
    cube, headers = pixel_decorrelation.loadPixelFile(pixFile, tlimits=[dat.time.min() - 2454833 - 1e-6, np.inf])
    time, data, edata = cube['time'],cube['flux'],cube['flux_err']
    data, edata = pixel_decorrelation.preconditionDataCubes(data, edata, medSubData=False)

    try:
        inTransit = np.abs((time-tt + per/2.) % per - per/2) < (t14/2.)
    except:
        print "Could not compute 'inTransit' -- setting to 'None'."
        inTransit = None


    # Prepare for Cloud Plot
    c1, c2, ec1, ec2 = computeCentroids(data, edata, input.crudeApertureMask)
    index = input.noThrusterFiring

    # Start plotting
    fig = py.figure(tools.nextfig(), [10, 8])
    axs = []
    fig, ax1 = plotFrame(input, fig=fig, figpos=[0.5, 0.5, 0.4, 0.4], fontsize=fontsize)
    axs.append(ax1)

    fig, ax2 = plotLightCurves(input.time[index], input.cleanFlux[index], inTransit=inTransit[index], fontsize=fontsize, medFiltWid=medFiltWid, fig=fig, figpos=[0.0, 0.04, 1, 0.4])
    axs.append(ax2)

    fig, ax34 = cloudPlot(c1[index], c2[index], dat.cleanFlux[index], inTransit[index], e_cen1=ec1[index], e_cen2=ec2[index], time=time[index], fig=fig, figpos=[0.0, 0.4, 0.55, 0.6], fontsize=fontsize*0.65, medFiltWid=medFiltWid)
    axs = axs + list(ax34)

    return fig, axs

def centroidTransit(time, flux, transitModel, cen1, cen2, ecen1, ecen2, medFiltWid=47, rescaleErrors=True):
    """Apply centroid-motion vs. transit test; Sec. 2.2 of Bryson et al. 2013.

    :INPUTS:
      time : 1D NumPy array
        Time index.

      transitModel : 1D NumPy array
        Model light curve, equal to unity out-of-transit.

      cen1, cen2, ecen1, ecen2 : 1D NumPy arrays
        Centroid motions and their uncertainties.

    :RETURNS:
       shift_distance, e_shift_distance, gamma1, egamma1, \
         gamma2, egamma2, ell_1, ell_2, 

        'shift_distance' and its uncertainty derive from Eqs. 6+7.

        'gamma' values refer to centroid shift in each axis.

        'ell_1' and 'ell_2' are the metric of Eq. 8 (B13).

    :TO_DO:
      Computation of 'ell_1' and 'ell_2' still seems suspect.  Check this!
    """
    # 2014-10-03 15:22 IJMC: Created

    # Decorrelate spacecraft-induced motion from centroid-motion fit:
    mostlyJunk = pixel_decorrelation.getArcLengths(time, [(cen1, cen2)], retmodXY=True)
    cen1model, cen2model = mostlyJunk[-1][0]
    cen1corrected = cen1 - cen1model
    cen2corrected = cen2 - cen2model

    mean_cen1 = (cen1corrected)[transitModel==1].mean()
    mean_cen2 = (cen2corrected)[transitModel==1].mean()
    deltaCen1 = cen1corrected  - mean_cen1
    deltaCen2 = cen2corrected  - mean_cen2

    # Appropriately pre-whiten the centroids and transit model; 
    #    normalize the latter:
    filtDeltaCen1 = signal.medfilt(deltaCen1, medFiltWid)
    filtDeltaCen2 = signal.medfilt(deltaCen2, medFiltWid)
    filtTransitModel = transitModel / signal.medfilt(transitModel, medFiltWid) - 1.
    filtTransitModel /= filtTransitModel.std()

    # Eq. 4:
    gamma1 = ((deltaCen1 - filtDeltaCen1) * filtTransitModel / ecen1**2).sum() / \
        ((filtTransitModel/ecen1)**2).sum()
    gamma2 = ((deltaCen2 - filtDeltaCen2) * filtTransitModel / ecen2**2).sum() / \
        ((filtTransitModel/ecen2)**2).sum()

    if rescaleErrors:
        chi1 = (((deltaCen1 - filtDeltaCen1 - gamma1*filtTransitModel) / ecen1)**2).sum()
        ecen1 *= np.sqrt(chi1 / ecen1.size)
        chi2 = (((deltaCen2 - filtDeltaCen2 - gamma2*filtTransitModel) / ecen2)**2).sum()
        ecen2 *= np.sqrt(chi2 / ecen1.size)
    

    # Eq. 5:
    egamma1 = 1./np.sqrt(((filtTransitModel/ecen1)**2).sum())
    egamma2 = 1./np.sqrt(((filtTransitModel/ecen2)**2).sum())

    #print gamma1, egamma1
    #print gamma2, egamma2

    # Eq. 6:
    shift_distance = np.sqrt(gamma1**2 + gamma2**2)
    e_shift_distance0 = np.sqrt((gamma1 * egamma1)**2 + (gamma2 + egamma2)**2) / shift_distance
    #pdb.set_trace()
    # I can't reproduce Eq. 7, so I just use Monte Carlo:
    mc_g1 = np.random.normal(gamma1, egamma1, 1e5)
    mc_g2 = np.random.normal(gamma2, egamma2, 1e5)
    e_shift_distance = np.sqrt(mc_g1**2 + mc_g2**2).std()

    #print shift_distance, e_shift_distance, e_shift_distance0

    ell_1 = ((deltaCen1 - filtDeltaCen1) * filtTransitModel).sum() / \
        (egamma1 * np.sqrt((filtTransitModel**2).sum()))
    ell_2 = ((deltaCen2 - filtDeltaCen2) * filtTransitModel).sum() / \
        (egamma2 * np.sqrt((filtTransitModel**2).sum()))

    #print ell_1, ell_2

    return shift_distance, e_shift_distance, gamma1, egamma1, gamma2, egamma2, ell_1, ell_2, 


def centroidSourceOffset(time, flux, transitModel, cen1, cen2, ecen1, ecen2, depth, edepth, medFiltWid=47, rescaleErrors=True, cen1model=None, cen2model=None):
    """Estimate position offset of transiting source. See Sec. 2.3 of Bryson+2013.

    depth, edepth : scalars
      reported transit depth and its uncertainty

    """
    # 2014-10-03 16:49 IJMC: Created

    if depth is None:
        depth = transitModel[transitModel<1].mean()

    inTransit = transitModel < 1.

    # Apply pointing model:
    if cen1model is None or cen2model is None:
        mostlyJunk = pixel_decorrelation.getArcLengths(time, [(cen1, cen2)], retmodXY=True)
        cen1model, cen2model = mostlyJunk[-1][0]
    cen1corrected = cen1 - cen1model
    cen2corrected = cen2 - cen2model
    filtcen1 = signal.medfilt(cen1corrected, medFiltWid)
    filtcen2 = signal.medfilt(cen2corrected, medFiltWid)

    # Compute centroids in, out, and global:
    mean_cen1, e_mean_cen1, mean_cen1_out, e_mean_cen1_out, mean_cen1_in, e_mean_cen1_in, sdom_mean_cen1_in = \
        computeMeanCentroids(cen1corrected - filtcen1, ecen1, inTransit)
    mean_cen2, e_mean_cen2, mean_cen2_out, e_mean_cen2_out, mean_cen2_in, e_mean_cen2_in, sdom_mean_cen2_in = \
        computeMeanCentroids(cen2corrected - filtcen2, ecen2, inTransit)

    # Compute centroid-shift distance:
    shift_distance, e_shift_distance, gamma1, egamma1, \
        gamma2, egamma2, ell_1, ell_2 = \
        centroidTransit(time, flux, transitModel, cen1, cen2, ecen1, ecen2, \
                            medFiltWid=medFiltWid, rescaleErrors=rescaleErrors)


    # Apply Eq. 9:
    dilution = (1./depth - 1.)
    offset1 = dilution * gamma1
    offset2 = dilution * gamma2

    # Now Eq. 11:
    eoffset1 = np.sqrt(dilution**2 * (e_mean_cen1_in**2 + e_mean_cen1_out**2)  + \
        (gamma1 * edepth)**2 / depth**4  + e_mean_cen1**2)
    eoffset2 = np.sqrt(dilution**2 * (e_mean_cen2_in**2 + e_mean_cen2_out**2)  + \
        (gamma2 * edepth)**2 / depth**4  + e_mean_cen2**2)

    return offset1, eoffset1, offset2, eoffset2
