"""
Functions to conduct light-curve analyses, especially of Kepler/K2
targets.

Many of these functions follow the discussions by Bryson et
al. (2013). See: http://adsabs.harvard.edu/abs/2013PASP..125..889B

The main, 'driver' function is :func:`plotDiagnostics`, but many other
functions can be called individually.
"""

import pixel_decorrelation
import analysis as an
import tools
import phot

import numpy as np
import pylab as py
import sys
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits

import pandas as pd
import pixel_io


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
        pickledat = ld.tools.loadpickle(procfn.replace('.fits', '.pickle'))
        cube, headers = pixel_decorrelation.loadPixelFile(pixfn, tlimits=[dat.time.min() - 2454833 - 1e-6, np.inf])
        time, data, edata = cube['time'],cube['flux'],cube['flux_err']

        tt, per, t14 = 2456768.6558394236, 5.3156579762, 0.26
        inTransit = np.abs((time-tt + per/2.) % per - per/2) < (t14/2.)
        mask = pickledat.crudeApertureMask
        c1, c2, ec1, ec2 = ld.computeCentroids(data, edata, mask)
        ind = dat.noThrusterFiring.astype(bool)
        ld.cloudPlot(c1[ind], c2[ind], dat['cleanFlux'][ind], inTransit[ind], e_cen1=ec1[ind], e_cen2=ec2[ind], time=time[ind])

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
        ind = dat.noThrusterFiring.astype(bool)
        ld.cloudPlot(c1[ind], c2[ind], dat['cleanFlux'][ind], inTransit[ind], e_cen1=ec1[ind], e_cen2=ec2[ind], time=time[ind])
      

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
    
    
    
def plotFrame(input, fontsize=16, fig=None, figpos=None, shift=[0,0], cmap=None, title='log10(Median Frame)', colorbar=False):
    """Plot the log-stretched median frame, and nearby EPIC objects.

    :INPUTS:
      input : object
        Pickled-object output from our photometric calibration
        routines. This must include subfields of: 'medianFrame',
        'crudeApertureMask', 'catcut', 'epic', 'x', and 'y'.
        """
    
    #  Set up for plotting:
    if figpos is None:
        figpos = [0, 0, 1, 1]
    x0, y0, dx, dy = figpos
    pos = [x0+0.1*dx, y0+0.1*dy, 0.8*dx,  0.8*dy]
    if fig is None:
        fig = py.figure()

    ax = fig.add_subplot(111, position=pos)
    py.sca(ax)
    logframe = np.log10(input.medianFrame)
    logframe = np.ma.masked_invalid(logframe)
    logframe.fill_value=0
    logframe = logframe.filled()
    ax.plot(input.y, input.x, '.r')
    pixel_decorrelation.plot_label(logframe,input.catcut,input.epic, shift=shift, colorbar=False, cmap=cmap)
    if colorbar:
        py.colorbar()
    py.contour(input.crudeApertureMask, [0.5], linewidths=2, colors='g')
    py.title(str(title))
    return fig, ax

def plotLightCurves(time, flux, inTransit=None, fontsize=16, title='', fig=None, figpos=None, medFiltWid=1, per=None, inCol='brown', outCol='g'):
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
    if inTransit is not None and inTransit.sum()>0:
        ax.plot(time[True-inTransit] % per, 1e6*(flux/filtFlux - 1.)[True-inTransit], '.', color=outCol)
        ax.plot(time[inTransit] % per, 1e6*(flux/filtFlux - 1.)[inTransit], '.', color=inCol)
        ax.set_xlabel('Phase-Folded Time (days)', fontsize=fontsize)
    else:
        ax.plot(time % per, 1e6*(flux/filtFlux - 1.), '.', color=outCol)
        ax.set_xlabel('Time (days)', fontsize=fontsize)
        timespan = time.max() - time.min()
        ax.set_xlim([time.min() - timespan*0.01, time.max() + timespan*0.01])

    [lab.set_fontsize(fontsize*0.8) for lab in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.set_ylabel('$\Delta$ Flux (ppm)', fontsize=fontsize)
    ax.minorticks_on()
    return fig, ax


def plotDiagnostics_wrap(lcFits, lcPickle, pixFile, candcsv):
    df = pd.read_csv(candcsv)
    df = df.dropna(subset=['starname P t0 fit_p'.split()])
    df['starname'] = df.starname.astype(int).astype(str)
    df.index = df.starname 
    cube,headers = pixel_io.loadPixelFile(pixFile)

    starname = '%s' % headers[0]['KEPLERID'] 
    d = df.ix[starname]
    if type(d) is pd.core.frame.DataFrame:
        print "Warning: %i columns, choosing first" % len(d)
        d = d.iloc[0]

    tt, per = 2454833+d['t0'], d['P']
    depth = d['fit_p']**2
    edepth = d['fit_p']**2/30
    tau = d['fit_tau']
    b = d['fit_b']

    rsa = 2*np.pi*tau/per
    tani = 1./(b*rsa)
    t14 = (per/np.pi) * np.arcsin(rsa * np.sqrt((1+depth**0.5)**2 - b**2) / np.sin(np.arctan(tani)))

    if not np.isfinite(t14): t14 = tau

    time = pyfits.getdata(lcFits).time
    inTransit = np.abs((time-tt + per/2.) % per - per/2) < (t14/2.)
    poorTransitModel = (1./depth - inTransit) * depth

    fig, axs = plotDiagnostics(lcFits, lcPickle, pixFile, poorTransitModel, medFiltWid=47, tt=tt, per=per, t14=t14, depth=depth, edepth=edepth)
    return fig,axs
    
def plotDiagnostics(lcFits, lcPickle, pixFile, transitModel, fontsize=14, medFiltWid=1, tt=None, per=None, t14=None, depth=None, edepth=None, empiricalErrors=False):
    """An evolving construction -- combine all diagnostic plots in one.

    :INPUTS:
      lcFits - output FITS filename, from photometry analysis

      lcPickle - output Pickle filename, from photometry analysis

      pixFile - target pixel filename, from K2 website

      transitModel - model light curve, 1 outside of
                     transit. Eventually, we will compute this
                     in-function.

    :EXAMPLE:
      ::


        import lightcurve_diagnostics as ld
        import atpy
        import os
        import numpy as np

        _home = os.path.expanduser('~')
        cand = atpy.Table(_home+'/proj/mdwarfs/kepler2/cycle0/C0_10-10_planet-candidates_status_v2.csv', type='ascii')
        field = 0
        _pix = _home+'/proj/transit/kepler/data/'
        _proc = _home+'/proj/transit/kepler/proc/c0_v2/'

        ii = 10
        if cand.fit_b[ii]<0.99:
            epic = cand.starname[ii]
            tt, per = 2454833+cand.t0[ii], cand.P[ii], 
            depth, edepth = cand.fit_p[ii]**2, (cand.fit_p[ii]**2)/30
            tau, b = cand.fit_tau[ii], cand.fit_b[ii]
            rsa = 2*np.pi*tau/per
            tani = 1./(b*rsa)
            t14 = (per/np.pi) * np.arcsin(rsa * np.sqrt((1+depth**0.5)**2 - b**2) / np.sin(np.arctan(tani)))
            if not np.isfinite(t14): t14 = tau

            pixfn = _pix + 'ktwo%i-c%02i_lpd-targ.fits' % (epic, field)
            procfn = _proc + 'new%i.fits' % epic
            procpickle = _proc + 'new%i.pickle' % epic

            time = ld.pyfits.getdata(procfn).time
            inTransit = np.abs((time-tt + per/2.) % per - per/2) < (t14/2.)
            poorTransitModel = (1./depth - inTransit) * depth

            fig, axs = ld.plotDiagnostics(procfn, procpickle, pixfn, poorTransitModel, medFiltWid=47, tt=tt, per=per, t14=t14, depth=depth, edepth=edepth)

    """
    # 2014-10-03 13:29 IJMC: Created

    # Load & massage data files:
    dat = pyfits.getdata(lcFits)

    import cPickle as pickle
    from pixel_decorrelation import baseObject
    with open(lcPickle,'r') as f:
        input = pickle.load(f)

    cube, headers = pixel_decorrelation.loadPixelFile(pixFile, tlimits=[dat.time.min() - 2454833 - 1e-6, np.inf])
    time, data, edata = cube['time'],cube['flux'],cube['flux_err']
    data, edata = pixel_decorrelation.preconditionDataCubes(data, edata, medSubData=True)
    nobs = time.size

    inTransit, preTran, postTran = indexInOutTransit(dat.time, transitModel, per)

    ## Robust variability estimate:
    #if empiricalErrors:
    #    readnoise = headers[1]['READNOIS']
    #    tint = headers[1]['INT_TIME']
    #    nframe = headers[1]['NUM_FRM']
    #    # nphoton_1frame = data * tint
    #    # e_nphot_1 = np.sqrt(nphoton_1frame + readnoise**2)
    #    # e_nphot_N = 
    #    # e_data = e_nphot_N / np.sqrt(nframe)
    #    # e_data = np.sqrt(data * (tint * nframe) + (readnoise*nframe)**2) / np.sqrt(nframe)
    #    edata_phot = np.sqrt(data * (tint * nframe) + (nframe * readnoise)**2) / (tint * nframe)
    #    edata_phot = np.sqrt(data * tint + readnoise**2) / tint
    #    pdb.set_trace()
    #    edata = np.array([edata, edata_phot]).max(0)

    # Prepare for Cloud Plot

    c1, c2, ec1, ec2 = computeCentroids(data, edata, input.crudeApertureMask)
    index = input.noThrusterFiring
    if depth is not None and edepth is not None:
        coffset1, e_coffset1, coffset2, e_coffset2, mean_out1, e_mean_out1, mean_out2, e_mean_out2 = centroidSourceOffset(\
            time[index], input.cleanFlux[index], transitModel[index], \
                c1[index], c2[index], ec1[index], ec2[index], depth, edepth, \
                medFiltWid=47, rescaleErrors=True, \
                cen1model=None, cen2model=None)

    # Start plotting
    fig = py.figure(tools.nextfig(), [15, 9])
    axs = []
    fig, ax1 = plotFrame(input, fig=fig, figpos=[0.73, 0.72, 0.25, 0.25], fontsize=fontsize, colorbar=False)
    axs.append(ax1)

    fig, ax2 = plotLightCurves(input.time[index], input.cleanFlux[index], inTransit=inTransit[index], fontsize=fontsize, medFiltWid=medFiltWid, per=per, fig=fig, figpos=[0.02, 0.04, .64, 0.4])
    axs.append(ax2)


    fig, ax34 = cloudPlot(c1[index], c2[index], dat.cleanFlux[index], inTransit[index], e_cen1=ec1[index], e_cen2=ec2[index], time=time[index], fig=fig, figpos=[0.02, 0.4, 0.35, 0.6], fontsize=fontsize*0.65, medFiltWid=medFiltWid)
    axs = axs + list(ax34)
    axs[2].set_title(input.epic)


    diffImage, e_diffImage, inImage, e_inImage, outImage, e_outImage = \
        constructDiffImage(time, data, transitModel, per, edata=edata, \
                               posx=c1, posy=c2, shift='xcshift', \
                               retall=True, empiricalErrors=False)
    fig, ax5678, caxs = plotDiffImages(diffImage, e_diffImage, inImage, outImage, apertureMask=input.crudeApertureMask, catcut=input.catcut, epic=input.epic, figpos=[0.37, 0.45, 0.4, 0.5], fig=fig, cmap=py.cm.cubehelix, conCol='r', fontsize=fontsize*0.8, loc=input.loc)
    axs = axs + ax5678


    fig, ax9 = plotPixelTransit(input, data, transitModel, mask=np.ones(data.shape[1:]), figpos=[.73, .45, .25, .25], fig=fig, fontsize=fontsize*0.8)
    if np.abs(coffset1/e_coffset1)>3 or np.abs(coffset2/e_coffset2)>3:
        ax9.plot([mean_out2, mean_out2+coffset2], [mean_out1, mean_out1+coffset1], '.r-')
        axs[3].text(.5, .9, '%1.1e +/- %1.1e' % (coffset1, e_coffset1), color='b', horizontalalignment='center', fontsize=fontsize*0.8, transform=axs[3].transAxes, weight='bold')
        axs[3].text(.5, .8, '%1.1e +/- %1.1e' % (coffset2, e_coffset2), color='r', horizontalalignment='center', fontsize=fontsize*0.8, transform=axs[3].transAxes, weight='bold')
    axs.append(ax9)


    # PRF fits to the difference and out-of-transit images:
    #fitDiff, e_fitDiff, modDiff = fitPRF2Image(diffImage, e_diffImage+(1.-input.crudeApertureMask)*9e9, pixFile, input.loc, input.apertures[0]*2, ngrid=40, reterr=True)
    #fitOut, e_fitOut, modOut = fitPRF2Image(outImage, e_outImage+(1.-input.crudeApertureMask)*9e9, pixFile, input.loc, input.apertures[0]*2, ngrid=40, reterr=True)
    #PRFmotion = fitOut - fitDiff
    #e_PRFmotion = np.sqrt(e_fitOut**2 + e_fitDiff**2)
    #pm_metric = np.sqrt(((PRFmotion / e_PRFmotion)**2).sum())
    #ax10 = fig.add_subplot(111, position=[0.8, 0.7, 0.15, 0.25])
    #prf_diff_text = ['','PRF-fitting D.I.A.', '$\Delta$ ax1 =', '   %1.3f +\- %1.3f pix' % (PRFmotion[0], e_PRFmotion[0]), '$\Delta$ ax2 =', '   %1.3f +\- %1.3f pix' % (PRFmotion[1], e_PRFmotion[1]), '', 'Metric = %1.1f' % pm_metric, '']
    #tools.textfig(prf_diff_text, ax=ax10, fig=fig, fontsize=fontsize*0.8)

    fig, ax1014 = plotDIA_PRF_fit(pixFile, input, diffImage, e_diffImage, outImage, e_outImage, ngrid=30, fontsize=fontsize*0.75, fig=fig, figpos=[0.6, 0, 0.4, 0.45])
    axs += ax1014


    return fig, axs

def computeCentroidTransit(time, flux, transitModel, cen1, cen2, ecen1, ecen2, medFiltWid=47, rescaleErrors=True):
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
    cen1corrected = cen1 - cen1model + cen1model.mean()
    cen2corrected = cen2 - cen2model + cen2model.mean()

    mean_cen1 = (cen1corrected)[transitModel==1].mean()
    mean_cen2 = (cen2corrected)[transitModel==1].mean()
    #deltaCen1 = cen1corrected  - mean_cen1
    #deltaCen2 = cen2corrected  - mean_cen2

    # Appropriately pre-whiten the centroids and transit model; 
    #    normalize the latter:
    filtDeltaCen1 = cen1corrected / signal.medfilt(cen1corrected, medFiltWid) - 1.
    filtDeltaCen2 = cen2corrected / signal.medfilt(cen2corrected, medFiltWid) - 1.
    filtTransitModel = transitModel / signal.medfilt(transitModel, medFiltWid) - 1.
    filtTransitModel /= filtTransitModel.std() # <-- according to S. Bryson

    # Eq. 4:
    gamma1 = ((filtDeltaCen1) * filtTransitModel / ecen1**2).sum() / \
        ((filtTransitModel/ecen1)**2).sum()
    gamma2 = ((filtDeltaCen2) * filtTransitModel / ecen2**2).sum() / \
        ((filtTransitModel/ecen2)**2).sum()

    if rescaleErrors:
        chi1 = (((filtDeltaCen1 - gamma1*filtTransitModel) / ecen1)**2).sum()
        ecen1 *= np.sqrt(chi1 / ecen1.size)
        chi2 = (((filtDeltaCen2 - gamma2*filtTransitModel) / ecen2)**2).sum()
        ecen2 *= np.sqrt(chi2 / ecen1.size)
    

    # Eq. 5:
    egamma1 = 1./np.sqrt(((filtTransitModel/ecen1)**2).sum())
    egamma2 = 1./np.sqrt(((filtTransitModel/ecen2)**2).sum())

    #print gamma1, egamma1
    #print gamma2, egamma2

    # Eq. 6:
    shift_distance = np.sqrt(gamma1**2 + gamma2**2)
    e_shift_distance0 = np.sqrt((gamma1 * egamma1)**2 + (gamma2 + egamma2)**2) / shift_distance
    # I just use Monte Carlo for Eq. 7:
    mc_g1 = np.random.normal(gamma1, egamma1, 1e5)
    mc_g2 = np.random.normal(gamma2, egamma2, 1e5)
    e_shift_distance = np.sqrt(mc_g1**2 + mc_g2**2).std()

    #print shift_distance, e_shift_distance, e_shift_distance0

    ell_1 = ((filtDeltaCen1) * filtTransitModel).sum() / \
        (egamma1 * np.sqrt((filtTransitModel**2).sum()))
    ell_2 = ((filtDeltaCen2) * filtTransitModel).sum() / \
        (egamma2 * np.sqrt((filtTransitModel**2).sum()))

    #print ell_1, ell_2

    return shift_distance, e_shift_distance, gamma1, egamma1, gamma2, egamma2, ell_1, ell_2, 




def computePixelTransit(input, data, transitModel, medFiltWid=47, rescaleErrors=True, mask=None):
    """Apply pixel-flux vs. transit tests; Sec. 4 of Bryson et al. 2013.

    :INPUTS:
      input : pickle object output from our photometry calibratios

      data : 3D NumPy array
        Pixel data.

      transitModel : 1D NumPy array
        Model light curve, equal to unity out-of-transit.

    :RETURNS:
       gammas, e_gammas -- 2D NumPy Arrays

    :NOTE:

      We must be much more clever with K2 data. Targets move so much
      on the detector that any given pixel's time series shows
      essentially no evidence of transits. So instead, we decorrelate
      vs. motion and a transit model

    """
    # 2014-10-09 14:42 IJMC: Created.
    

    s_norm = pixel_decorrelation.normalizeVector(input.arcLength)
    y_norm = pixel_decorrelation.normalizeVector(input.y)
    svecs = np.vstack([s_norm**nn for nn in xrange(input.nordPixel1d)])
    if hasattr(input, 'nordPixel2d') and input.nordPixel2d>1:
        yvecs = np.vstack([y_norm**nn for nn in xrange(1, input.nordPixel2d)])
        vecs = np.vstack((transitModel, svecs, yvecs))
    else:
        vecs = np.vstack((transitModel, svecs))
        
    ind = input.goodvals * input.noThrusterFiring
    vecs = vecs[:, ind]
    data = data[ind]
    transitModel = transitModel[ind]
    

    #if posx is not None and posy is not None:
    #    pdb.set_trace()
    #    data = shiftImages(data, posx-posx.mean(), posy-posy.mean())
        
    nobs, nrow, ncol = data.shape
    gammas = np.zeros((nrow, ncol), dtype=float)
    egammas = np.zeros((nrow, ncol), dtype=float)
    if mask is None:
        mask = input.crudeApertureMask

    # S. Bryson says that we do *not* pre-whiten or normalize the transit model:
    filtTransitModel = transitModel - 1. 
    denom = (filtTransitModel**2).sum()
    sqrtvec = np.ones(nobs) / np.sqrt(nobs)
    for irow in xrange(nrow):
        for icol in xrange(ncol):
            if mask[irow, icol]:
                pix = data[:,irow,icol]
                # Eq. 4:
                fit, efit = an.lsq(vecs.T, pix, checkvals=False)
                mod = np.dot(fit, vecs)
                mod1 = np.dot(fit[1:], vecs[1:]) + fit[0]
                pixmod = pix / mod1
                filtPix = (pixmod) / signal.medfilt(pixmod, medFiltWid) - 1.
                #if irow==12 and icol==12: pdb.set_trace()
                filtPix[True-np.isfinite(filtPix)] = 0.
                gammas[irow,icol] = (filtPix * (filtTransitModel)).sum() / denom
                if rescaleErrors:
                    chi = ((filtPix - gammas[irow,icol]*(filtTransitModel))**2).sum()
                    ecen = sqrtvec * np.sqrt(chi)
                # Eq. 5:
                egammas[irow,icol] = 1./np.sqrt((((filtTransitModel)/ecen)**2).sum())


    return gammas, egammas


def centroidSourceOffset(time, flux, transitModel, cen1, cen2, ecen1, ecen2, depth, edepth, medFiltWid=47, rescaleErrors=True, cen1model=None, cen2model=None):
    """Estimate position offset of transiting source. See Sec. 2.3 of Bryson+2013.

    depth, edepth : scalars
      reported transit depth and its uncertainty

    :RETURNS:
      offset1, eoffset1, offset2, eoffset2
    """
    # 2014-10-03 16:49 IJMC: Created

    if depth is None:
        depth = transitModel[transitModel<1].mean()

    inTransit = transitModel < 1.

    # Apply pointing model:
    if cen1model is None or cen2model is None:
        mostlyJunk = pixel_decorrelation.getArcLengths(time, [(cen1, cen2)], retmodXY=True)
        cen1model, cen2model = mostlyJunk[-1][0]
    cen1corrected = cen1 - cen1model + cen1model.mean()
    cen2corrected = cen2 - cen2model + cen2model.mean()
    #filtcen1 = signal.medfilt(cen1corrected, medFiltWid)
    #filtcen2 = signal.medfilt(cen2corrected, medFiltWid)

    # Compute centroids in, out, and global:
    mean_cen1, e_mean_cen1, mean_cen1_out, e_mean_cen1_out, mean_cen1_in, e_mean_cen1_in, sdom_mean_cen1_in = \
        computeMeanCentroids(cen1corrected, ecen1, inTransit)
    mean_cen2, e_mean_cen2, mean_cen2_out, e_mean_cen2_out, mean_cen2_in, e_mean_cen2_in, sdom_mean_cen2_in = \
        computeMeanCentroids(cen2corrected, ecen2, inTransit)

    # Compute centroid-shift distance:
    shift_distance, e_shift_distance, gamma1, egamma1, \
        gamma2, egamma2, ell_1, ell_2 = \
        computeCentroidTransit(time, flux, transitModel, cen1, cen2, ecen1, ecen2, \
                            medFiltWid=medFiltWid, rescaleErrors=rescaleErrors)


    # Apply Eq. 9:
    dilution = (1./depth - 1.)
    offset1 = dilution * gamma1
    offset2 = dilution * gamma2

    # Now Eq. 11:
    ## Full equations:
    #eoffset1 = np.sqrt(dilution**2 * (e_mean_cen1_in**2 + e_mean_cen1_out**2)  + \
    #    (gamma1 * edepth)**2 / depth**4  + e_mean_cen1**2)
    #eoffset2 = np.sqrt(dilution**2 * (e_mean_cen2_in**2 + e_mean_cen2_out**2)  + \
    #    (gamma2 * edepth)**2 / depth**4  + e_mean_cen2**2)

    ## Equations for simplified analysis:
    eoffset1 = egamma1 * dilution
    eoffset2 = egamma2 * dilution

    return offset1, eoffset1, offset2, eoffset2, mean_cen1_out, e_mean_cen1_out, mean_cen2_out, e_mean_cen2_out, 


def indexInOutTransit(time, transitModel, period, nbuffer=3, inDepthFrac=0.75, depth=None, cadence=None):
    """Return Boolean Masks for in- and out-of-transit indices, *NEAR
    TRANSITS*.  This follows the discussion in Bryson et al. (2013),
    Sec. 3.2, and Figs 14 & 15.

    :INPUTS:
      time : 1D NumPy array
        Time index.

      transitModel : 1D NumPy array
        Model transit light curve, equal to unity out-of-transit.

      period : positive scalar
        Planet's orbital period

      nbuffer : positive int
        Buffer between transit ingress/egress and the start of the
        included out-of-transit region.

      inDepthFrac : positive scalar
        Values where (transitModel - 1.) <= (-inDepthFrac*depth) are
        included as 'in' transit.

      depth : positive scalar
        Transit depth, from your models.  If not included, set to
        min(1. - transitModel)

      cadence : positive scalar
        The temporal sampling of 'time', used to compute cadence numbering. 

    :TO_DO:
      Account for bad cadence numbers (as described by Bryson et al.)
      for data gaps, thermal events, thruster firings/attitude tweaks,
      other transits, etc.
    """
    # 2014-10-07 16:43 IJMC: Created

    if cadence is None:
        cadence = np.diff(time).min()
    if depth is None:
        depth = 1. - transitModel.min()
    cadNo = ((time - time.min()) / cadence).round().astype(int)

    inTransit = (transitModel - 1.) <= (-inDepthFrac*depth)
    whollyInTransit = transitModel < 1
    transitNo = (np.round((time - time[inTransit.nonzero()[0][0]]) / period)).astype(int)
    preTransit = np.zeros(time.size, dtype=bool)
    postTransit = np.zeros(time.size, dtype=bool)
    for ntran in np.unique(transitNo):
        index = transitNo==ntran
        nCadInside = whollyInTransit[index].sum()
        insideInds = (whollyInTransit*index).nonzero()[0]
        if insideInds.size>0:
            firstInside, lastInside = cadNo[insideInds[[0,-1]]]
            pre = (firstInside - nbuffer > cadNo) * (firstInside - nbuffer - nCadInside <= cadNo)
            post = (lastInside + nbuffer +1 < cadNo) * (lastInside + nbuffer +1 + nCadInside >= cadNo)
            
            #outTransit[pre + post] = True
            preTransit += pre
            postTransit += post

    return inTransit, preTransit, postTransit
        


def imageShiftFFT(data, deltax, deltay, phase=0, return_abs=False, return_real=True, oversamp=2):
    """
    FFT-based sub-pixel image shift
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

    Will turn NaNs into zeros

    2014-10-07 16:58 IJMC: Copied from Adam Ginsburg's python code
    repository, currently online at:
    https://code.google.com/p/agpy/source/browse/trunk/AG_fft_tools/shift.py

    2014-10-07 17:09 IJMC: Added 'oversamp' option to reduce ringing near edges.
    """

    #fftn = np.fft.fftn
    #ifftn = np.fft.ifftn
    #fftn,ifftn = fast_ffts.get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)

    ny,nx = data.shape
    ny2 = ny * oversamp
    nx2 = nx * oversamp
    if oversamp>1:
        data = an.pad(data, ny2, nx2)

    Nx = np.fft.ifftshift(np.linspace(-np.fix(nx/2),np.ceil(nx/2)-1,nx2))
    Ny = np.fft.ifftshift(np.linspace(-np.fix(ny/2),np.ceil(ny/2)-1,ny2))
    Nx,Ny = np.meshgrid(Nx,Ny)
    gg = np.fft.ifftn( np.fft.fftn(data)* np.exp(1j*2*np.pi*(-deltax*Nx/nx-deltay*Ny/ny)) * np.exp(-1j*phase) )
    if oversamp>1:
        gg = an.pad(gg, ny, nx)
    if return_real:
        return np.real(gg)
    elif return_abs:
        return np.abs(gg)
    else:
        return gg


def shiftImages(data, posx, posy, shift='scipy1'):
    """
    :INPUTS:
      posx, posy : 1D NumPy Arrays
        Measured position of field.  Note that x is "from right to
        left" (second array index) and y is "from top to bottom"
        (primary array index). Further, note that these values are the
        negative of the 'amount to shift'.  So x=1 means that the
        image shifted to the right by 1 column, and so will be shifted
        back to the left.

      shift : str
       Mode for shifting images.
        'fft' -- use FFT-based shifting via :func:`imageShiftFFT`. Pretty noisy!

        'intshift' -- use :func:`tools.shift_image`. Only integer shifts!

        'scipy1', 'scipy3' -- use scipy.ndimage.interpolation.shift
            with N-order splines


        """

    # 2014-10-09 15:10 IJMC: Created
    # 2014-11-16 09:47 IJMC: Now can handle 2D data (single frames)
    from scipy import ndimage
    if data.ndim==2:
        data = np.reshape(data, (1, data.shape[0], data.shape[1]))
    if not hasattr(posx, '__iter__'):
        posx = np.array([posx])
    if not hasattr(posy, '__iter__'):
        posy = np.array([posy])


    nobs = data.shape[0]
    if shift.lower()=='fft':
        data = np.array([imageShiftFFT(data[ii], -posx[ii], -posy[ii]) for ii in xrange(nobs)]) 
    elif shift.lower()=='intshift':
        posxi = np.round(posx)
        posyi = np.round(posy)
        data = np.array([tools.shift_image(data[ii], -posyi[ii], -posxi[ii]) for ii in xrange(nobs)]) 
    elif shift.lower().find('scipy')==0:
        nspline = int(shift[5])
        data = np.array([ndimage.interpolation.shift(data[ii], (posy[ii], posx[ii]), order=nspline, mode='constant', cval=0) for ii in xrange(nobs)])             
    else:
        print "Unknown image-shifting method '%s', exiting." % shift
        return -1

    return data

    # 2014-10-09 15:06 IJMC: Created

def constructDiffImage(time, data, transitModel, period, edata=None, posx=None, posy=None, shift='xcshift', retall=False, empiricalErrors=False):
    """
    Construct (in-out) difference image from a set of (possibly shifted) frames.

    :INPUTS:
      time : 1D NumPy array
        Time index.

      data : 3D NumPy array
        Input target pixel file, of shape (Nobs x Nrow x Ncol).  This
        does not nede to be background-subtracted, but it might help.

      transitModel : 1D NumPy array
        Model transit light curve, equal to unity out-of-transit.

      period : positive scalar
        Planet's orbital period (for selecting in-, pre-, and
        post-transit points).

      edata : 3D NumPy array

        Uncertainties on the input 'data', of shape (Nobs x Nrow x
        Ncol).  If None, then uncertainties are instead estimated by
        'shiftedData.std(0).'

      posx, posy : None or 1D NumPy Arrays
        Measured position of field.  Note that x is "from right to
        left" (second array index) and y is "from top to bottom"
        (primary array index). Further, note that these values are the
        negative of the 'amount to shift'.  So x=1 means that the
        image shifted to the right by 1 column, and so will be shifted
        back to the left.

        If None, no shifting is done.

      shift : str or None
        Mode to use for re-shifting images.  See :func:`shiftImages`,
        input 'reg' for :func:`image_registration.register_images`, or
        use the default 'xcshift' to calculate the shifts anew via
        cross-correlation, then use scipy.ndimage.shift.

        None -- no shifting is done. With K2, this means your images
            will look *noisy*!

      empiricalErrors : bool
        If False, use the input 'edata' to estimate & propagate all
        uncertainties. If True, compute the stack-standard-deviation,
        tile it, and select the maximum of either 'edata' or this
        metric at each pixel.  For K2, 'False' inflates your
        confidence metrics.

    :RETURNS:
      diffImage, err_diffImage, (inImage, e_inImage, outImage, e_outImage)

    """
    # 2014-10-07 17:14 IJMC: Created
    # 2014-10-15 13:46 IJMC: Added option for shift='reg'
    # 2014-10-24 21:00 IJMC: Added option for shift='xcshift', using
    #                        ideas from diffimage.py
    from scipy import ndimage as nd

    nobs = time.size
    inTran, preTran, postTran = indexInOutTransit(time, transitModel, period)

    if (posx is not None and posy is not None) or shift is not None:
        xm = posx - posx.mean()
        ym = posy - posy.mean()
        if shift.lower().find('reg')==0:
            data0 = data.copy()
            dmean = data0.mean(0)
            xnew, ynew = np.zeros(nobs), np.zeros(nobs)
            for ii, frame in enumerate(data0):
                xnew[ii], ynew[ii], data[ii] = pixel_decorrelation.register_images(\
                    dmean, frame, usfac=100, return_registered=True)
                data[ii] = np.fft.fftshift(data[ii])

            #data = shiftImages(data0, xnew-xnew.mean(), ynew-ynew.mean(), shift='fft') 
        elif shift.lower()=='xcshift':
            dx,dy = pixel_decorrelation.subpix_reg_stack(data, refmode='mean')
            data = np.array([nd.shift(data[jj], [-dy[jj], -dx[jj]], order=3) for jj in range(nobs)])
            if edata is not None: edata = np.array([nd.shift(edata[jj], [-dy[jj], -dx[jj]], order=3) for jj in range(nobs)])
        else:
            data = shiftImages(data, xm, ym, shift=shift)
            if edata is not None:  edata = shiftImages(edata, xm, ym, shift=shift)


    inImage = data[inTran].sum(0) / inTran.sum()
    preImage = data[preTran].sum(0) / preTran.sum()
    postImage = data[postTran].sum(0) / postTran.sum()
    diffImage = 0.5 * (preImage + postImage) - inImage
    
    # Robust variability estimate:
    loind = int(np.round(nobs * (1. -.683) / 2.))
    hiind = int(np.round(nobs * (1. - (1. -.683) / 2.)))
    edata_empirical = np.tile(np.sort(data, axis=0)[hiind] - np.sort(data, axis=0)[loind], (nobs, 1, 1))
    if edata is None:
        edata = edata_empirical
    elif empiricalErrors:
        edata = np.array([edata, edata_empirical]).max(0)


    var_inImage = (edata[inTran]**2).sum(0) / inTran.sum()**2
    var_preImage = (edata[preTran]**2).sum(0) / preTran.sum()**2
    var_postImage = (edata[postTran]**2).sum(0) / postTran.sum()**2
    e_diffImage = np.sqrt(var_inImage + 0.25 * (var_preImage + var_postImage))
        
    ret = diffImage, e_diffImage
    if retall:
        outImage = 0.5 * (preImage + postImage)
        e_outImage = 0.5 * np.sqrt(var_preImage + var_postImage)
        ret += (inImage, np.sqrt(var_inImage), outImage, e_outImage)

    return ret


def plotDiffImages(diffImage, e_diffImage, inImage, outImage, apertureMask=None, fig=None, figpos=None, cmap=py.cm.jet, conCol='white', fontsize=14, catcut=None, epic=None, loc=None):
    """Plot results from a difference-image analysis.  All inputs are
    2D NumPy arrays.

    :EXAMPLE:
      ::

        import lightcurve_diagnostics as ld

        pixFile = '/Users/ianc/proj/transit/kepler/data/ktwo202126876-c00_lpd-targ.fits'
        photFile = '/Users/ianc/proj/transit/kepler/proc/202126876.fits'
        photPickle = '/Users/ianc/proj/transit/kepler/proc/202126876.pickle'

        P, t0, tdur = 10.3734632407, 2454833+1941.1116135904, 0.1673320053

        dat = ld.pyfits.getdata(photFile)
        pic = ld.tools.loadpickle(photPickle)
        cube, headers = ld.pixel_decorrelation.loadPixelFile(pixFile, tlimits=[dat.time.min() - 2454833 - 1e-6, np.inf])
        time, data, edata = cube['time'],cube['flux'],cube['flux_err']
        c1, c2 = dat.x, dat.y

        inTransit = np.abs((time-t0 + P/2.) % P - P/2) < (tdur/2.)
        transitModel = 1. - inTransit

        diffImage, e_diffImage, inImage, e_inImage, outImage, e_outImage = \
            ld.constructDiffImage(time, data, transitModel, P, edata=edata, \
                                   posx=c1, posy=c2, shift='xcshift', \
                                   retall=True, empiricalErrors=False)
        fig, axs, caxs = ld.plotDiffImages(diffImage, e_diffImage, inImage, outImage, apertureMask=pic.crudeApertureMask, catcut=pic.catcut, epic=pic.epic, cmap=ld.py.cm.cubehelix, conCol='r', fontsize=12)
    
    """

    # 2014-10-07 19:30 IJMC: Created
    # 2014-10-23 16:50 IJMC: Updated documentation.

    if apertureMask is None:
        apertureMask = np.ones(diffImage.shape, bool)

    imshow = pixel_decorrelation.imshow2
    #  Set up for plotting:
    if figpos is None:
        figpos = [0, 0, 1, 1]
    x0, y0, dx, dy = figpos
    wid = 0.35
    pos1 = [x0+0.05*dx, y0+0.6*dy, wid*dx,  wid*dy]
    pos2 = [x0+0.55*dx, y0+0.6*dy, wid*dx,  wid*dy]
    pos3 = [x0+0.05*dx, y0+0.1*dy, wid*dx,  wid*dy]
    pos4 = [x0+0.55*dx, y0+0.1*dy, wid*dx,  wid*dy]
    pos1c = [pos1[0]+pos1[2]*1.0, pos1[1], pos1[2]*0.05, pos1[3]]
    pos2c = [pos2[0]+pos2[2]*1.0, pos2[1], pos2[2]*0.05, pos2[3]]
    pos3c = [pos3[0]+pos3[2]*1.0, pos3[1], pos3[2]*0.05, pos3[3]]
    pos4c = [pos4[0]+pos4[2]*1.0, pos4[1], pos4[2]*0.05, pos4[3]]

    def doim(im, catcut=None, epic=None):
        if catcut is None:
            im = imshow(diffImage, cmap=cmap)
        else:
            im = pixel_decorrelation.plot_label(im,catcut,epic, colorbar=False, cmap=cmap, retim=True)
        return im

    diffSNR = (diffImage / e_diffImage)
    dats = diffImage, outImage, inImage, diffSNR

    if fig is None:
        fig = py.figure()
    ax1=fig.add_subplot(221, position=pos1)
    im1= imshow(dats[0], cmap=cmap)
    ax1.set_title('Difference Flux (e/cadence)', fontsize=fontsize)
    #py.clim([0, diffImage.max()])
    ax2=fig.add_subplot(222, position=pos2)
    #im2=imshow(dats[1], cmap=cmap)
    im2= doim(dats[1], catcut, epic)
    ax2.set_title('Out-of-Transit Flux (e/cadence)', fontsize=fontsize)
    ax3=fig.add_subplot(223, position=pos3)
    im3=imshow(dats[2], cmap=cmap)
    ax3.set_title('In-Transit Flux (e/cadence)', fontsize=fontsize)
    ax4=fig.add_subplot(224, position=pos4)
    im4=imshow(dats[3], cmap=cmap)
    ax4.set_title('Difference S/N', fontsize=fontsize)

    # A crude stopgap until we implement WCS:
    #targetPoint = ((outImage*apertureMask)==outImage[apertureMask.astype(bool)].max()).nonzero()
    if loc is None:
        targetPoint = [el[0] for el in pixel_decorrelation.getCentroidsStandard(apertureMask, mask=apertureMask)]
    else:
        targetPoint = loc

    axx = ax4.axis()
    [ax.plot(targetPoint[1], targetPoint[0], 'x', mew=2, color='r', ms=fontsize*0.6, alpha=0.5) for ax in [ax3, ax4]]
    [ax.axis(axx) for ax in [ax3, ax4]]

    # Prepare for colorbars:
    cax1 = fig.add_subplot(221, position=pos1c)
    cax2 = fig.add_subplot(221, position=pos2c)
    cax3 = fig.add_subplot(221, position=pos3c)
    cax4 = fig.add_subplot(221, position=pos4c)
    axs = [ax1, ax2, ax3, ax4]
    ims = [im1, im2, im3, im4]
    caxs = [cax1, cax2, cax3, cax4]

    for ax, im, cax, dat in zip(axs, ims, caxs, dats):
        cb = py.colorbar(im, cax=cax, format='%1.1e')
        if apertureMask is not None:
            ax.contour(apertureMask, [0.5], colors=conCol, linestyles='dotted', linewidths=2.) 
        ax.set_xlabel('Pixel Number', fontsize=fontsize*0.8)
        ax.set_ylabel('Pixel Number', fontsize=fontsize*0.8)
        all_tick_labels = ax.get_xticklabels() + ax.get_yticklabels() + \
            cax.get_xticklabels() + cax.get_yticklabels()
        [lab.set_fontsize(fontsize*0.55) for lab in all_tick_labels]
        ax.grid()
        lolim = dat[apertureMask * np.isfinite(dat)].min()
        hilim = dat[apertureMask * np.isfinite(dat)].max()
        im.set_clim([lolim, hilim])

    return fig, axs, caxs

def fitPRF2Image(image, e_image, pixfn, loc, targApDiam, ngrid=40, reterr=False, verbose=False, retfull=True):
    """Return scipy.optimize.fmin fit location of PRF to input image.

    image, e_image : 2D NumPy arrays

    pixfn : filename of pixel file (needed to load correct PRF model)

    loc : 2-sequence, index coordinates of target location

    targApDiam : scalar, diameter of target aperture

    :EXAMPLE:
      ::

        fitDiff, e_fitDiff = ld.fitPRF2Image(diffImage, e_diffImage+(1.-pic.crudeApertureMask)*9e9, pixfn, pic.loc, pic.apertures[0]*2, ngrid=40, reterr=True)
        fitOut, e_fitOut = ld.fitPRF2Image(outImage, e_outImage+(1.-pickledat.crudeApertureMask)*9e9, pixfn, pic.loc, pic.apertures[0]*2, ngrid=40, reterr=True)

        PRFmotion = fitOut - fitDiff
        sigmotion = PRFmotion / np.sqrt(e_fitOut**2 + e_fitDiff**2)

    """
    # 2014-10-08 21:06 IJMC: Created
    # 2014-11-21 09:15 IJMC: Fixed bug with zoomLevel and divide-by-zero.

    if reterr:
        maxiter = 10
    else:
        maxiter = 1

    prf, sampling = pixel_decorrelation.loadPRF(file=pixfn)
    dframe = np.round(targApDiam).astype(int)
    weights = 1./e_image**2

    gridpts1 = np.linspace(-targApDiam/1.5,targApDiam/1.5,ngrid)*sampling
    gridpts2 = np.linspace(-targApDiam/1.5,targApDiam/1.5,ngrid)*sampling

    zoomIn = True
    iter = 0
    grids = []
    xys = []
    while zoomIn and (iter < maxiter):
        iter += 1
        testgrid = phot.psffit(     prf, image, loc, weights, scale=sampling, dframe=dframe,    xoffs=gridpts1, yoffs=gridpts2, verbose=False, retvec=False, useModelCenter=True)
        redChisqGrid = testgrid[2] / testgrid[2].min()
        dof = testgrid[0].size - 4 # x, y, scaling, background
        prob = np.exp(-0.5*(dof * (redChisqGrid - redChisqGrid.min())))
        #red_chisq_1sigma = tools.invChisq(dof, .683) / dof
        grids.append(testgrid)
        xys.append((gridpts1, gridpts2))
        sigma_region_size = min((prob>an.confmap(prob, .683)).sum(), 1)
        if (sigma_region_size < (prob.size*0.1)):
            if verbose: print "Finished iteration %i, continuing..." % iter
            zoomIn = True
            if sigma_region_size==0:
                zoomLevel = 4
            else:
                zoomLevel = 1.1*np.sqrt(prob.size*0.1 / sigma_region_size)
            gridpts1 = testgrid[5] + np.linspace(-targApDiam/2.,targApDiam/2.,ngrid)*sampling/zoomLevel
            gridpts2 = testgrid[6] + np.linspace(-targApDiam/2.,targApDiam/2.,ngrid)*sampling/zoomLevel
        else:
            zoomIn = False
        #pdb.set_trace()

    if reterr:
        cum1 = np.cumsum(prob.sum(0) / prob.sum())
        cum2 = np.cumsum(prob.sum(1) / prob.sum())
        lolim1 = (np.abs(cum1 - 0.158) == np.abs(cum1 - 0.158).min()).nonzero()[0][0]
        lolim2 = (np.abs(cum2 - 0.158) == np.abs(cum2 - 0.158).min()).nonzero()[0][0]
        hilim1 = (np.abs(cum1 - 0.842) == np.abs(cum1 - 0.842).min()).nonzero()[0][0]
        hilim2 = (np.abs(cum2 - 0.842) == np.abs(cum2 - 0.842).min()).nonzero()[0][0]
        err1 = np.diff(gridpts1).mean() * (hilim1 - lolim1)/2. 
        err2 = np.diff(gridpts2).mean() * (hilim2 - lolim2)/2. 

    #pdb.set_trace()

    guess = testgrid[5:7]
    fitargs = (prf, image, weights, sampling, dframe, loc, False, False, True)
    fit = an.fmin(phot.psffiterr, guess, args=fitargs, xtol=0.5, ftol=0.1, full_output=True, nonzdelt=1)

    
    #modelpsf, data, chisq, background, fluxscale, xoffset, yoffset, xoffs, yoffs, chisq[ii,jj],background[ii,jj], fluxscale[ii,jj]
    #nmc = 100
    #mc_data = np.random.normal(image, e_image, size=(nmc,)+image.shape)
    #mc_fits = np.zeros((nmc, fit[0].size), dtype=float)
    #for ii in xrange(nmc):
    #    these_fitargs = (prf, mc_data[ii], weights, sampling, dframe, loc, False, False, True)
    #    mc_fits[ii] = an.fmin(phot.psffiterr, fit[0], args=these_fitargs, xtol=0.5, ftol=0.1, full_output=False, nonzdelt=1)

    #if mcmc:
    #    print "It's broke, it won't work yet!"##
    #
    #    pdb.set_trace()
    #
    #    #import emcee
    #    #import phasecurves as pc
    #    nwalkers = 10
    #    chicut = np.sort(testgrid[2].ravel())[nwalkers]
    #    chiinds = (testgrid[2]<chicut).nonzero()
    #    pos0 = np.vstack((testgrid[-4][chiinds[0]], testgrid[-5][chiinds[1]])).T
    #    pos0[-1] = fit[0]
    #    fits = [an.fmin(phot.psffiterr, par, args=fitargs, xtol=0.5, ftol=0.1, full_output=True, nonzdelt=1) for par in pos0]
    #    #pos0, chi0 = tools.get_emcee_start(bestparams, var_bestparams, nwalkers, 1.1*nobs, jfitargs, homein=True, retchisq=True)
    #    pdb.set_trace()
    #
    #    #mcargs = (phot.psffiterr, args=fitargs, xtol=0.5, ftol=0.1, full_output=True, nonzdelt=1)
    #    #this_model = jfitargs[0](bestparams[1:], *jfitargs[1:-3])
    #    
    #    #sampler = emcee.EnsembleSampler(nwalkers, 2, pc.lnprobfunc, args=fitargs) #, pool=pool) #threads=nthreads)
    #    #pos1, prob1, state1 = sampler.run_mcmc(pos0, nstep)


    ret = fit[0]/sampling
    if retfull:
        mod = phot.psffit(prf, image, loc, weights, scale=sampling, dframe=dframe, xoffs=[fit[0][0]], yoffs=[fit[0][1]], verbose=verbose)
        ret = (ret, np.array([err1, err2])/sampling, mod)
    elif reterr:
        ret = (ret, np.array([err1, err2])/sampling)

    return ret

def plotPixelTransit(*args, **kwargs):
    """
    Plot results of pixel-flux vs. transit tests (from :func:`computePixelTransit)

    :INPUTS:
      input : pickle object output from our photometry calibratios

      data : 3D NumPy array
        Pixel data.

      transitModel : 1D NumPy array
        Model light curve, equal to unity out-of-transit.

    :RETURNS:
       gammas, e_gammas -- 2D NumPy Arrays
       """
    # 2014-10-09 16:21 IJMC: Created

    # Parse Inputs:
    if 'fontsize' in kwargs:
        fontsize = kwargs.pop('fontsize')
    else:
        fontsize = 14
    if 'medfilt' in kwargs:
        medfilt = kwargs.pop('medfilt')
    else:
        medfilt = 3
    if 'figpos' in kwargs:
        figpos = kwargs.pop('figpos')
    else:
        figpos = [0,0,1,1]
    if 'fig' in kwargs:
        fig = kwargs.pop('fig')
    else:
        fig = py.figure()


    input = args[0]
    # Compute pixel-correlation image:
    gammas, egammas = computePixelTransit(*args, **kwargs)

    # Set up for plotting:
    x0, y0, dx, dy = figpos
    pos1 = [x0+0.1*dx, y0+0.08*dy, 0.8*dx,  0.86*dy]
    pos1c = [x0+0.85*dx, y0+0.12*dy, 0.05*dx,  0.86*dy]

    #fGamma = signal.medfilt2d(gammas, medfilt)
    fGamma = gammas / egammas
    lolim = fGamma[input.crudeApertureMask * np.isfinite(fGamma)].min()
    hilim = fGamma[input.crudeApertureMask * np.isfinite(fGamma)].max()

    ax = fig.add_subplot(111, position=pos1)
    im = pixel_decorrelation.plot_label(fGamma, input.catcut, input.epic, retim=True, colorbar=False)

    cax = fig.add_subplot(221, position=pos1c)
    cb = py.colorbar(im, cax=cax) #, format='%1.1e')

    #cb = py.colorbar(im, ax=ax)
    im.set_clim([0, hilim])
    ax.contour(input.crudeApertureMask, [0.5], linewidths=2, colors='g')
    
    all_tick_labels = ax.get_xticklabels() + ax.get_yticklabels() + \
        cax.get_xticklabels() + cax.get_yticklabels()
    [lab.set_fontsize(fontsize*0.8) for lab in all_tick_labels]
    ax.set_title('Pixel/Transit Correlation', fontsize=fontsize)

    return fig, ax


def plotDIA_PRF_fit(pixFile, input, diffImage, e_diffImage, outImage, e_outImage, ngrid=40, fig=None, figpos=None, cmap=py.cm.jet, fontsize=14, minApWidth=7):
    """Fit PRFs to difference & o.o.t. images, and plot results.

    Mainly a helper function -- and a pretty slow one, too!
    """
    # 2014-10-24 15:21 IJMC: Created

    imshow = pixel_decorrelation.imshow2

    if fig is None:
        fig = py.figure()

    if figpos is None:
        figpos = [0, 0, 1, 1]
    x0, y0, dx, dy = figpos
    pos1 = [x0+0.03*dx, y0+0.55*dy, 0.35*dx,  0.35*dy]
    pos2 = [x0+0.38*dx, y0+0.55*dy, 0.35*dx, 0.35*dy]
    pos3 = [x0+0.03*dx, y0+0.1*dy, 0.35*dx, 0.35*dy]
    pos4 = [x0+0.38*dx, y0+0.1*dy, 0.35*dx, 0.35*dy]
    pos5 = [x0+0.72*dx, y0+0.1*dy, 0.26*dx, 0.8*dy]

    prfFitApertureWidth = max(minApWidth, input.apertures[0]*2)
    fitDiff, e_fitDiff, modDiff = fitPRF2Image(diffImage, e_diffImage+(1.-input.crudeApertureMask)*9e9, pixFile, input.loc, prfFitApertureWidth, ngrid=ngrid, reterr=True)
    fitOut, e_fitOut, modOut = fitPRF2Image(outImage, e_outImage+(1.-input.crudeApertureMask)*9e9, pixFile, input.loc, prfFitApertureWidth, ngrid=ngrid, reterr=True)
    #fitIn, e_fitIn, modIn = fitPRF2Image(inImage, e_inImage+(1.-input.crudeApertureMask)*9e9, pixFile, input.loc, prfFitApertureWidth, ngrid=ngrid, reterr=True)

    PRFmotion = fitOut - fitDiff
    e_PRFmotion = np.sqrt(e_fitOut**2 + e_fitDiff**2)
    pm_metric = np.sqrt(((PRFmotion / e_PRFmotion)**2).sum())
    prf_diff_text = ['','PRF-fitting D.I.A.', '$\Delta$ ax1 =', '%1.3f +\- %1.3f pix' % (PRFmotion[0], e_PRFmotion[0]), '$\Delta$ ax2 =', '%1.3f +\- %1.3f pix' % (PRFmotion[1], e_PRFmotion[1]), '', 'Metric = %1.1f' % pm_metric, '']

    dats = modOut[1], modOut[1]-modOut[0], modDiff[1], modDiff[1]-modDiff[0]
    ax1 = fig.add_subplot(111, position=pos1)
    im1 = imshow(dats[0], cmap=cmap)
    ax1.set_title('Out-of-Transit', fontsize=fontsize)
    ax2 = fig.add_subplot(111, position=pos2)
    im2 = imshow(dats[1], cmap=cmap)
    ax2.set_title('O.O.T. Residuals', fontsize=fontsize)
    ax3 = fig.add_subplot(111, position=pos3)
    im3 = imshow(dats[2], cmap=cmap)
    ax3.set_title('Difference Image', fontsize=fontsize)
    ax4 = fig.add_subplot(111, position=pos4)
    im4 = imshow(dats[3], cmap=cmap)
    ax4.set_title('D.I. Residuals', fontsize=fontsize)
    ax5 = fig.add_subplot(111, position=pos5)
    tools.textfig(prf_diff_text, ax=ax5, fig=fig, fontsize=fontsize*0.8)

    oot_lolim = dats[1][np.isfinite(dats[1])].min()
    oot_hilim = dats[1][np.isfinite(dats[1])].max()
    [im.set_clim([oot_lolim, oot_hilim]) for im in [im1, im2]]
    dif_lolim = dats[3][np.isfinite(dats[3])].min()
    dif_hilim = dats[3][np.isfinite(dats[3])].max()
    [im.set_clim([dif_lolim, dif_hilim]) for im in [im3, im4]]

    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        #cb = py.colorbar(im, cax=cax, format='%1.1e')
        #if apertureMask is not None:
            #ax.contour(apertureMask, [0.5], colors=conCol, linestyles='dotted', linewidths=2.) 
        #ax.set_xlabel('Pixel Number', fontsize=fontsize*0.8)
        #ax.set_ylabel('Pixel Number', fontsize=fontsize*0.8)
        all_tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [lab.set_fontsize(fontsize*0.55) for lab in all_tick_labels]
        ax.grid()
        #lolim = dat[np.isfinite(dat)].min()
        #hilim = dat[np.isfinite(dat)].max()
        #im.set_clim([lolim, hilim])
        ax.minorticks_on()

    return fig, axs

if __name__=='__main__':
    from argparse import ArgumentParser
    from pixel_decorrelation import baseObject
    p = ArgumentParser()
    p.add_argument('lcFits',type=str)
    p.add_argument('lcPickle',type=str)
    p.add_argument('pixFile',type=str)
    p.add_argument('candcsv',type=str)
    args = p.parse_args()
    
    fig,axs = plotDiagnostics_wrap(
        args.lcFits, args.lcPickle, args.pixFile, args.candcsv)
    pathpdf = args.lcFits.replace('.fits','.pdf')
    pathpng = args.lcFits.replace('.fits','.png')

    fig.savefig(pathpdf)
    fig.savefig(pathpng)

