
 #########


def lightcurve_segments(lc0):
    nseg = 8
    nrows = 4
    fig = plt.figure(figsize=(18,10))
    gs = GridSpec(nrows,nseg)
    lc_segments = np.array_split(lc0,nseg)
    plt.rc('lines',markersize=6,markeredgewidth=0)
    def plot(*args,**kwargs):
        plt.plot(*args,alpha=0.5,**kwargs)

    for i in range(nseg):
        if i==0:
            ax0L = [fig.add_subplot(gs[j,0]) for j in range(nrows)]
            axiL = ax0L
        else:
            axiL = [
                fig.add_subplot(gs[0,i],sharey=ax0L[0]),
                fig.add_subplot(gs[1,i],sharey=ax0L[1]),
                fig.add_subplot(gs[2,i],sharey=ax0L[2]),
                fig.add_subplot(gs[3,i],sharex=ax0L[3],sharey=ax0L[3]),
            ]

            for ax in axiL:
                plt.setp(
                    ax.yaxis,
                    visible=False,
                    major_locator=MaxNLocator(3)
                )
                plt.setp(
                    ax.xaxis,
                    visible=False,
                    major_locator=MaxNLocator(3)
                )

        lc = lc_segments[i]
        lc = lightcurve.Lightcurve(lc)
        fm = lc.get_fm('fsap')
        ftnd = lc.get_fm('ftnd_t_roll_2D')
        
        plt.sca(axiL[0])
        plot(lc['t'],lc['roll'],label='x: t, y: roll')

        plt.sca(axiL[1])
        plot(lc['roll'],fm,'.',label='x: roll, y: flux')
        plot(lc['roll'],ftnd,'.',label='x: roll, y: flux model')

        plt.sca(axiL[2])
        plot(lc['t'],lc['roll'],label='x: t, y: roll')

        plt.sca(axiL[3])
        
        xpr = lc.get_fm('xpr')
        ypr = lc.get_fm('ypr')
        plot(xpr,ypr,'.',label='x: xpr, y: yrp')

    for i in range(nrows):
        plt.sca(ax0L[i])
        plt.legend(fontsize='x-small')
    fig.subplots_adjust(
        left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0.001, wspace=0.001
    )


def diag(dv,tpar=False):
    """
    Print a 1-page diagnostic plot of a given h5.
    
    Right now, we recompute the single transit statistics on the
    fly. By default, we show the highest SNR event. We can fold on
    arbitrary ephmeris by setting the tpar keyword.

    Parameters
    ----------
    h5   : h5 file after going through terra.dv
    tpar : Dictionary with alternate ephemeris specified by:
           Pcad - Period [cadences] (float)
           t0   - transit epoch [days] (float)
           twd  - width of transit [cadences]
           mean - depth of transit [float]
           noise 
           s2n
    """

    # Top row
    axPeriodogram  = fig.add_subplot(gs[0,0:8])
    axAutoCorr = fig.add_subplot(gs[0,8])

    # Second row
    axPF       = fig.add_subplot(gs[1,0:2])
    axPFzoom   = fig.add_subplot(gs[1,2:4],sharex=axPF,)
    axPF180    = fig.add_subplot(gs[1,4:6],sharex=axPF)
    axPFSec    = fig.add_subplot(gs[1,6:8],sharex=axPF)
    axSingSES  = fig.add_subplot(gs[1,-2])

    # Last row
    axStack        = fig.add_subplot(gs[2:8 ,0:8])
    axStackZoom    = fig.add_subplot(gs[2:8 ,8:])

    # Top row
    sca(axPeriodogram)
    periodogram(dv)

    sca(axAutoCorr)
    autocorr(dv)
    AddAnchored("ACF",prop=tprop,frameon=True,loc=2)    

