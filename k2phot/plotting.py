
 #########



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

