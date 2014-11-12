


import os
path_pointing = os.path.join(os.environ['K2PHOTFILES'],'c0_p2_pointing.txt')
from astropy.time import Time
pntcad = 5./60/24


def parse_time(x):
    """
    x : Pandas series with date and time keys
    """
    hour,minute = x['time'].split(':')
    month,day,year = x['date'].split('/')
    year = '20' + year
    utc = '%s-%s-%s %s:%s:00' % (year,month,day,hour,minute)
    t = Time(utc,scale='utc')
    return t

def read_pointing():
    pnt = pd.read_csv(path_pointing,sep='\s*',skiprows=2,
                      names='date time theta y z'.split())    
    
    pnt['time2'] = pnt.apply(parse_time,axis=1)    
    pnt['jd'] = pnt.time2.apply(lambda x : x.jd)

    theta = np.array(pnt.theta)
    dtheta = (theta[1:] - theta[:-1]) / pntcad
    dtheta = np.append(dtheta,[np.nan])
    pnt['dtheta'] = dtheta
    return pnt

def mask_pointing(pnt):
    """
    Mask out troublesome cadences
    """
    dtheta = np.array(pnt.dtheta)
    mad = np.median(np.abs(dtheta))
    pmask = np.abs(dtheta) > 10*mad
    pmask[-1] = True

    pmask = pmask | ~pnt.theta.between(-0.0002,0.00005)
    print "%i/%i pointings masked out" % (pmask.sum(), pmask.size)
    pnt['pmask'] = pmask

    return pnt

def merge_lc_pointing(lc,pnt):
    """
    Downsample pointing values onto the long cadence light curve
    """
    # Generate buckets to place pointing values into
    long_cad = 30./60./24. # Long cadence in days
    lc['t_cad_start'] = lc['t'] - 0.5 * long_cad
    lc['t_cad_stop'] = lc['t'] + 0.5 * long_cad
    bins = lc.t_cad_start.tolist() + [lc.t_cad_stop.iloc[-1]]

    # Take the mean of the pointing value in each cadence
    idx = np.digitize(pnt.jd,bins)
    idx -= 1 # Weird indexing error
    g = pnt.groupby(idx)
    pntmean = g.mean()

    # Merge into lc array. 
    # If any pointing value is masked, the lc is masked.
    lc.index = range(len(lc))
    lc2 = pd.merge(lc,pntmean,left_index=True,right_index=True)
    lc2['pmask'] = lc2.pmask > 0 
    return lc2

from pixel_decorrelation2 import lc_to_X
from sklearn.gaussian_process import GaussianProcess
from pixel_decorrelation2 import fmad

def decorrelate_theta(lc,plot=True):
    """
    Decorrelate light curve against theta

    Parameters
    ----------
    lc : DataFrame. Light curve
    """
    
    # Bin up the points by roll angel
    nbins = 10
    d = lc[~lc.pmask].theta.describe()
    thetabins = np.linspace(d['min'],d['max'],nbins+1)
    g = lc.groupby(pd.cut(lc.theta,thetabins))
    lcb = g.median()
    lcb['ferr'] = g['f'].apply(fmad) / np.sqrt(g['f'].count())

    # Fit a GP to flux as a function of roll angle
    gp = GaussianProcess(nugget=lcb.ferr**2,thetaL=0.001,thetaU=0.1,regr='constant')
    X = lc_to_X(lcb,['theta'])
    gp = gp.fit(X,lcb.f)

    # Fit out the dependency on theta
    ftnd = gp.predict(lc_to_X(lc,['theta']))

    lc['f_gp_theta'] = ftnd
    lc['fdt_theta'] = lc['f'] - lc['f_gp_theta']

    if plot==True:
        fig,ax = plt.subplots()
        thetai = np.linspace(d['min'],d['max'],100)
        ftndi = gp.predict(thetai.reshape(-1,1))
        plt.xlim(d['min'],d['max'])
        plt.ylim(*np.sort(ftndi)[[0,-1]])
        plt.plot(thetai,ftndi,lw=2)

        time = lc.t - lc.t.min()
        plt.scatter(lc.theta,lc.f,c=time,lw=0,s=8)
        plt.errorbar(lcb.theta,lcb.f,yerr=lcb.ferr,fmt='o')
        plt.colorbar()

        #
        fig,axL = plt.subplots(nrows=2,sharex=True,sharey=True)
        plt.sca(axL[0])
        f = ma.masked_array(lc.f,lc.pmask)
        ftnd = ma.masked_array(lc.f_gp_theta,lc.pmask)
        fdt = ma.masked_array(lc.fdt_theta,lc.pmask)

        plt.plot(lc.t,f)
        plt.plot(lc.t,ftnd)

        plt.sca(axL[1])
        plt.plot(lc.t,fdt)

    return lc

def plot_pointing(lc,pnt):
    """
    Diagnostic plot to make sure pointing was incorporated correctly.
    """

    def plot(x,y,**kwargs):
        plt.plot(x,y.data)
        plt.plot(x,y,label='Masked')

    fig,axL = plt.subplots(nrows=3,sharex=True,figsize=(12,8))
    plt.sca(axL[0])
    fm = ma.masked_array(lc.f,lc.pmask)
    plot(lc.t,fm)
    plt.title('Flux')
    plt.legend()

    plt.sca(axL[1])
    thetam = ma.masked_array(pnt.theta,pnt.pmask)
    thetamlc = ma.masked_array(lc.theta,lc.pmask)
    plot(pnt.jd,thetam)
    plt.plot(lc.t,thetamlc,label='Down Sampled')
    plt.ylabel('$\Theta$ [Deg]')
    plt.yscale('symlog',linthreshy=0.0003)
    plt.title('Roll Angle')
    plt.legend()

    plt.sca(axL[2])
    dthetam = ma.masked_array(pnt.dtheta,pnt.pmask)
    plot(pnt.jd,dthetam)
    plt.ylabel('$d\Theta$ [Deg/day]')
    plt.yscale('symlog',linthreshy=0.003)
    plt.title('Derivative of Roll Angle')
    fig.set_tight_layout(True)
    plt.legend()


    

