"""
Plotting functions for pipeline object
"""

from .config import *

def noise_vs_aperture_size(pipe):
    dfaper = pipe.dfaper
    dmin = dfaper.sort('noise').iloc[0]
    dfaper = dfaper.sort('npix')
    
    plt.semilogy()
    plt.plot(dfaper['npix'],dfaper['noise'],'o-')
    plt.plot(dmin['npix'],dmin['noise'],'or',mfc='none',ms=15,mew=2,mec='r')

    xlab = 'Target Aperture Size [pixels]'
    txtStr = 'Minimum: %.1f ppm at R=%.1f pixels' % \
             (dmin['noise'],dmin['npix'])

    plt.xlabel(xlab)
    plt.ylabel('Noise [ppm]')
    plt.minorticks_on()

    noise_min = dfaper.noise.min()
    noise_max = dfaper.noise.max()
    desc = dfaper['noise'].describe()
    factor = 1.2
    yval = noise_min/factor,noise_max * factor
    plt.ylim(yval)
    yticks = np.logspace(np.log10(yval[0]), np.log10(yval[1]), 8)
    plt.yticks(yticks, ['%i' % el for el in yticks])
    plt.minorticks_off()
    tit = pipe.name_mag()
    plt.title(tit)
