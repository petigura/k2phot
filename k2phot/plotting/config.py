import matplotlib.pylab as plt
import numpy as np
from numpy import ma

from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import MaxNLocator
import matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import contextlib

pntskw = dict(marker='.',linestyle='-',alpha=0.8,mew=0,ms=5,mfc='RoyalBlue',color='RoyalBlue')
legkw = dict(frameon=False,fontsize='x-small',loc='lower left')

def imshow(im,**kwargs):
    """
    """
    extent = None
    if kwargs.has_key('cmap')==False or kwargs['cmap'] is None:
        kwargs['cmap'] = plt.cm.gray 

    im = plt.imshow(
        im, interpolation='nearest', origin='lower', extent=extent, **kwargs
        )

    return im
