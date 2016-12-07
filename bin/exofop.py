#!/usr/bin/env python 
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
import pandas as pd
from jinja2 import Environment, PackageLoader, FileSystemLoader
import k2phot.phot
import os
import k2phot.plotting
import numpy as np
from matplotlib import pylab as plt

import k2phot.exofop

def main():
    psr = ArgumentParser()
    psr.add_argument('run',type=str)
    psr.add_argument('starlist',type=str)
    psr.add_argument('njobs',type=int)
    psr.add_argument('jobid',type=int)
    psr.add_argument('--plots', action='store_true')

    args = psr.parse_args()
    stars = pd.read_csv(args.starlist,names=['starname'])
    idxL = np.array_split(stars.index, args.njobs)
    idxL = idxL[args.jobid]
    nstars = len(idxL)
    print "cfop: creating CFOP plots for {}".format(nstars)
    for idx in idxL:
        starname = stars.ix[idx].starname
        try: 
            exofop.exofop(args.run, starname, plots=args.plots)
            print "cfop: created plots {}, {}".format(idx, starname)
        except IOError:
            print "cfop: failed {}, {}".format(idx, starname)
            pass
        
if __name__=='__main__':
    main()


