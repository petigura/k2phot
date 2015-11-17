#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
from k2phot import pipeline

if __name__ == "__main__":
    p = ArgumentParser(description='Pixel Decorrelation')
    p.add_argument('pixfile',type=str)
    p.add_argument('lcfile',type=str)
    p.add_argument('transfile',type=str)
    p.add_argument('--debug',action='store_true',help='run in debug mode?')
    p.add_argument(
        '--tmin', type=float, default=-np.inf,help='Minimum valid time index'
    )
    p.add_argument(
        '--tmax', type=float, default=np.inf,help='Max time'
    )
    p.add_argument(
        '--atmin', type=float, default=-np.inf,help='Minimum valid time index'
    )
    p.add_argument(
        '--atmax', type=float, default=np.inf,help='Max time'
    )

    p.add_argument(
        '--tex', type=str, default=None,help='Exclude time range'
    )

    args  = p.parse_args()
    tex = args.tex
    if tex=='':
        tex=None
    elif type(args.tex)!=type(None):
        tex = eval("np.array(%s)" % tex)

    tlimits = [args.tmin,args.tmax]
    ap_select_tlimits = [args.atmin,args.atmax]
    pipeline.pipeline(
        args.pixfile, args.lcfile, args.transfile, debug=args.debug,
        tlimits=tlimits, tex=tex, ap_select_tlimits = ap_select_tlimits
        )

