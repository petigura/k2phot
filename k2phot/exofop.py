# Generate plots for the ExoFOP
import os

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from jinja2 import Environment, PackageLoader, FileSystemLoader

import k2phot.phot
import k2phot.plotting

K2_ARCHIVE = os.environ['K2_ARCHIVE']

def exofop(run, starname, plots=False, lc=True):
    os.path.join(K2_ARCHIVE,'photometry/',run,'')
    outdir = '{}/photometry/{}/output/{}'.format(K2_ARCHIVE, run, starname)

    starname = outdir.split('/')[-1]
    fitsfn = "{}/{}.fits".format(outdir, starname)
    phot = k2phot.phot.read_fits(fitsfn,'optimum')
    lcout = phot.lc['t cad fsap fbg fdt_t_rollmed fdt_t_roll_2D bgmask thrustermask fdtmask fmask'.split()]

    if lc:
        env = Environment(loader=PackageLoader('k2phot', 'templates'))
        template = env.get_template('lightcurve-csv-header.csv')
        npix = phot.ap_weights.sum()
        starname = phot.header['KEPLERID']
        csvfn = "{}/{}.csv".format(outdir,starname)

        with open(csvfn,'w') as f:
            s = template.render(npix=npix,epic=starname)
            s += '\n' # need to add another return
            f.writelines(s)
            lcout.to_csv(f,index=False,float_format="%.6f")

        print "cfop: created: {}".format(csvfn)
            
    if plots:
        pnglist = []
        pngfn = "{}/{}_0-aperture.png".format(outdir,starname)
        pnglist.append(pngfn)
        k2phot.plotting.phot.aperture(phot)
        plt.gcf().savefig(pngfn,dpi=160)
        plt.close()

        pngfn = "{}/{}_5-fdt_t_rollmed.png".format(outdir,starname)
        pnglist.append(pngfn)
        k2phot.plotting.phot.detrend_t_rollmed(phot)
        plt.gcf().savefig(pngfn,dpi=160)
        plt.close()

        pdffn = "{}/{}.pdf".format(outdir,starname)
        files = pnglist + [pdffn]
        cmd = 'convert ' + " ".join(files)
        print cmd
        os.system(cmd)
        print "cfop: created: {}".format(pdffn)
