from argparse import ArgumentParser
import pandas as pd
from jinja2 import Environment, PackageLoader, FileSystemLoader
import k2phot.phot
import os
import k2phot.plotting
import numpy as np
from matplotlib import pylab as plt

K2_ARCHIVE=os.environ['K2_ARCHIVE']

def cfop(run, starname):
    os.path.join(K2_ARCHIVE,'photometry/',run,'')
    outdir = '{}/photometry/{}/output/{}'.format(K2_ARCHIVE, run, starname)

    starname = outdir.split('/')[-1]
    fitsfn = "{}/{}.fits".format(outdir, starname)
    phot = k2phot.phot.read_fits(fitsfn,'optimum')
    lcout = phot.lc['t cad fsap fbg fdt_t_rollmed fdt_t_roll_2D bgmask thrustermask fdtmask fmask'.split()]

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

def main():
    psr = ArgumentParser()
    psr.add_argument('run',type=str)
    psr.add_argument('starlist',type=str)
    psr.add_argument('njobs',type=int)
    psr.add_argument('jobid',type=int)

    args = psr.parse_args()
    stars = pd.read_csv(args.starlist,names=['starname'])
    idxL = np.array_split(stars.index, args.njobs)
    idxL = idxL[args.jobid]
    nstars = len(idxL)
    print "cfop: creating CFOP plots for {}".format(nstars)
    for idx in idxL:
        starname = stars.ix[idx].starname
        try: 
            cfop(args.run, starname)
            print "cfop: created plots {}, {}".format(idx, starname)
        except IOError:
            print "cfop: failed {}, {}".format(idx, starname)
            pass
        
if __name__=='__main__':
    main()


