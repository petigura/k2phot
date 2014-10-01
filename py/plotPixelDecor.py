from argparse import ArgumentParser
from pixel_decorrelation import plotPixelDecorResults,baseObject
import cPickle as pickle
import tools
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *

parser = ArgumentParser()
parser.add_argument('picklefile',nargs='+',type=str,
                    help='list of pickle files to process')
args = parser.parse_args()

picklefile = args.picklefile

def plot_picklefile(picklefile):
    with open(picklefile,'r') as f:
        o = pickle.load(f)

    plotPixelDecorResults(o)
    pdffn = picklefile.replace('.pickle','.pdf')
    tools.printfigs(pdffn, pdfmode='gs')
    close('all')


for picklefn in picklefile:
    plot_picklefile(picklefn)
