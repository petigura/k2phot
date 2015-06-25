from numpy import ma
import plotting
import pandas as pd
import ses 
import config

class Lightcurve(pd.DataFrame):
    def get_fm(self,col,maskcol='fmask'):
        fm = ma.masked_array(self[col],self[maskcol])
        return fm
    def get_ses(self,col,maskcol='fmask'):
        fm = self.get_fm(col,maskcol=maskcol)
        fm = ma.masked_invalid(fm)
        dfses = ses.ses_stats(fm)
        return dfses


class Normalizer:
    def __init__(self,xmed):
        self.xmed = xmed
    def norm(self,x):
        return x / self.xmed - 1
    def unnorm(self,x):
        return (x + 1) * self.xmed
