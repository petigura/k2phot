import h5py
import numpy as np
import os
from matplotlib import mlab
import sys
#import pyfits

class File(h5py.File):
   def __init__(self,name,mode=None,driver=None,**kwds):
      """
      Simple extension to h5py.File

      Additional mode 'c' will 

      """
      if mode == 'c':
         if os.path.exists(name):
            os.remove(name)
            print 'removing %s ' % name
         mode = 'a'

      h5py.File.__init__(self,name,mode=mode,driver=driver,**kwds)
      
   def __setitem__(self,key,val):
      if self.keys().count(unicode(key)) is 1:
         print "removing %s " % key
         del self[key]      
      h5py.File.__setitem__(self,key,val)

   def create_group(self,name):
      try:
         group = h5py.File.create_group(self,name)
      except ValueError:
         print "removing %s " % name
         del self[name]
         group = h5py.File.create_group(self,name)
      return group

   def group2dict(self,name):
      """
      Retrive scalar datasets in a group.
      """
      d = {}
      for n in self[name].keys():
         d[n] = self[name][n][()]
      return d

   def dict2group(self,name,d):
      """
      Take all the values in a dictionary and shove them into a group
      """
      for n in d.keys():
         self[name][n] = d[n]

class iohelper(object):
   """
   Simple class from which to build other objects. 
   
   If we create an attribute we wish to save, use the add_attr function 
   If we create a data set we wish to save, use the add_dset function 
   """

   def __init__(self):
      self.attrs = {} # 
      self.attrs_keys = [] # List of attributes to store as h5 attrs
      self.attrs_desc = [] # Description of h5 attrs

      self.dset_keys = [] # List of attributes to store as h5 datasets
      self.dset_desc = [] # Description of h5 datasets

   # 
   # Book keeping functions
   #
   def add_attr(self,name,value,description=''):
      """
      Sets an attribute of DV object and records that we want this
      attribute saved.
      """
      setattr(self,name,value)
      self.attrs_keys.append(name)
      self.attrs_desc.append(description)

   def add_dset(self,name,value,description=''):
      """
      Sets an attribute of DV object and records that we want this
      attribute saved as a h5 dataset.
      """
      setattr(self,name,value)
      self.dset_keys.append(name)
      self.dset_desc.append(description)

   def get_valuedesc(self,k,keys_list,desc_list):
      """
      Get key and description
      """
      value = getattr(self,k)
      i = keys_list.index(k)
      desc = desc_list[i]
      return value,desc

   def get_attrs_dict(self):
      # Just return a dictionary of all the attributes
      return dict([(k,getattr(self,k)) for k in self.attrs_keys])

   def to_hdf(self,h5file,group):
      with File(h5file) as h5:
         group = h5.create_group(group)

         for k in self.attrs_keys:
            value,desc = self.get_valuedesc(
               k,self.attrs_keys,self.attrs_desc)
            group.attrs[k] = value
            group.attrs[k+'_description'] = desc

         for k in self.dset_keys:
            value,desc = self.get_valuedesc(
               k,self.dset_keys,self.dset_desc)
            group[k] = value
            group[k].attrs['description'] = desc

         group.attrs['attrs_keys'] = self.attrs_keys
         group.attrs['dset_keys'] = self.dset_keys

         group.attrs['attrs_desc'] = self.attrs_desc
         group.attrs['dset_desc'] = self.dset_desc
      

def read_iohelper(h5file,group):
   io = iohelper()
   
   with h5py.File(h5file,'r') as h5:
      group = h5[group]

      io.attrs_keys = group.attrs['attrs_keys']
      io.attrs_desc = group.attrs['attrs_desc']

      io.dset_keys = group.attrs['dset_keys']
      io.dset_desc = group.attrs['dset_desc']

      for k in io.attrs_keys:
         setattr(io,k,group.attrs[k])

      for k in io.dset_keys:
         setattr(io,k,group[k][:])

   return io

def add_attrs(h5,d):
   """
   Add elements of a dictionary as attributes to h5 file.
   """
   for k in d.keys():
      h5.attrs[k] = d[k]

def compChunks(elsize,ncolmax):
   """
   Compute Chunck Size
   """

   # Compute chunksize.
   csize     = 300e3 # Target size uncompressed size for the chunks.
   crowsize = 100
   ccolsize  = int(csize/elsize/crowsize)
   ccolsize  = min(ccolsize,ncolmax)
   chunks = (ccolsize,crowsize)
   return chunks

def atpy2h5(files,out,diff='all',name='ds'):
   """
   atpy format to h5

   Parameters
   ----------

   inp  : globable string specifying where the input files are
   out  : output h5 file.  In none exists, we create it.

   diff : List of fields that are stored as stacked arrays.  Those
          that are not different, we store the first element.
   """
   nfiles = len(files)
   t0 = atpy.Table(files[0])
   h5 = File(out)
   ds,ds1d = diffDS(t0.table_name,t0.data.dtype,(nfiles,t0.data.size)
                    ,h5,diff=diff)
   
   kicL = []   
   nFail = 0 
#   import pdb;pdb.set_trace()
   for i in range(nfiles):
      if np.mod(i,100)==0:
         print i
      try:
         hdu = pyfits.open(files[i])
         data = hdu[1].data
         kic  = hdu[1].header['KEPLERID']
         assert type(kic) == int
         kicL.append(kic)
         
         if diff!='all':
            data = mlab.rec_keep_fields(data,diff)
            ds1d[:] =  mlab.rec_drop_fields(data,diff)

         ds[i-nFail] = data
   
      except:
         print sys.exc_info()[1]
         nFail +=1
         
   ds.resize(ds.shape[0]-nFail,axis=0)
   kicL = np.array(kicL)
   h5.create_dataset('KIC',data=kicL)
   print "%i files failed" % nFail
   h5.close()

def diffDS(name,dtype,shape,h5,diff='all'):
   """
   Create Datasets

   Try to chunk it up into 300K sizes.

   Parameters
   ----------
   name   : Table name
   dtype  : Element data type
   shape  : 2D array shape (nLightCurves, nCadences)
   f      : h5 file handle
   """
   nrows,ncols = shape

   names = list(dtype.names)
   if diff != 'all':
      [names.remove(d)  for d in diff]
      dnames = diff
      
      sstr = "%s " % ', '.join(map(str,names))
      dstr = "%s " % ', '.join(map(str,dnames))

      print """
Same Columns
------------
%s

Diff Columns
------------
%s 

""" %(sstr,dstr )  

      sdtype = np.dtype([ (n,dtype[n]) for n in names])
      ds1d     = h5.create_dataset(name+'1d',shape=(ncols,),dtype=sdtype)
   else:
      ds1d = None
      dnames = names

   ddtype = np.dtype([ (n,dtype[n]) for n in dnames])


   # There chunksize cannot be bigger than shape (there should be a
   # more explicit warning for this.

   chunks = compChunks(dtype.itemsize,nrows)
   chunks = min(chunks[0],ncols),chunks[1]
   ds = h5.create_dataset(name,shape,ddtype,chunks=chunks,compression='lzf',
                          shuffle=True)

   print "ds.shape = (%i,%i); ds.chunks=(%i,%i)" % (ds.shape + ds.chunks)
   return ds,ds1d

def attchKW(ds,kwL,keys):
   """
   Attach keywords 

   Parameters
   ----------
   tL    : List of open tables
   keys  : List of h5 compatible keywords

   """
   for k in keys:
      k = str(k) # h5 saves saves keys as unicode
      kwarray = np.array( [kw[k] for kw in kwL] )
      try:
         ds.attrs[k] = kwarray
      except:
         print "could not attach %s" % k

   return ds

def ext(inp,ext,out=None):
   """
   Convience function for extension short cuts.
   """
   
   bname = os.path.basename(inp)
   bname_no_ext = bname.split('.')[0]

   if out is None:
      out = bname_no_ext+ext
   elif os.path.isdir(out):
      out = out+bname_no_ext+ext

   return out

      
