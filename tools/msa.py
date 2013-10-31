"""
   Input/Output module for MSA-files
     
   USAGE
     # read single output file (header + data)
     data = MSA("../tests/raw.msa"); 
   
  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;
from sys import stdout;
from datetime import datetime;
from copy import copy;

required_fields = ('format' ,'version' ,'title' ,'date' ,'time' ,'owner',\
     'npoints', 'ncolumns' ,'xunits' ,'yunits' ,'datatype' ,'xperchan' ,'offset');

def write_msa(x, y, out=stdout, title='', owner='', xunits='', yunits='',\
              xperchan=1, offset=0, opt_keys = {}, verbosity=1):
  """
  writes msa files according to standard defined in  
  http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total

  x        ... x-axis, might be None for DATATYPE 'Y'
  y        ... y-axis, shape npoints,ncolumns
  out      ... (opt) file object, default stdout
  opt_keys ... (opt) list of tuples (keyword, value), e.g. ('#CHOFFSET',87)
                  
  """
  # -- HEADER --------------------------------------------------
  if x is None:   
    datatype = 'Y';
    npoints,ncolumns = y.shape if y.ndim==2 else (y.size,1);
  else:
    datatype = 'XY';
    npoints,ncolumns = (y.size,1);
    if y.ndim>1:
      raise ValueError("Only one single column is allowed in 'XY' mode.  \n"\
                     + "            Use write_msa(None,y,...) for 'Y' mode.");

  # write required keywords
  out.write('#FORMAT      : %s\r\n'% 'EMSA/MAS Spectral Data File'        );
  out.write('#VERSION     : %s\r\n'% '1.0'                                );
  out.write('#TITLE       : %s\r\n'% title[:64]                           );
  out.write('#DATE        : %s\r\n'% datetime.today().strftime("%d-%m-%Y"));
  out.write('#TIME        : %s\r\n'% datetime.today().strftime("%H:%M")   );
  out.write('#OWNER       : %s\r\n'% owner                                );
  out.write('#NPOINTS     : %d\r\n'% npoints                              );
  out.write('#NCOLUMNS    : %d\r\n'% ncolumns                             );
  out.write('#XUNITS      : %s\r\n'% xunits                               );
  out.write('#YUNITS      : %s\r\n'% yunits                               );
  out.write('#DATATYPE    : %s\r\n'% datatype                             );
  out.write('#XPERCHAN    : %g\r\n'% xperchan                             );
  out.write('#OFFSET      : %g\r\n'% offset                               );

  # write optional keywords (alphabetic order)
  for key,value in opt_keys:
    if key.strip('# ').lower() in required_fields: 
      if verbosity>3: print "WARNING: skip keyword '%s' in opt_keys";
      continue;
    out.write('%-13s: %s\r\n' % (key[:13],value));  # left aligned

  # write data
  out.write('#SPECTRUM    : Spectral Data Starts Here\r\n');
  for i in range(npoints):
    if datatype == 'XY':  
      out.write('%6g,%12g\r\n'%(x[i], y[i]));
    else:
      for val in y[i]: out.write('%12g,'%val);
      out.write('\r\n');
  out.write('#ENDOFDATA   :\r\n');



class MSA:
  """
  MSA: reads EMSA/MSA files (*.msa)
    -> all parameters given in the header
    -> the raw-data 
  
  http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total
  """
  def __init__(self,filename,verbosity=1):
    "filename ... full name of the inputfile"

    self.verbosity = verbosity;
    try:
      self.__read_file(filename);
    except:
      print "\nERROR! while reading file '%s'.'\n\n"%(filename)
      raise

    
  def __read_file(self,filename):

    # scan header
    file=open(filename);
    self.original_parameters=[];
    while True:
  
      line=file.readline();
      if not line or line[0]<>"#": 
        raise IOError("Unexpected header format.");
      key,value = line.split(':',1);
  
      # determine bare-keyname and unit, ex. '#CHOFFSET -px' -> 'CHOFFSET'
      barekey = [ s.strip().strip('#') for s in key.split('-')];
      param,unit = barekey if len(barekey)==2 else (barekey[0],None);
      if param == 'SPECTRUM': break;

      # convert value to correct type
      value=value.strip();
      if value.isdigit():  value=int(value);      # integer
      else: 
        try:               value=float(value);    # float
        except ValueError: pass;                  # string 
      self.original_parameters.append(( key, value, param.lower(), unit ));
      if self.verbosity > 2:
        print '%s: %8s, %8s, %8s, %8s' \
            %(key,param.lower(),str(value),str(unit),str(type(value)));

    # scan data
    self.original_data = np.genfromtxt(file,delimiter=',',dtype=float);
    if all(np.isnan(self.original_data[:,-1])):   # correct last ',' in case of 'Y'-type
      self.original_data = self.original_data[:,:-1];
    file.close();

    # parameter dictionary:
    self.param = {};
    for keyword,value,key,unit in self.original_parameters:
      self.param[key] = value;
    self.param['filename']=filename;

    # test validity of MSA file
    if len(self.original_data) <> self.param['npoints']:
      raise IOError("Unexpected number of data points: "\
 + "got %d lines, but npoints=%d" % (len(self.original_data), self.param['npoints']));

    ncol = self.original_data.shape[1];
    if self.param['datatype']=='Y' and  ncol <> self.param['ncolumns']:
      raise IOError("Unexpected number of data columns: "\
 + "got %d data columns, but ncolumns=%d" % (ncol,self.param['ncolumns']));
    if self.param['datatype']=='XY' and  ncol-1 <> self.param['ncolumns']:
      raise IOError("Unexpected number of data columns: "\
 + "got %d data columns, but ncolumns=%d" % (ncol-1,self.param['ncolumns']));



  def get_axis(self):
    if self.param['datatype']=='Y':
      return np.arange(0,self.param['npoints'],dtype=float) \
              * self.param['xperchan'] + self.param['offset'];
    else:
      return self.original_data[:,0];

  def get_data(self,col=None):
    if self.param['datatype']=='XY':
      data = self.original_data[:,1:];
    elif col is None:
      data = self.original_data;
    else:
      data = self.original_data[:,col];      
    # eventually flatten array
    if data.ndim>1 and data.shape[1]==1:
      data = data.flatten();
    return data;

  def get_opt_params(self):
    " return optional keywords from msa-header "
    opt = [];
    for keyword,value,key,unit in self.original_parameters:
      if key not in required_fields: 
        opt.append((keyword,value));
    return opt;

  def get_req_params(self):
    " get required keywords from msa-header "
    req = [];
    for keyword,value,key,unit in self.original_parameters:
      if key in required_fields: 
        req.append((keyword,value));
    return req;

  def write(self,out=stdout):
    " write file to file object out "
    params  = dict( (k,self.param[k]) for k in ('title','owner','xunits','yunits','xperchan','offset'));
    # write
    if self.param['datatype']=='XY':
      write_msa(self.get_axis(), self.get_data(), out=out, opt_keys=self.get_opt_params(), **params);
    else:
      write_msa(None,self.get_data(), out=out, opt_keys=self.get_opt_params(), **params);

  def __str__(self):
    p = "%s \n" % self.param['format'];
    p+= "filename:  %s\n" % self.param['filename'];
    p+= "tile    :  %s\n" % self.param['title'];
    p+= "npoints :  %d\n" % self.param['npoints'];
    p+= "ncolumns:  %d\n" % self.param['ncolumns'];
    p+= "datatype:  %s\n" % self.param['datatype'];

    return p;

  def __len__(self):
    return self.original_data.shape[1];
    
  def __getitem__(self,i):
    return self.original_data[:,i];

  def plot(self):
    import matplotlib.pylab as plt;
    fig= plt.figure();
    ax = fig.add_subplot(111);
    x = self.get_axis();
    y = self.get_data().reshape(self.param['npoints'],self.param['ncolumns']);
    for i in range(self.param['ncolumns']):
      ax.plot(x,y[:,i], label='col #%d'%i);
    ax.set_xlabel(self.param['xunits']);
    ax.set_ylabel(self.param['yunits']);
    ax.set_title(self.param['filename']+': '+ self.param['title']);
    plt.legend();
    return fig;

# -- main ----------------------------------------
if __name__ == '__main__':
   import matplotlib.pylab as plt;

   data = MSA("../tests/raw.msa",verbosity=1);
   # test functions
   print data;
   len(data), data[0];
   data.plot();

   # test xaxis
   assert(np.allclose(data.original_data[:,0], data.get_axis()));

   data.write(open('murx','w'));
    
   plt.show();
