"""
  RUNSCRIPT for removing the q-distortion of a measured WQmap using
  the momentum calibration object from a previous run.

  USAGE 

    The script rebins the WQmap along the q-axis to get an undistorted
    WQmap of size (nq,N). The energy-axis will be left unchanged however.
    Before execution, the following parameters should be specified:

    /important/
       q     ... list of momentum transfers, for each q one gets a
                 spectrum (avoid oversampling, i.e. q steps such that
                 spectra are averaged over several pixels)
       qdisp_name 
             ... filename for QDispersion object 
       filename 
             ... filename for distorted WQmap
       x0,y0 ... position of reference point in WQmap
       xl,xr ... slit borders in WQmap
       Gl,Gr ... position of Bragg spots in WQmap
       G     ... q [1/A] at one of the Bragg spots

    /optional/
       descr ... description for undistorted WQmap
       tifffile, pklfile
             ... outfile names
       verbosity
             ... for debugging, use value >10
       N     ... standard image size (to determine binning)

  OUTPUT
    writes undistorted WQmap to disk as
    * 32bit-Tif for import in other applications)
    * pkl-object for use with Python

  TODO
    - clean code ! separate steps using functions
    - better names for parameters
    - generalize to SREELS case

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
     
"""

# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir 
import pickle

import numpy as np
import matplotlib.pylab as plt

import TEMareels.tools.tifffile as tiff
import TEMareels.tools.transformations as trafo
from TEMareels.qcal.momentum_dispersion import calibrate_qaxis
from TEMareels.gui.wq_stack import WQBrowser, WQStackBrowser
from TEMareels.tools import rebin
from TEMareels import Release

# 0. parameters
descr    ="Undistorted Test map";
N        =4096;
verbosity=11;                       # debug: >10
dq       =0.5;                      # q-resolution for rebinning
q        =np.arange(-2.5,2.6,dq);   # q-values to be extracted
#q        =None;


# 1. get calibration object
qdisp_name = 'QDisp.pkl';
FILE = open(qdisp_name,'r');
qcal= pickle.Unpickler(FILE).load();
FILE.close();
s2u,q2s = qcal['s2u'],qcal['q2s']; # unpack the single trafos
history = qcal['history'];
history+= "\n\nremove_qdistortion.py, version %s (%s)" % \
                     (Release.version, Release.version_id);
print 'READ QDispersion: ' + qcal['descr'];

# 2. distorted E-q map and normalize q-axis
# TODO: automatic fit for border ?
filename= '../tests/qseries_sum.tif';
img     = tiff.imread(filename);
Ny,Nx   = img.shape;
ybin,xbin=N/float(Ny), N/float(Nx);

x0,y0 = 2377,967;                  # position of reference point in WQmap
xl,xr = 0,N;                       # (opt) slit borders in WQmap (fine-tuning of spec-mag)
Gl,Gr = 1193,3527;                 # coordinates of left+right Bragg spot at y0
G = 2.96;                          # length |G| corresponding to x0-Gl and Gr-x0
u2x = trafo.Normalize(x0,y0,xl,xr);# (u,v) -> (x,y)
s2x = trafo.Seq(u2x,s2u);          # s -> u -> x
sl,_= s2x.inverse(Gl,y0);
sr,_= s2x.inverse(Gr,y0);
q2s = calibrate_qaxis(q2s,sl,sr,G);

history+="\nRawfile: %s"%filename;
history+="\n"+u2x.info(3);
history+="\nCalibrate_qaxis: Gl=%d, Gr=%d, |G|=%8.5f"%(Gl,Gr,G);
history+="\n"+q2s.info(3);
history+="\nRebinning for qmin=%8.5f, qmax=%8.5f, dq=%8.5f"%(q.min(), q.max(),dq);

# 3. create bin boundaries ( in slit coordinates )
if q is None:                       # create q-values inside slit
  q2x   = trafo.Seq(u2x,s2u,q2s);   # combined trafo q->s->u->x
  qmin,_=q2x.inverse(xl,y0); qmin=np.floor(qmin/dq)*dq;
  qmax,_=q2x.inverse(xr,y0); qmax=np.floor(qmax/dq)*dq;
  q     =np.arange(qmin,qmax+dq,dq); 

#q2s.xrange=[-2,2];
qbins  = list(q-dq/2.)+[q[-1]+dq/2.];
sbins,_= q2s.transform(qbins,qbins);

# DEBUG: plot raw image and bin borders
if verbosity>9:
  s2x = trafo.Seq(u2x,s2u);          # combined trafo    s->u->x
  t  = np.linspace(-N/2,1.5*N,100);  # sampling along energy axis
  S,T= np.meshgrid(sbins,t); 
  X,Y= s2x.transform(S,T);
  X0,Y0=s2x.transform(0,t);          # q=0 line
  info = {'desc': 'DEBUG: input image', 'xperchan':xbin, 'yperchan':ybin};
  WQB  = WQBrowser(img,info,aspect='auto');
  WQB.axis.plot(X,Y,'r');
  WQB.axis.plot(X0,Y0,'g');
  plt.show();

# 4. rebinning for each line
ret = []; 
s2x = trafo.Seq(u2x,s2u);         # combined trafo    s->u->x
for n,line in enumerate(img):
  y = ybin*n+ybin/2.;
  x = np.arange(Nx,dtype=float);  # otherwise rounding errors for neg. x
  _,t     = s2x.inverse(0,y);     # linearized coord. of horizontal line
  xbins,_y= s2x.transform(sbins,[t]*len(sbins));
  assert np.allclose(_y,y);
  ret.append(rebin.rebin(x*xbin,line,xbins));
ret=np.asarray(ret);

if verbosity>9:
  info=[];
  info.append({'desc': 'DEBUG: rebinned image', 'yperchan':ybin,
          'xperchan': dq, 'xunits':'1/A', 'xlabel':'q', 'xoffset':q[0]});
  info.append({'desc': 'DEBUG: rebinned image (reversed q-axis)', 'yperchan':ybin,
          'xperchan': -dq, 'xunits':'1/A', 'xlabel':'-q', 'xoffset':q[-1]});
  fig  = WQStackBrowser([ret,ret],info,aspect='auto');
  plt.show();

# 5. save undistorted w-q map
# 5.1 save as TIFF (readable by DM3, ImageJ)
tiffile = '%s-rebinned%dx%d.tif'%(filename.split('/')[-1].split('.tif')[0],
                                  ret.shape[0],ret.shape[1]);
print 'write to file %s'%tiffile;
tiff.imsave(tiffile,ret.astype(np.float32));

# 5.2 pickle array and history for extract_spectrum
pklfile = tiffile.replace('.tif','.pkl');
print 'write to file %s'%pklfile;
FILE=open(pklfile,'w');
data={'yqmap':ret, 'qaxis':q, 'qdisp':qcal, 
      'rawfile':filename,'descr':descr, 'history':history};
pickle.Pickler(FILE).dump(data);
FILE.close();






