"""
  RUNSCRIPT for applying the energy-calibration to the set of raw
  spectra from the undistorted EQmap.

  USAGE 

    The script rebins the WQmap along the E-axis and applies an
    aperture correction. Before execution, the following parameters
    should be specified:

    /important/
       edisp_name ... filename for EDispersion object 
       filename   ... filename for undistorted WQmap 
                      (output from EQmap_remove_qdistortion)
       qmin,qmax  ... only spectra with qmin<q<qmax will be extracted
       
    /optional/
       E          ... energy axis after rebinning (like for q, one 
                       should avoid bin sizes smaller than a few pixels)
       apply_APC  ... if True, we apply an aperture correction
       E0         ... beam energy
       dqy        ... slit width in y-direction
       ytilt, time, qdir, title, owner
                  ... parameters that will be written to the MSA header
       outname    ... name-pattern for the MSA files
       verbosity  ... for debugging, use value >10
       N          ... standard image size (to determine binning)

  OUTPUT
    writes one MSA file for each q-spectrum

  TODO
    - clean code ! separate steps using functions
    - better names for parameters
    - generalize to SREELS case

  Copyright (c) 2013, rhambach, pwachsmuth 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.

"""
# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir

import pickle
import numpy as np
import matplotlib.pylab as plt
from   scipy.integrate import quad

from TEMareels.tools import rebin, msa
from TEMareels.ecal  import find_zlp
from TEMareels.gui.wq_stack import WQBrowser
from TEMareels.aperture.TEM_wqslit import TEM_wqslit
import TEMareels.tools.tifffile as tiff
import TEMareels.tools.conversion as conv


N         = 4096;
verbosity = 10;

# Experimental parameters
E0        = 40;         # keV, beam energy
E0_offset = 0;          # eV, beam energy offset (e.g. for core-loss)
ytilt     = 0;          # degree, tilt of sample
time      = 400000;     # ms,  total exposure time
qdir      = 'GM';       # crystallographic axis of slit
title     = 'Title';
owner     = 'Owner';    
dqy       = 0.21;       # 1/A, slit width  

qmin,qmax = -1.4, 1.4;  # 1/A, limits for q vectors to be written to file
apply_APC = True;       # True when aperture correction should be performed

# energies
dE        = 1; 
E         = np.arange(10,40,dE)+dE;
E         = None;       # automatically determine Erange (over full image)

# 1. get calibration object
edisp_name = './EDisp.pkl';
FILE = open(edisp_name,'r');
edisp= pickle.Unpickler(FILE).load();
FILE.close();
e2y  = edisp['e2x'];
print 'READ Energy Calibration: ' + edisp['descr'];

# 2. read undistorted E-q map
filename = './qseries_sum-rebinned64x11.pkl';
FILE = open(filename,'r');
data = pickle.Unpickler(FILE).load();
FILE.close();
yqmap    = data['yqmap'];
qaxis    = data['qaxis'];
dqx      = qaxis[1]-qaxis[0];  
assert np.allclose(dqx, np.diff(qaxis)); # require evenly space qpoint list
Ny,Nx    = yqmap.shape;
ybin,xbin= N/float(Ny), N/float(Nx);     # binning
print "READ undistorted E-q map: " + data['descr'];

#find and fit zlp
y0_fit = find_zlp.find_zlp(yqmap,delta=10,verbosity=verbosity);
plt.show();

# 3. create bin boundaries ( in slit coordinates )
if E is None:                             # create list of E on screen
  Emin  = np.ceil(e2y.inverse(0,0)[0]/dE+1)*dE;
  Emax  = np.floor(e2y.inverse(N,0)[0]/dE)*dE;
  E     = np.arange(Emin,Emax,dE);
Ebins  = list(E-dE/2.)+[E[-1]+dE/2.];
ybins,_= e2y.transform(Ebins,Ebins);

# DEBUG: plot raw image and bin borders
if verbosity>9:
  X,Y= np.meshgrid(ybins,qaxis); 
  info = {'desc': 'DEBUG: input WQmap + bin borders (red lines)', 'xperchan':ybin,
          'yperchan':dqx, 'ylabel':'q', 'yunits':'1/A',
          'yoffset' :qaxis[0]};
  WQB  = WQBrowser(yqmap.T,info,aspect='auto');
  WQB.axis.plot(X,Y,'r');
  plt.show();

# 4. rebinning for each line
y     = np.arange(Ny)*ybin; 
EQmap =[];
for iq, q in enumerate(qaxis):
  # rebin spectrum at q
  line = yqmap[:,iq];
  spectrum = rebin.rebin(y,line,ybins);
  EQmap.append(spectrum);
EQmap=np.asarray(EQmap).T;  # first index E, second q
history="\nRebinning for Emin=%8.5f, Emax=%8.5f, dE=%8.5f"%(E.min(), E.max(),dE);
# DEBUG: plot calibrated Eqmap
if verbosity>9:
  info = {'desc': 'DEBUG: rebinned WQmap', 
          'xperchan':dE, 'xlabel':'E', 'xunits':'eV',
          'yperchan':dqx,'ylabel':'q', 'yunits':'1/A',
          'yoffset' :qaxis[0]};
  WQB  = WQBrowser(EQmap.T,info,aspect='auto');
  plt.show();

# 5. save E-q map as readable tif
outfile = '%s-calibrated.tif'%(filename.split('/')[-1].split('.pkl')[0]);
print 'write to file %s'%outfile;
tiff.imsave(outfile,EQmap.astype(np.float32));

# 6. save energy-loss function
for iq, q in enumerate(qaxis):
  if qmin > q or q > qmax: continue 

  # calibrate offset in energy axis (so far E=0eV at x=0px)
  E_ZLP,_ = e2y.inverse(y0_fit(iq),y0_fit(iq)); 
  Ecorr   = E + E0_offset - E_ZLP;
  # calculate aperture correction function for given q (rectangular aperture)
  # note: APC for negative energies is well defined (energy gain)
  if apply_APC==True:
    aperture = TEM_wqslit(q*conv.bohr,dqx*conv.bohr,dqy*conv.bohr,E0);
    APC      = [aperture.get_APC(_E) for _E in Ecorr];
  else: APC  = 1;
  elf = EQmap[:,iq]/APC

  # write file containing energy-loss function:
  # required parameters
  param = { 'title'   : title + ', ELF',
            'owner'   : owner,
            'xunits'  : 'eV', 'yunits'  : 'counts',
            'xperchan': dE,   'offset'  : -E_ZLP};
  # optional parameters
  opt       = [];
  opt.append(('#SIGNALTYPE','ELS'));
  opt.append(('#ELSDET',    'PARALL'));
  opt.append(('#BEAMKV   -kV',E0));
  opt.append(('#YTILTSTGE-dg',ytilt));
  opt.append(('#INTEGTIME-ms',time));
  opt.append(('##q      -1/A',q));
  opt.append(('##dqx    -1/A',dqx));
  opt.append(('##dqy    -1/A',dqy));
  opt.append(('##qdirection ','GM'));
  opt.append(('##EOFFSET -eV',E0_offset));
  opt.append(('##APC        ',str(apply_APC)));
  opt.append(('##rawfile',data['rawfile']));

  # history
  for l in data['history'].split('\n'):   opt.append(('#COMMENT',l));
  opt.append(('#COMMENT',''));
  for l in edisp['history'].split('\n'):  opt.append(('#COMMENT',l));
  opt.append(('#COMMENT',''));
  for l in history.split('\n'):  opt.append(('#COMMENT',l));

  # write data
  root = filename.split('/')[-1].split('.tif')[0];
  outname = 'EELS_%s_q%s.msa'%( root, ('%+05.3f'%q).replace('.','_') );
  print "writing file '%s' ..."%outname;
  out = open(outname,'w');
  msa.write_msa(E,elf,out=out,opt_keys=opt,**param);
  out.close();
  
