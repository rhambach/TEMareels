#!/usr/bin/python
"""
  RUNSCRIPT for a manual energy- and momentum calibration.

  USAGE 
    $ python EQmap_manual_calibration.py <edisp>
    Writes calibration objects assuming a constant energy- and
    momentum dispersion. The energy dispersion should be explicitly
    specified in eV/px.

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
     
"""

# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir
import pickle
import sys
import TEMareels.tools.transformations as trafo

# get parameters from commandline
try:
  EVpPX = float(sys.argv[1]);
except:  
  print "Usage: python EQmap_manual_calibration.py <edisp>      ";
  print "Writes calibration objects assuming a constant energy- ";
  print "and momentum dispersion. The energy dispersion should  ";
  print "be explicitly specified in eV/px.";
  sys.exit(0);

# ENERGY CALIBRATION (constant energy-dispersion corresponds to 
# polynomial of order 0 for disp)
e2x = trafo.NonlinearDispersion([1./EVpPX],scale=1); 
history ="MANUAL energy calibration using %f eV/px\n" % EVpPX;
history+=e2x.info(3);

# save as named dictionary
print 'SAVE Energy Calibration (Manual)';
FILE=open('EDisp_manual.pkl','w');
data={'e2x':e2x, 'descr':history.split('\n')[0], 'history':history};
pickle.Pickler(FILE).dump(data);
FILE.close();


# FAKE MOMENTUM CALIBRATION (no image transformation at all)
q2s = trafo.NonlinearDispersion([1]);  # identity (will be scaled later)
s2u = trafo.I();                       # identity
descr ="NO momentum calibration, assuming linear q-axis.";

# save as named dictionary
print 'SAVE Momentum Calibration (Identity Transformation)';
FILE=open('QDisp_manual.pkl','w');
data={'s2u':s2u, 'q2s':q2s};           # transformations
data['descr']=descr;                   # description
data['history']=descr;                 # string with parameters
pickle.Pickler(FILE).dump(data);
FILE.close();


