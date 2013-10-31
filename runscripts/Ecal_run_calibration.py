"""
  RUNSCRIPT for energy-calibration.

  USAGE 

    Specify the pre-processed images and the reference spectra 
    for the peak fitting. Run this script using
      $ python Ecal_run_calibration.py

    Eventually you will have to adapt the parameters used
    for get_dispersion(). For a brief documentation, execute
      $ pydoc TEMareels.ecal.energy_dispersion

    In case of runtime errors, set the verbosity flag to a high 
    number for debugging and run the script in ipython to get
    figures up to the exception call.
      $ ipython -pylab
      In [1]: run Ecal_run_calibration.py
      In [2]: plt.show()

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
     
"""

# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir
import pickle
import matplotlib.pylab as plt
from TEMareels.ecal.energy_dispersion import get_dispersion
from TEMareels import Release

# FILENAMES
# - description for series (to identify series later on)
# - series of shifted energy scale
# - reference of energy scale


# test files
descr   = "test series for energy calibration";
pattern = "../tests/Eseries%d.tif";
series  = [pattern % (i) for i in range(1,2) if i not in []];
refname = "../tests/Ereference.msa";


# DISPERSION (order=0 for constant energy dispersion, i.e., linear E-axis)
params={'order':2, 'ampl_cut':0.5};
e2x=get_dispersion(series,refname,verbosity=5,**params)


# write history
history ="energy_dispersion.py, version %s (%s)\n" % \
         (Release.version, Release.version_id);
history+="Get dispersion: %s \n" % \
          (" ,".join([key+": "+str(val) for key,val in params.items()]) );
history+=e2x.info(3);

# print and save (as named dictionary)
print 'SAVE ' + e2x.info();
FILE=open('EDisp.pkl','w');
data={'e2x':e2x, 'descr':descr, 'history':history};
pickle.Pickler(FILE).dump(data);

FILE.close();

plt.show();


