"""
  RUNSCRIPT for momentum-calibration.

  USAGE 

    Specify the pre-processed calibration images of the shifted
    aperture and an illumination reference, if the illumination was
    non-homogeneous. You will have to adapt several parameters for the
    fitting routines in QDispersion. For a brief documentation, see

      $ pydoc TEMareels.qcal.momentum_dispersion

    In case of runtime errors, set the verbosity flag to a value>10 in
    order to open all figures up to the exception call. First, try to
    get the fitting of the aperture borders right. Then adapt the
    parameters for the polynomial fitting (start with small I=1, J=1,
    then try to increase the order of the polynomials for better
    accuracy). For more documentation on the transformations, see
     
      $ pydoc TEMareels.tools.transformations


  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
     
"""

# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir
import sys
import pickle
import matplotlib.pylab as plt
from TEMareels.qcal.momentum_dispersion import QDispersion

# FILENAMES
# - description for series (to identify series later on)
# - series of shifted apertures
# - illumination reference (image without aperture)
descr   = "test series for momentum calibration";
pattern = "../tests/qseries%d.tif";
aperture_files   = [pattern % (i) for i in range(1,10) if i not in []];
ref_illumination = "../tests/qreference.tif";
verbosity = 11; 

# fit aperture borders
try:
  QDisp=QDispersion(aperture_files, ref_illumination,verbosity=verbosity);
  #QDisp.plot_reference();
  QDisp.crop_img(xmin=100,ymin=1000);
  QDisp.fit_aperture_borders(rel_threshold=0.2,dev=3);

  QDisp.normalize_coordinates(2377,967,0,4096); # x0, y0, xl, xr
  QDisp.fit_polynomial_distortions(I=3,J=2,const='fixed_slit');
  QDisp.linearize_qaxis(ord=2);
except:
  if verbosity>10: plt.show();            # plot before exiting
  raise

# print and save (as named dictionary)
print 'SAVE ' + QDisp.get_q2u().info();
FILE=open('QDisp.pkl','w');
data={'s2u':QDisp.s2u, 'q2s':QDisp.q2s};  # transformations
data['descr']=descr;                      # description
data['history']=QDisp.get_status()        # string with parameters
pickle.Pickler(FILE).dump(data);
FILE.close();

plt.show();

