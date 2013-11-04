"""
  RUNSCRIPT to extract spectra from series of images with 
  different energy offset

  USAGE
    Set infile names, energy steps, outfile name and run
    $ python run_image_preprocess.py 

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir

import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt

import TEMareels.tools.tifffile as tiff
from TEMareels.tools import tvips

# define in/outfile name and energy steps 
pattern = "/path/to/energy_calibration/img%d.tif";
series  = [pattern%i for i in range(329,350) if i <> 342]
dE      = 5;       # energy steps in eV 
outfile = "Ecalibration.tif";

# get image dimensions and energy list
image = tvips.load_TVIPS(series[0]);
Ny,Nx = image.shape
Ns    = len(series);
E     = np.arange(0,Ns,dE);  # dE=1eV

# iterate over all images and extract EELS spectrum
x0 = np.zeros(Ns);             # center of spectrum
dx = np.zeros(Ns);             # width of spectrum
spectra = np.zeros((Ns,Ny+1)); # extracted EELS spectra

for s in range(Ns):
  image = tvips.load_TVIPS(series[s]);       # shape Ny times Nx
  assert np.allclose(image.shape, (Ny,Nx));

  # determine position of spectrum along x-axis by fitting
  xline = np.sum(image,axis=0); 
  def gauss(x,x0,sigma,A,bg): # fit model
    return A*np.exp(-(x-x0)**2/(2*sigma**2))+bg;
  imax  = np.argmax(xline);   # initial parameters
  pconv = (imax, 10, xline[imax], 0);
  param, pconv = opt.curve_fit(gauss,range(Nx),xline,p0=pconv);
  x0[s] = param[0]; dx[s]=3*param[1];  # xmax +/- 3*sigma
  #print param

  # crop image and extract EELS
  xmin = max(x0[s]-dx[s],0); xmax = min(x0[s]+dx[s], Ny);
  spectra[s,0:Ny] = np.sum(image[:,xmin:xmax],axis=1); # spectrum
  spectra[s,Ny]   = E[s];                              # energy offset

  # DEBUG
  if s==0:
    plt.imshow(image,vmin=0,vmax=1000);
    plt.figure()
    plt.plot(xline,label='projected Tiff');
    plt.plot(gauss(range(Nx),*param),label='Gauss fit');
    window=np.zeros(Nx); window[xmin:xmax]=xline.max();
    plt.plot(window,label='selected region');
    plt.legend();
    plt.show(block=False);
 
# test if crop region has been too small
if x0.max()-x0.min() > dx.min():
  print 'WARNING: atomatic crop region might be too small'

# save spectra to TIFF
print "SAVE EELS spectra to file '%s'"%outfile;
tiff.imsave(outfile,spectra.astype(np.float32));

# plot all spectra
plt.figure()
for i,s in enumerate(spectra):
  plt.plot(s/float(s[0:Ny].max()) -i*0.1,'k',linewidth=1);

plt.show()
