"""
  RUNSCRIPT to rebin raw images of q-calibration series along
  the energy axis.

  USAGE
    Set infile names, binning and run
    $ python run_image_preprocess.py 

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
     
"""

# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir
import glob;
import numpy as np
import scipy.signal as sig
from TEMareels.tools.img_filter import binning
from TEMareels.tools import tvips
import TEMareels.tools.tifffile as tiff

# define infile name and binning along y-direction (E-axis)
pattern = "/path/to/q_calibration/img%d.tif";
files   = [pattern%i for i in range(344,345)]
ybin    = 64;
  
for filename in files:

  image = tvips.load_TVIPS(filename)     # img[iy,ix]
  #image = tiff.imread(filename).astype(np.float64);
  if ~np.allclose(image.shape, (4096,4096)): 
    raise IOError("Unexpected image size in file '%s'"%filename); 
  medimg= sig.medfilt2d(image,kernel_size=3); # filtering 
  binimg= binning(medimg,ybin);          # binning along y
  #binimg = binning(binimg.T,32).T;      # binning along x

  outfile = filename.split(".tif")[0]+"_filt_bin%d.tif"%ybin;
  tiff.imsave(outfile,binimg.astype(np.float32));






