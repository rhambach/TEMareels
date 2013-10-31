"""
  RUNSCRIPT to align and filter raw WQmaps.

  IMPLEMENTATION
  - sums and aligns wq-map-series
  - options to remove outliers, align, and remove stripes
  - use ref_file for alignment reference and box to narrow down 
     on feature to algin
  see outliers.py, remove_stripes.py, and align.py in TEMareels/tools

  Copyright (c) 2013, pwachsmuth, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
# use TEMareels package specified in _set_pkgdir (overrides PYTHONPATH)
import _set_pkgdir

import glob;
import numpy as np
import matplotlib.pylab as plt

from TEMareels.tools import tvips
import TEMareels.tools.tifffile as tiff
import TEMareels.tools.img_filter as im
import TEMareels.tools.align as align
import TEMareels.tools.outliers as ro
import TEMareels.tools.remove_stripes as rs

# 121218 GK long 
descr   = "Test Series"
pattern = "../tests/wqmap*.tif";
outpath = "."
files   = sorted(glob.glob(pattern));

# reference image for alignment
ref_file = files[0];       # first image as anchor
box = [0,500,1550,2050];

# reference position for stripe substraction
intstart =[150,1025,2700,3800];
intwidth = 200;

# outlier options
radius = 1;
abs_thresh = 30;
rel_thresh = 1;


ref_img = tvips.load_TVIPS(ref_file)
#ref_img = ref_img.repeat(8,axis=0).repeat(8,axis=1); # for testing
ref = ref_img[box[0]:box[1],box[2]:box[3]];
stack  = [];

# OPTIONS
outliers       = True; # remove outliers
stripes        = True; # remove stripes
alignment      = True; # align images

final = np.zeros(shape=(4096,4096))
imgcounter = 0;

for filename in files:

 img = tvips.load_TVIPS(filename)
 #img = img.repeat(8,axis=0).repeat(4,axis=1); for testing
 img_old = img.copy();
 if ~np.allclose(img_old.shape, (4096,4096)):
  print  "%d,%d image detected" %(img_old.shape[0],img_old.shape[1])
  print "adjusting size";
  img_rez = np.zeros(shape=(4096,4096));
  img_rez[0:4096,1024:3072] = img_old;
  img = img_rez;
   
 if outliers:
  img = ro.remove_outliers(img,radius=radius,abs_thresh=abs_thresh,rel_thresh=rel_thresh,verbosity=2);
 if stripes:
  imax = img.sum(axis=1).argmax(); 
  img = rs.remove_stripes(img,intwidth=intwidth, intstart=intstart, mask = range(imax-30,imax+30), verbosity = 0);
 if alignment:
  sub_img = img[box[0]:box[1],box[2]:box[3]];
  offsetND = align.get_offsetND(sub_img, ref);
  print offsetND;
  offsetND[0] = 0; #consider only shift in x-direction
  img = align.alignND(img, ref_img, shift=offsetND);
  
 final += img;
 imgcounter += 1;
 
outfile = outpath +'/'+ pattern.split('/')[-1].split('*')[0]+"_%dimges_ro%s_rs%s_align%s_vac_ref.tif" %(imgcounter,outliers,stripes,alignment);
print "writing sum of %d images to %s" %(imgcounter, outfile)
tvips.write_tiff(final, outfile);

