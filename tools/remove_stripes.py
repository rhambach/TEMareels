"""
  IMPLEMENTATION:
    - crude method for removing periodic noise in images recorded 
      on Tietz CMOS slave camera in wq-mode
    - integrates over several lines (e.g. 10x4096) of noise and 
      substracts signal from each line in sector
     
  Copyright (c) 2013, pwachsmuth, rhambach
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.

"""
import numpy as np
import matplotlib.pylab as plt

def remove_stripes(image, intwidth=10, intstart=[0,1025,2900,3800], 
                   sector_width=1024, mask = None, verbosity = 0):
 
 image = np.asarray(image)
 old_image = image.copy(); 
 
 offset = 0;
 for j in range(0,4):
  ref_line = old_image[:,intstart[j]:intstart[j]+intwidth].sum(axis=1)*(1./(intwidth));
  #ref_line[ref_line > thresh] = 0;
  imax = ref_line.argmax();
  if mask is not None:
   ref_line[mask] = 0;
  for i in range(offset,offset+sector_width):
   image[:,i] = image[:,i]-ref_line;
  offset += sector_width;
  #print offset 
 image[:,0:5]= image[:,-5:] = 0; 
 if verbosity > 0:
  plt.title("Remove Stripes: difference between old and new image");
  plt.imshow(image - old_image, aspect='auto')
  plt.show();
 return image;

 
 # -- main ----------------------------------------
if __name__ == '__main__':
  import TEMareels.tools.tifffile as tiff
  from TEMareels.tools import tvips


  image_file = '../tests/wqmap.tif';
  image = tiff.imread(image_file).astype(float);
  binning = 8;
  intstart= np.array([0,1025,2900,3800])/binning;
  
  img = remove_stripes(image, intwidth=100/binning, 
          intstart=intstart, sector_width=1024/binning, verbosity=1);
          
  #outfile = "script_test_.tif";
  #tvips.write_tiff(img, outfile);
  
 
  
