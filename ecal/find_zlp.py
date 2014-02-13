"""
  Finding and fitting of ZLP peak positions in rebinned wq-map 

  IMPLEMENTATION:
    - find highest intensity in linescan
    - polyfit peaks using parabola

  TODO
    - improve roboustness

  Copyright (c) 2013, pwachsmuth. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
import matplotlib.pylab as plt
from TEMareels.gui.wq_stack import WQBrowser


def find_zlp(img,delta=40,vmax=1000000,verbosity=0,N=4096):
 """ 
 delta  ignore all peaks more than delta away from reference value
 vmax   option for imshow, adjust contrast etc
 """
 
 img = np.asarray(img);
 Ny,Nx   = img.shape;
 ybin,xbin= N/float(Ny), N/float(Nx);     # binning
 x = np.arange(N,dtype=float);
 
 peaks = []
 sum = img.sum(axis=1)#np.empty(Nx);
 

 y0 = sum.argmax(); 
 #print y0
 y_start = y0;


 for i in range(0,Nx):
   peak_value = img[:,i].max();
   peak_index = img[:,i].argmax();
   if np.abs(peak_index - y_start)<delta:
     peaks.append([i*xbin,peak_index*ybin]);
 
 peaks = np.asarray(peaks);
 fit = np.polyfit(peaks[:,0],peaks[:,1],2);
 fitFunc = np.poly1d(fit)

 if verbosity > 9:
  info = {'desc': 'Find Zero-Loss Peak',
          'xperchan':xbin,'yperchan':ybin};
  WQB  = WQBrowser(img,info,aspect='auto');
  WQB.axis.plot(peaks[:,0],peaks[:,1], 'ro');  
  WQB.axis.plot(x,fitFunc(x)); 


 return fitFunc;
 
# -- main ----------------------------------------
if __name__ == '__main__':
  import TEMareels.tools.tifffile as tiff

  image = "../tests/wqmap.tif"; 
  img = tiff.imread(image)
  find_zlp(img, delta=10, vmax=img.max(), verbosity=11);
  plt.show();
