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


def find_zlp(img,qaxis,delta=40,vmax=1000000,verbosity=0,N=4096):
 """ 
 img    Eqmap (q-distortion removed)
 qaxis  axis along q
 delta  ignore all peaks more than delta away from reference value
 vmax   option for imshow, adjust contrast etc
 N      defined number of pixels along E

 RETURNS fit function y0_zlp(q) which gives the position
  of the zero-loss peak in the image (in px) for any given q (in 1/A) 
 """
 
 img = np.asarray(img);
 Ny,Nq = img.shape;
 assert len(qaxis)==Nq;
 ybin = N/float(Ny);     # binning in E-direction
  
 peaks = []
 sum = img.sum(axis=1);  # find ZLP roughly (average over all q) 
 y0  = ybin*sum.argmax(); 
 y_start = y0;

 for iq,q in enumerate(qaxis): # find ZLP for each q value
   peak_value = img[:,iq].max();
   peak_index = ybin*img[:,iq].argmax();
   if np.abs(peak_index-y_start) < delta:
     peaks.append([q,peak_index]);
 
 peaks = np.asarray(peaks);
 fit = np.polyfit(peaks[:,0],peaks[:,1],2);
 fitFunc = np.poly1d(fit)

 if verbosity > 9:
  info = {'desc': 'Find Zero-Loss Peak',
          'yperchan':ybin, 'xperchan':qaxis[1]-qaxis[0], 
          'xlabel':'q', 'xunits':'1/A','xoffset' :qaxis[0]};
  WQB  = WQBrowser(img,info,aspect='auto');
  WQB.axis.plot(peaks[:,0],peaks[:,1], 'ro');  
  WQB.axis.plot(qaxis,fitFunc(qaxis)); 


 return fitFunc;
 
# -- main ----------------------------------------
if __name__ == '__main__':
  import TEMareels.tools.tifffile as tiff

  image = "../tests/wqmap.tif"; 
  img = tiff.imread(image);
  N,nq= img.shape;
  qaxis = np.linspace(-3,3,nq);
  find_zlp(img, qaxis, delta=10, vmax=img.max(),verbosity=11);
  plt.show();
