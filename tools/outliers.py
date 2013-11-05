"""
  Removing spikes from an grayscale image using an absolute
  and relative threshold.

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;
import scipy.ndimage as ndimg

def remove_outliers(img,radius,abs_thresh=50,rel_thresh=1,verbosity=0):
  """
  Removing dark and white spikes from an grayscale image. To find the
  spikes, we compare each pixel value with the median of pixel values
  in the neighborhood of given radius. Two criteria must be fulfilled

   - absolute_threshold:  |val - median|        > abs_thresh
   - relative_threshold:  |val - median|/median > rel_thresh 
   
  Input parameters:
    img       ... image
    radius    ... radius for neighborhood of pixels (box of size 2*radius+1)
    abs_thresh... absolute threshold, can be 0 [counts]
    rel_thresh... relative threshold, can be 0
    verbosity ... 0: silient, 1: print infos, 3: debug

  NOTE: in ImageJ outliers will be removed if (for bright pixels)
           val - median > abs_thresh
    This is not really adequate for WQmaps, where we have a huge 
    dynamic range in each image and it is difficult to define a 
    absolute threshold (would artificially remove signal at q=0)

  RETURNS:
    corrected image
  """
  # calculate deviation from smoothed image |val - median|
  img =np.asarray(img);
  med =ndimg.filters.median_filter(img,size=2*radius+1);
  err =np.abs(img-med);

  # find bad pixels
  bad_abs =  err > abs_thresh;
  bad_rel =  err / (np.abs(med)+1.) > rel_thresh;  # avoid division by 0
  bad     =  np.logical_and(bad_abs,bad_rel);

  # mask pixels
  img_cr      = img.copy();
  img_cr[bad] = med[bad];

  # DEBUG
  if verbosity>0: print "Number of outliers: ", np.sum(bad);
  if verbosity>1: print "Variance of error:  ", np.std(err);
  if verbosity>2:
    import TEMareels.gui.wq_stack as wqplt
    import matplotlib.pylab as plt
    desc = [ {'desc': title} for title in \
      ('original image', 'filtered image', 'outliers', 'noise+outliers')];
    WQB=wqplt.WQStackBrowser([img,img_cr,img-img_cr,err],info=desc);
    WQB.AxesImage.set_clim(0,abs_thresh);
    WQB._update();
    plt.show();
  
  return img_cr;


# -----------------------------------------------------------------
if __name__ == "__main__":

  # create test image with outliers
  x=np.arange(1000,dtype=float);
  y=10000/((x-200)**2 + 1.5**2) + 100000/((x-700)**2+10**2);
  img=np.transpose(np.tile(y,[1000,1]).T*(np.arange(1000))/5000.);
  # add noise
  img+=np.random.normal(0,10,(1000,1000));
  # add outliers
  out=np.random.randint(0,1000,(2,10000));
  img[out[0],out[1]]=np.random.normal(0,100,out.shape[1]);

  # remove outliers 
  remove_outliers(img,1,abs_thresh=40,rel_thresh=1,verbosity=3);
