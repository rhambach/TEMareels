"""
  Filter images (line by line)
     
  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;
import scipy.signal as sig;
import scipy.optimize as opt;

from TEMareels.tools import models

def binning(img,px,axis=0):
  " binning of image by px pixels along specified axis" 
  if px==1: return np.asfarray(img);
  N = img.shape[axis];
  assert(N%px==0);  # N must be integer multiple of px
  lines = [np.sum(img[i*px:(i+1)*px], dtype=float, axis=axis) for i in range(N/px)];
  return np.asarray(lines);

def gaussfilt1D(img, broad):
  " 1D convolution with Gaussian of specified width (blurring) "
  if broad<1e-5: return img;
  
  N = img.shape[0];
  Nmask  = np.ceil(5*broad);
  E_gauss= np.arange(-Nmask,Nmask+1,1);    # symmetric gauss at center
  gauss  = np.exp(-0.5*(E_gauss/broad)**2);
  norm   = np.sum(gauss);
  if img.ndim==1:
    return np.convolve(img,gauss,'same')/norm;
  else:
    lines = [  np.convolve(line,gauss,'same')/norm for line in img ];
    return np.asarray(lines);

def medfilt1D(img,medfilt_radius):
  " 1D median filter using specified radius "
  if medfilt_radius<=1: return img;
  if img.ndim==1:
    return sig.medfilt(img,medfilt_radius);
  else:
    lines = [ sig.medfilt(line ,medfilt_radius) for line in img ];
    return np.asarray(lines);

def lorentzfit1D(img,offset=0,sigma=None):
  """ 
   Perform a Lorentz fit for each line y >= offset, 
   sigma determines the inverse weight of the x-points 
   for the lorentz fit (see optimize.curve_fit)
  """
  img   = np.atleast_2d(img);
  Ny,Nx = img.shape;
  ret   = np.empty((Ny,Nx));
  x     = np.arange(Nx);
  for iy,line in enumerate(img):

    if iy<offset: ret[iy]=0; continue
    param = (np.argmax(line), np.sum(line), 10);  # initial guess for x0, area, width
    param, pconv = opt.curve_fit(models.lorentz,x,line,p0=param,sigma=sigma);
    ret[iy] = models.lorentz(x,*param);
    #print  "pos max: ", imax, ",  fit params: ", param

  return ret;


def lorentz_with_background_fit1D(img,offset=0,sigma=None):
  """ 
   Perform a Lorentz+bg fit for each line y >= offset, 
   sigma determines the inverse weight of the x-points 
   for the lorentz fit (see optimize.curve_fit)
  """
  img   = np.atleast_2d(img);
  Ny,Nx = img.shape;
  ret   = np.empty((Ny,Nx));
  x     = np.arange(Nx);
  for iy,line in enumerate(img):

    if iy<offset: ret[iy]=0; continue
    param = (np.argmax(line), np.sum(line), 10, 0, 0);  # initial guess for x0, area, width, linear background
    param, pconv = opt.curve_fit(models.lorentz_background,x,line,p0=param,sigma=sigma);
    ret[iy] = models.lorentz_background(x,*param);
    #print  "iy: ", iy, ", fit params: ", param

  return ret;
