"""
  Commonly used model functions
     
  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;

def asymmetry(x, x0, fwhm, a):
  """
  asymmetric broadening following
  [Vibrational Spectroscopy 47 (2008) 66-69]
  """
  return 2.*fwhm / ( 1. +  np.exp( a * (x-x0) ) );

def gauss(x, x0, A, fwhm):
  """
   normalised gaussian function 
   A   ... integral
   x0  ... position
   fwhm... full widht half maximum
   [http://mathworld.wolfram.com/GaussianFunction.html]
  """
  sigma = fwhm / 2.354820045;  # fwhm / 2*sqrt(2 ln(2))
                               # normalis. const 1/sqrt(2pi)
  return A/2.50662827/sigma * \
          np.exp(-(x-x0)**2/(2*sigma**2));

def gauss_asymmetric(x, x0, A, fwhm, a):
  return gauss(x, x0, A, asymmetry(x,x0,fwhm,a));
                                 
def lorentz(x, x0, A, fwhm):
  """
   normalised lorentzian function 
   A   ... integral
   x0  ... position
   fwhm... full widht half maximum
   [http://mathworld.wolfram.com/LorentzianFunction.html]
  """
  sigma = fwhm / 2.;
  return A/np.pi *  sigma / ( sigma**2 + (x-x0)**2 );

def lorentz_asymmetric(x, x0, A, fwhm, a):
  return lorentz(x, x0, A, asymmetry(x,x0,fwhm,a));

def lorentz_background(x, x0, A, fwhm, n, m):
  sigma = fwhm / 2.;
  return A/np.pi *  sigma / ( sigma**2 + (x-x0)**2 ) + m*x + n

def fano(x, x0, A, fwhm, a=1, q=10.):
  eps = (x-x0)/0.5/fwhm;
  return A*( a**2 * (q+eps)**2 / q**2/ (1.+eps)**2 + (1-a) );
