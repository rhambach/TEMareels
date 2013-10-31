"""
  Rebinning of data keeping intensity locally constant. To this
  end, the integral of the image intensity is interpolated.
     
  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np

def rebin(x,f,bins):
  """
  REBIN a given function f(x) at equidistant sample points x
   according to the new binning intervals [bins[i], bins[i+1]]
   we use an interpolation of Int(f) in order to keep 
   the total flux constant
  """

  # calculate boundaries of bins (exactly between two x points)
  a = get_bin_boundaries(x);

  # perform interpolation of Int(f)
  F = np.insert(np.cumsum(f),0,0);  # F[i] is integrated signal up to bin boundary a[i]
  Fp= np.interp(bins,a,F);          # linear interpolation for F(bins);
  fp= np.diff(Fp);                  # fp[i] is signal of bin at xp[i]
                                    #   with borders ap[i] and ap[i+1]
  return fp;

def get_bin_boundaries(x):
  """
    Return bins according to bin centers x[i] such that
      x[i] = (a[i+1]+a[i])/2
  """
  a = np.zeros(len(x)+1);
  a[1:-1] = ( x[1:] +  x[:-1]) / 2.;
  a[0] = 2*x[0] - a[1];
  a[-1]= 2*x[-1]- a[-2];
  #print  max(abs((a[1:]+a[:-1])/2.-x)), len(a), len(x)
  assert(np.allclose( (a[1:]+a[:-1])/2., x));
  return a;


# -- main ----------------------------------------
if __name__ == '__main__':
  # todo: expand tests
  f=np.ones(1000,dtype=float);
  x=np.arange(1000);
  bins=np.arange(100)+100.5;
  fp=rebin(x,f,bins);

  assert fp.sum()==99;
  assert np.allclose(fp,1);


  # sum test?
  
#   imin=np.where(x<bins[0])[0][-1];
#   imax=np.where(x>bins[-1])[0][0];
#   #print "%f > %f > %f"%(np.sum(f[imin:imax+1]), np.sum(fp), np.sum(f[imin+1:imax-1])) 
#   #print x[imin],bins[0],bins[-1],x[imax]
#   assert x[imin] < bins[0]  < x[imin+1];
#   assert x[imax] > bins[-1] > x[imax-1];
#   assert np.sum(f[imin:imax+1]) > np.sum(fp);  # upper bound
#   assert np.sum(f[imin+1:imax-1]) < np.sum(fp);  # lower bound  # ToDo: error occurs with correct boundary imax
 

  print " all tests passed ..."
