"""
  Smoothing of Spectra using Whittaker-Handerson Graduation Method

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""

from scipy.misc import comb
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

def whfilter(a, weights=None, lamb=1600, p=3, ):
    """
    Generalized Whittaker-Handerson Graduation Method

    Parameters
    ----------
    a : array-like
          The input array, shape (n,)
    weights : array-like or None
          Weights
    lamb : float
          The relative importance between goodness of fit 
          and smoothness (smoothness increases with lamb).
    p : integer, default 3
          The degree of smoothness. We minimize the p-th 
          finite-differences of the graduated data. Examples:
          p=2 Hodrick-Prescott filter;
          p=3 Whittaker-Henderson method;
          Note: moments 0..p-1 will be conserved by graduation
    
    Returns
    -------
    out : array
          The smoothed data

    References
    ----------
    implementation of scikits.statsmodels.tsa.filters.hp_filter.py
    Alicja S. Nocon & William F. Scott (2012): "An extension of the 
       Whittaker-Henderson method of graduation", Scandinavian 
       Actuarial Journal, 2012:1, 70-79
    Whittaker, E. T. (1922). "On a new method of graduation", 
       Proceedings of the Edinburgh Mathematical Society 41,63-75.

    """
    # input data
    a = np.squeeze(a); 
    if a.ndim>1: raise ValueError("input array a must be 1d");
    n = a.size;

    # weights
    W = np.squeeze(weights) if weights is not None else np.ones(n);
    if np.any(W==0) or not np.all(np.isfinite(W)): 
      raise ValueError("weights must be non-zero and finite.");
    W = sparse.dia_matrix((W, 0), shape=(n,n));

    # set up difference Matrix K, shape (n-p, n)
    # K_ij = k(j-i),  l=j-i
    # k(l) = (-1)^l Binomial(p,l) if 0<=l<=p else 0
    l = np.arange(p+1);
    k = (-1)**l * comb(p,l);       # same as K_0j
    diags  =np.tile(k,(n,1)).T;    # side-diagonal K_i,i+l; n-times k(l)
    offsets=np.arange(p+1);        # index of side-diagonals
    K = sparse.dia_matrix((diags,offsets),shape=(n-p,n)); # K_ij

    # solve quadratic optimization problem 
    return spsolve(W+lamb*K.T.dot(K), W.dot(a));



# -----------------------------------------------------------------
if __name__ == "__main__":
  import matplotlib.pylab as plt
 
  x = np.arange(200);
  y = x**2 + np.random.normal(0,1000,200);
  
  yp= whfilter(y,lamb=0.01,weights=1./(1+np.abs(y)));
  plt.plot(x,y,label='original data');
  plt.plot(x,yp,label='smoothed data');
  plt.plot(x,y-yp,label='error');

  plt.show();

