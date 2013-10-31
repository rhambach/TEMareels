"""
  Module defining image transformations and its inverse

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""

import abc
import copy 

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize  import brentq,newton
from scipy.integrate import quad

# abstract base class, see http://www.doughellmann.com/PyMOTW/abc/
class Transformation(object):
  """
  Parent class for transformations
  """
  __metaclass__ = abc.ABCMeta

  def plot(self,points=None):
    """
    visualise transformation properties for a square grid
    RETURNS figure handle
    """
    # define grid lines for (u,v) coordinates
    u,v = np.mgrid[0:1:10j,0:1:10j] if points is None else points;
    x,y = self.transform(u,v);

    # plotting
    fig=plt.figure(); ax=fig.add_subplot(111);
    ax.set_title(self.info());
    ax.plot(x,y,'r');
    ax.plot(x.T,y.T,'r');
    return fig;

  @abc.abstractmethod
  def info(self, verbosity=0):
    """ 
    return info string describing the transformation,
    verbosity ... 0: one-liner, 1: +parameters, 2: debug
    """
    return;

  @abc.abstractmethod
  def transform(self,u,v): return;

  @abc.abstractmethod
  def inverse(self,x,y):   return;


class Shift(Transformation):
  " Set (y0,x0) as new origin. "

  def __init__(self,x0,y0):
    " new origin x0, y0 in [px] "
    self.x0=x0
    self.y0=y0

  def transform(self,u,v):
    " transform normalised coordinates (u,v) to image pixels (x,y) "
    return u+self.x0, u+self.y0;

  def inverse(self,x,y):
    " transform image pixels (x,y) to normalised coordinates (u,v) "
    return x-self.x0, y-self.y0;

  def info(self,verbosity=0):
    ret = "Shift";
    if verbosity>0: ret+= "\n|- x0: %g, y0: %g"%(self.x0,self.y0)
    return ret;


class Normalize(Transformation):
  """
  Transformation of slit coordinates (u,v) to image pixels (x,y) in
  E-q maps. The origin is mapped to the position of the direct beam at
  (x0,y0), and the unit vector (1,0) is identified with the difference
  between left and right slit border in the image (xr,y0)-(xl,y0). By 
  default, the y-axis is not scaled. Otherwise, the aspect dx/dy can
  be specified explicitely (use 1 for common magnification of x and y axis).
  """

  def __init__(self,x0,y0,xl,xr,aspect=None):
    """
    x0    ... horizontal position of the direct beam [px]
    y0    ... vertical slit position [px]
    xl,xr ... horizontal left/right slit position [px]
    aspect... (opt) change of apect ratio (du/dv)/(dx/dy) 
    """
    self.x0=x0;
    self.y0=y0;
    self.xl=xl;
    self.xr=xr;
    self.width =float(xr-xl);
    self.aspect =(1./self.width) if aspect is None else float(aspect);
	
  def transform(self,u,v):
    " transform normalised coordinates (u,v) to image pixels (x,y) "
    u,v= np.atleast_1d(u,v);
    x = u*self.width+self.x0;
    y = v*self.width*self.aspect+self.y0;
    return x,y

  def inverse(self,x,y):
    " transform image pixels (x,y) to normalised coordinates (u,v) "
    x,y= np.atleast_1d(x,y);
    u = (x-self.x0)/self.width;
    v = (y-self.y0)/self.width/self.aspect;
    return u,v;

  def info(self,verbosity=0):
    ret = "Normalize";
    if verbosity>0: ret+= "\n|- x0: %g, y0: %g, xl: %g, xr: %g, aspect: %g"\
                             %(self.x0,self.y0,self.xl,self.xr,self.aspect);
    return ret;

class NonlinearDispersion(Transformation):
  """
  Correction of the nonlinear dispersion dx/du of the first axis.
  We transform the undistorted coordinates (u,v) to 
  distorted coords (x,y), where
 
    u(x) = Int_0^x dx'/disp(x'),  v=y,  disp(x) = DX/DU

  """

  def __init__(self, disp, scale=1, xrange=None):
    """
    disp  ... array of polynom coefficients giving the dispersion dx/du
    scale ... (opt) size of scale DU used for dispersion measurement
    xrange... (opt) range of allowed x-values (where disp>0)
    """
    self.disp   = np.poly1d(disp);
    self.scale  = scale;
    if xrange is None: 
      x0=self.disp.r;                         # roots of polynom (evtl. complex)
      x0=np.hstack((x0[x0.imag==0].real,      # real roots + -inf, inf
                    [-np.inf,np.inf]));
      xrange=(max(x0[x0<0]),min(x0[x0>0]));   # range of monotonic u(x)

      # slightly reduce xrange to avoid 1/0 in _inv_disp()   
      if np.all(np.isfinite(xrange)): 
        x0 = np.sum(xrange)/2.;
        xrange = np.asarray(xrange)*(1-1e-6) + x0 * 1e-6;
    self.xrange = np.asarray(xrange);

    # test xrange
    xmin,xmax = xrange;
    assert xmin < xmax;
    # are real roots in our xrange ? (sign change of disp -> not invertible)
    for x0 in self.disp.r:
      if x0.imag==0 and xmin < x0 < xmax:
        raise ValueError('dispersion polynom changes sign in xrange as \n' +
             '  it includes one of the roots: ' + str(self.disp.r) + '.\n'  +
             '  Choose a smaller xrange where the trafo becomes invertible!');

    # handle different cases for xrange: 
    #  use newton-root finding in transform if we have infinite interval
    #  use brentq for finite interval
    if np.isinf(xmin) and np.isinf(xmax):       # everywhere monotonic
      self._newton_x0 = 0;
    if np.isinf(xmin) and np.isfinite(xmax):    # upper bound
      self._newton_x0 = xmax-10;
    if np.isfinite(xmin) and np.isinf(xmax):    # lower bound
      self._newton_x0 = xmin+10;
    if np.isfinite(xmin) and np.isfinite(xmax): # u(x) is monotonic on interval
      self._newton_x0 = None;                   # -> use brentq

  def _inv_disp(self,x):
    return 1./self.disp(x);

  def scale_u(self,fact):
    "change size of scale DU used for dispersion measurement u(x) -> fact*u(x)"
    self.scale *= fact;

  def transform(self,u,v):
    y = np.atleast_1d(v);
    u = np.asarray(u,dtype=float).flatten();
    x = np.empty_like(u);
    for n in range(u.size):
      fn  = lambda x: self.inverse(x,0)[0]-u[n];  # fn(x) = u(x) - u_n
      try:
        if self._newton_x0 is None:               
          x[n]= brentq(fn, *self.xrange);         # solve fn(x)=0 in xrange
        else:
          x[n]= newton(fn,self._newton_x0);       # solve fn(x)=0 close to x0
      except:
        x[n]= np.nan;                             # could not find root in xrange
      # print 'NonlinearDispersion: n,u,x = ', n, u[n], x[n]

    if y.size==1: return(x[0],y[0]);              # convert to scalar
    return (x.reshape(y.shape),y);

  def inverse(self,x,y):
    v = np.atleast_1d(y);
    x = np.asarray(x,dtype=float).flatten();
    # test xrange
    if np.any(self.xrange[0]>x) or np.any(x>self.xrange[1]):
      xerr = x[np.logical_or(self.xrange[0]>x, x>self.xrange[1])]
      raise ValueError("the following x-values are not in xrange [%g,%g]:\n" % 
                       tuple(self.xrange) + "   x = %s \n " % str(xerr) +
                       "  Limit the x-coordinates to the allowed range.");
    u  = np.empty_like(x);
    for n in range(x.size):
      # u[x] = DU * Int_0^x dx' 1/DX(x'), DU=self.scale
      u[n] = self.scale*quad(self._inv_disp,0,x[n])[0];
    if v.size==1: return (u[0],v[0]);             # convert to scalar
    return (u.reshape(v.shape), v);

  def info(self,verbosity=0):
    ret = "NonLinearDispersion";
    if verbosity>0: ret+= "\n|- disp:   "+str(self.disp.coeffs).replace('\n','\n|          ');
    if verbosity>0: ret+= "\n|- scale: %8.3g " % self.scale;
    if verbosity>1: ret+= "\n|- xrange: (%8.3g,%8.3g) " %tuple(self.xrange);

    return ret;


class TrapezoidalDistortion(Transformation):
  """
  Transformation of slit coordinates (u,v) to distorted coordinates
  (x,y) in E-q maps using linear scaling of u to vanishing point vp.
  """

  def __init__(self,vp):
    """
    vp ... vanishing point (x,y)
    """
    self.vp=np.asarray(vp,dtype=float);

  def transform(self,u,v):
    " transform normalised coordinates (u,v) to distorted coords (x,y)"
    u,v= np.atleast_1d(u,v);
    x = u+(self.vp[0]-u)*(v/self.vp[1]);
    y = v;
    return x,y

  def inverse(self,x,y):
    " transform distorted coords (x,y) to normalised coordinates (u,v) "
    x,y= np.atleast_1d(x,y);
    u = (x*self.vp[1] - y*self.vp[0])/(self.vp[1]-y);
    v = y;
    return u,v;

  def info(self,verbosity=0):
    ret = "Trapezoidal Distortion";
    if verbosity>0: ret+= "\n|- vanishing point (x0,y0): (%g,%g)"%(self.vp[0],self.vp[1]);
    return ret;


class PolynomialDistortion(Transformation):
  """
  Transformation of slit coordinates (u,v) to distorted coordinates
  (x,y) in E-q maps using non-linear scaling of u-coordinates in 
  dependence of v:

   x(u,v) = Sum_kl  C_kl u^k v^l
  """

  def __init__(self,coeff,urange=(-np.inf,np.inf),T=None):
    """
    coeff ... coefficients C_kl for the non-linear transform
              array of shape (K,L)
    optional parameters for refining multiple solutions in inverse():
    urange... (opt) range of allowed u-values 
    T     ... (opt) Transformation object with approximate trafo
    """
    self.coeff     = np.asarray(coeff);
    self.K, self.L = self.coeff.shape;
    self.urange    = urange;
    self.approx_trafo=T;
    
  def transform(self,u,v):    
    # calculate coefficients C_l(u), shape(l,n)  n...number of coord (u,v)
    u,v= np.atleast_1d(u,v);
    cl = np.sum([np.outer(self.coeff[k],u**k) for k in range(self.K)],axis=0);
    cl = cl.reshape((self.L,)+u.shape);  # restore shape of flattened array u, see outer()
    x  = np.sum([cl[l]*v**l for l in range(self.L)], axis=0);
    y  = v;
    if y.size==1: return (x[0],y[0]);             # convert to scalar
    return x,y

  def set_urange(self,umin,umax): self.urange=[umin,umax];
 
  def set_approx_trafo(self,T): self.approx_trafo=copy.deepcopy(T);

  def inverse(self,x,y):
    """
    transform distorted coords (x,y) to normalised coordinates (u,v) 
    range ... (opt) specifies valid range for u-parameter (to select solution)
    """
    v = np.atleast_1d(y);
    x = np.asarray(x,dtype=float).flatten();
    y = np.asarray(y,dtype=float).flatten();
    assert x.size==y.size;

    # calculate coefficients C_k(v), shape(k,n) n...number of coord (x,y)
    ck = np.sum([np.outer(self.coeff[:,l],v**l) for l in range(self.L)],axis=0);
    u  = np.empty_like(x); u.fill(np.nan);
    for n in range(x.size):
      roots  = (np.poly1d(ck[::-1,n])-x[n]).r;  # possible solutions for u
      real   =  roots[ roots.imag==0 ].real;    # select real solutions
      inrange=   real[ self.urange[0] <= real ];
      inrange=inrange[ self.urange[1] >= inrange ];
      if len(inrange)==1:       # found solution
        u[n] = inrange[0];
      elif len(inrange)>1:      # multiple solutions
        if self.approx_trafo is not None: 
          up,vp = self.approx_trafo.inverse(x[n],y[n]);
          u[n] = min(inrange,key=lambda _u: (_u-up)**2);  # least distance from approx. solution
        else:
          print "WARNING: PolynomialDistortion.inverse() found several solutions for inverse of \n"+\
                "  (x,y) = (%g,%g), roots = "%(x[n],y[n]) + str(roots) + "\n"\
                "  You might want to restrict the solutions using the 'urange' or 'T' parameter.";
      elif len(inrange)==0:      # no solution found
          print "WARNING: PolynomialDistortion.inverse() could not find inverse of \n"+\
                "  (x,y) = (%g,%g), roots = "%(x[n],y[n]) + str(roots);

    # brentq(lambda u: self.u2v(u)-v, -0.1, 1.1);

    if v.size==1: return (u[0],v[0]);  # convert to scalar
    return u.reshape(v.shape),v;       # restore shape of flattened array u

  def info(self,verbosity=0):
    ret = "Polynomial Distortion";
    if verbosity>0: ret+= "\n|- c_kl = " + str(self.coeff).replace('\n','\n|         ');
    if verbosity>1: 
      ret+= "\n|- urange = (%g,%g)"%tuple(self.urange);
      if self. approx_trafo is not None:
        ret+= "\n|- approx. trafo: " + self.approx_trafo.info(verbosity).replace('\n','\n|  ');
    return ret;


# Matrix operators
class Inv(Transformation):
  """
  Inversion of a Transformation object: 
   Inv(T).transform(u,v) = T.inverse(u,v)
  """
  def __init__(self,T):
    self.T=T;
  def info(self, verbosity=0): 
    return "Inverse of "+self.T.info(verbosity);
  def transform(self,u,v): 
    return self.T.inverse(u,v);
  def inverse(self,x,y):   
    return self.T.transform(x,y);

class Seq(Transformation):
  """
  Sequence of several transformations: 
   Seq(R,S).transform(u,v) = R.transform( S.transform(u,v) )
  """
  def __init__(self,*args):
    " sequence of transformation objects "
    self.Tsequence=[copy.deepcopy(T) for T in args];

  def info(self, verbosity=0): 
    ret="Sequence of following %d Transformations:"%len(self.Tsequence);
    for T in self.Tsequence:
      ret+="\n|- "+T.info(verbosity).replace('\n','\n|  ');
    return ret;

  def transform(self,u,v): 
    for T in reversed(self.Tsequence):   # (R*S)(x) = R(S(x))
      u,v = T.transform(u,v);
    return u,v;     

  def inverse(self,x,y):   
    for T in self.Tsequence:             # (R*S)^(u) = S^(R^(u))
      x,y = T.inverse(x,y);
    return x,y

class I(Transformation):
  " Identity "
  def info(self, verbosity=0): return "Identity Transformation";
  def transform(self,u,v):     return u,v;
  def inverse(self,x,y):       return x,y;



# -- main ----------------------------------------
if __name__ == '__main__':
  import pickle

  # test grids
  x=np.arange(-1000,4000,33);
  y=np.arange(-500, 4500,23);
  X,Y=np.meshgrid(x,y);

  # normalize: shift and scaling
  norm = Normalize(99,100,34,3982,1.2); print norm.info(3);
  assert np.allclose((0,0),norm.inverse(*norm.transform(0,0)));  # single tuple
  assert np.allclose((X,Y),norm.transform(*norm.inverse(X,Y)));  # list of tuples
  assert np.allclose(((0,1,0),(0,0,1/1.2)), 
    norm.inverse(np.asarray((99,99+3982-34,99)), np.asarray((100,100,100+3982-34))) );

  # NonlinearDispersion
  u=np.arange(0,1,0.07);
  U,V=np.meshgrid(u,[0,1]);
  disp = NonlinearDispersion([-1,2,10],scale=1.2); print disp.info(3);
  assert np.allclose((0,0),disp.inverse(*disp.transform(0,0)));  # single tuple
  assert np.allclose((0,0),disp.transform(0,0));                 # u(0)=0
  assert np.allclose((U,V),disp.inverse(*disp.transform(U,V)));  # list of tuples
  
    
  # test operators
  T=Inv(norm); print T.info(3);
  assert np.allclose((X,Y),T.inverse(*T.transform(X,Y)));         # ~T*T=1
  assert np.allclose((X,Y),norm.transform(*T.transform(X,Y)));    # norm*T=1
 
  R=norm; S=Normalize(2222,200,34,982,1); T=Seq(R,S); print S.info(3);
  assert np.allclose(T.transform(X,Y),R.transform(*S.transform(X,Y))); # T=R S
  assert np.allclose(T.inverse(X,Y),  S.inverse(*R.inverse(X,Y)));     # Ti=Si Ri
  

  # trapezoidal distortion
  trapz= TrapezoidalDistortion((0.3,5)); print trapz.info(3);
  assert np.allclose((0,0),Seq(Inv(trapz),trapz).transform(0,0)); # single tuple
  assert np.allclose((X,Y),trapz.transform(*trapz.inverse(X,Y))); # list of tuples
  assert np.allclose(((0.7,0.3,0.3),(0,5,5)),
    trapz.transform(np.asarray((0.7,0,1)),np.asarray((0,5,5))));

  # polynomial distortion as generalisation of trapezoidal distortion
  coeff= [[ 0, trapz.vp[0]/trapz.vp[1]], [1, -1/trapz.vp[1]]];    # coeff. for vanishing point 
  poly = PolynomialDistortion(coeff); print poly.info(3);   
  assert np.allclose((0,0),Seq(Inv(poly),poly).transform(0,0));   # single tuple
  assert np.allclose((X,Y),poly.transform(*trapz.inverse(X,Y)));  # trapz==poly

  # non-linear polynomial distortion
  coeff= [[ 0, trapz.vp[0]/trapz.vp[1], 0.2], [1, -1/trapz.vp[1], -0.4], [0, -3, 4]] 
  poly = PolynomialDistortion(coeff,(0,1)); print poly.info(3);   
  assert np.allclose((1,1),Seq(Inv(poly),poly).transform(1,1));   # single tuple
  assert np.allclose([(0,1),(0,0)],poly.inverse((0,1),(0,0)));    # identity at v=0
  print "intentional warning following:";
  assert np.allclose(((0, 0.68317609),(1,1)),np.nan_to_num(poly.inverse((0,1),(1,1))));
                                                                  # no inverse
  # test inverse() of PolynomialDistortion with approximate Trafo
  trapz= TrapezoidalDistortion((0.3,5)); 
  coeff= [[ 0, trapz.vp[0]/trapz.vp[1], 0.2], [1, -1/trapz.vp[1], -0.4], [0, -3, 4]] 
  poly = PolynomialDistortion(coeff,T=trapz); 
  assert np.allclose((1,1),Seq(Inv(poly),poly).transform(1,1));  # single tuple

  # test pickling
  FILE=open('test.dump','w');
  pickle.Pickler(FILE).dump(Seq(Inv(poly),poly));
  FILE.close();

  FILE=open('test.dump','r');
  S=pickle.Unpickler(FILE).load();
  FILE.close();

  # test print and identity
  assert np.allclose( (U,V), S.transform(U,V) );
  print S.info(3);

  # plotting
  poly.plot(np.mgrid[-1:1:20j,-1:1:20j]);
  plt.show();

  print " all tests passed ... "
