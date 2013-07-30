"""
   Circular on-axis objective and detector aperture in a STEM.
   The objective opening semi-angle alpha and the detector entrance
   semi-angle beta define the EELS signal.
   
   USAGE

     The angular correction function can be calculated as follows

     def APC(alpha,beta,E,E0):
       AP = TEM_wqslit(q0,dqx,dqy,E0);
       return AP.moment(0,E);

  Copyright (c) 2013, rhambach. 
    This file is part of the TEM-AREELS package and released
    under the MIT-Licence. See LICENCE file for details.
"""

import numpy as np;
from   aperture import Aperture
import tools.conversion as conv

class STEM_onaxis(Aperture):
  " Circular on-axis aperture in a STEM "

  def __init__(self,alpha,beta,E0):   
    """
      alpha ... convergence semi-angle for incoming beam [mrad] 
      beta  ... detector collection semi-angle [mrad]
      E0    ... beam energy [keV]
    """
    self.alpha = alpha;
    self.beta  = beta;
    self.qA    = conv.Qperp(alpha,E0)*conv.bohr; # qA = k0*alpha [a.u.]
    self.qD    = conv.Qperp(beta, E0)*conv.bohr; # qD = k0*alpha [a.u.]
    super(STEM_onaxis,self).__init__(E0);

  def __intersecting_circles(self,R1,R2,d):
    """
    RETURN intersection area for two circles with 
      radius R1 and R2 at distance d
    """
    # ensure standard order R1>=R2
    if R1<R2: (R1,R2) = (R2,R1);
  
    # CASE 1: circles do not intersect
    if R1+R2 <= d: return 0;
    
    # CASE 2: circle 2 completely in circle 1
    if d <= R1-R2: return np.pi*R2**2;
  
    # CASE 3: intersecting circles
    # [see e.g. http://mathworld.wolfram.com/Circle-CircleIntersection.html]
    r1 = (R1**2-R2**2+d**2)/2/d;    
    r2 = (R2**2-R1**2+d**2)/2/d;   # distance of origin from radical line
  
    C1 = R1**2 * np.arccos(r1/R1);   # area of circle segments 
    C2 = R2**2 * np.arccos(r2/R2);   #  defined by radical line
  
    T  = d*np.sqrt(R1**2-r1**2);   # area of the rhombus spanned by circle 
                                   #  origins and intersection points
    return C1+C2-T;
    
  def p(self,q):
    """
    Momentum probability distribution p(|q|) for electrons entering the 
    circular on-axis aperture with opening semi angle beta
               /        /
     p(|q|) =  |  D(kf) | |A(kf-q)|^2 delta(|q|-|q'|) dkf dq'
               /        /
     D(kf) = Theta( qD - |kf| )   qD = k0*beta
     A(ki) = Theta( qA - |ki| )   qA = k0*alpha
    """

    # scalar version
    if np.isscalar(q):
      return 2*np.pi * q * self.__intersecting_circles(self.qA,self.qD,q);
   
    # parallel version
    else:
      area= [ self.__intersecting_circles(self.qA,self.qD,_q) for _q in q ];
      return 2*np.pi * q * np.asarray(area);

  def qmin(self):
    " minimal |q| with p(|q|) > 0 [a.u.]" 
    return 1e-5;  # avoid q=0 

  def qmax(self):
    " maximal |q| with p(|q|) > 0 [a.u.]"
    return self.qA+self.qD;


# -- main ----------------------------------------
if __name__ == '__main__':
  import matplotlib.pylab as plt;

  print " TEST1: intersection of two circles "
  AP = STEM_onaxis(0,0,0);
  intersecting_circles = AP._STEM_onaxis__intersecting_circles; ## private function
  plt.figure();
  plt.title("TEST1: Intersection of two circles with radius 1 and 2");
  d = np.arange(0,4,0.1);
  A = [intersecting_circles(1,2,_d) for _d in d];
  plt.plot(d,A);
  plt.axhline(np.pi, color='k');
  plt.xlabel("distance d between circles");
  plt.ylabel("Intersection area");

  print " TEST2: statistics for contributing momentum transfers"
  alpha=30; E=20; E0=60;
  fig=plt.figure(); ax=fig.add_subplot((111));
  plt.title("Contributions of different momentum transfers to STEM-EELS signal");
  q = np.arange(0.001,0.5,0.001)*conv.bohr; # [a.u.]
  for beta in (15,35,48,76):
    AP = STEM_onaxis(alpha,beta,E0);
    w  = AP.weight_q(q,E);
    ax.plot( q / conv.bohr, w, label="beta=%d"%beta );
  plt.xlabel("Momentum transfer q [1/A]");
  plt.ylabel("Weight in EELS signal");
  plt.legend(loc=0);
  plt.show();
