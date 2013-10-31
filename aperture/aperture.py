"""
   Interface for different apertures
"""
import numpy as np
import conversion as conv
from   scipy.integrate import quad

class Aperture(object):
  """
    Abstract base class for all special apertures 
    methods used: p(q), qmin(), qmax()
  """

  def __init__(self,E0):
    self.E0 = E0;    # beam energy in [keV] 

  # abstact methods (to be implemented in child class)
  def p(self,q):
    """
    Momentum probability distribution p(|q|) for electrons entering the 
    rectangular aperture ow width dqx * dqy at momentum q0
               / q0+dqx/2  / dqy/2
     p(|q|) =  |           |        delta(|q|-sqrt(qx^2+qy^2) dqx dqy
               / q0-dqx/2  / -dqy/2
    """
    raise NotImplemented('abstract method');

  def qmin(self):
    " minimal |q| with p(|q|) > 0 [a.u.]" 
    raise NotImplemented('abstract method');

  def qmax(self):
    " maximal |q| with p(|q|) > 0 [a.u.]"
    raise NotImplemented('abstract method');

  # methods
  def weight_q(self,q,E):
    """ 
    Weight of loss function -Imag 1/eps(|q|,E) in the total EELS signal
      EELS      = - Int d|q| w(|q|,qE) * Imag 1/eps(|q|,E)
      w(|q|,qE) = p(|q|) / (|q|^2 + qE^2),   qE ... on-axis momentum transfer

      q ... perpendicular momentum transfer |q| [a.u.]
      E ... Energy loss [eV]
    """ 
    qE = conv.Qmin(E,self.E0)  * conv.bohr;  # [a.u.]
    return 1. / (q**2 + qE**2) * self.p(q);
  
  def moment(self,n,E):
    """
    Calculate moment  <q^n> = Int d|q| w(|q|,qE) |q|^n [a.u.]
    """ 
    def f(q):
      return q**n * self.weight_q(q,E); # polar coord.
    F, err = quad(f,self.qmin(),self.qmax());
    assert(abs(err)<1e-5);
    return F;
  
  def mean(self,E):
    """
    Calculate average momentum transfer Int d|q| w(|q|,E) |q| 
    for electrons entering the aperture [a.u.]
    """ 
    return self.moment(1,E) / self.moment(0,E);
  
  def var(self,E):
    " Calculate variance; see mean() for details"
    norm = self.moment(0,E);
    return self.moment(2,E) / norm  - (self.moment(1,E)/norm)**2
  

  def get_eels(self, Iepsi, dq=0.01, qmax=None):
    """
    Calculate the Integral Int dq w(q,E) Iepsi(q,E) / Int dq w(q,E).

      Iepsi ... Object which returns Im 1/eps(q,E) for any q [a.u.], 
                  must provide the functions get_E() and get_eels(q[:])
      dq    ... (opt) integration step [a.u.]

    RETURNS
      eels as 1D array of length;
    """

    # loss function for discrete set of q's to sum over
    qmax = self.qmax() if qmax is None else qmax;
    q    = np.arange(self.qmin(), qmax, dq);
    E    = Iepsi.get_E();           # shape(nE)
    eels = Iepsi.get_eels(q);       # shape(nq,nE)
    assert( len(q) == eels.shape[0] );
    assert( len(E) == eels.shape[1] );

    # calculate weights for all q
    wq   = np.asarray([ self.weight_q(q[iq],E) for iq in range(len(q)) ]);
    APC  = dq*np.sum( wq, axis=0 );        # int dq w(q,E)
    eels = dq*np.sum( wq*eels, axis=0 );   # int dq w(q,E) Iepsi(q,E)
 
    return eels/APC;

  def get_APC(self,E):
    " RETURN aperture correction function [a.u.] for given energies [eV]"
    return self.moment(0,E);
