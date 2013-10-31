"""
     Common parameter conversions in the electron microscope
     
   VERSION

     $Id$
"""
import numpy as np;

# constants in atomic units, e=hbar=m=1/(4pi eps0)=1
c       = 137.0;           # a.u.
bohr    = 0.52917720859;   # Angstrom
Hartree = 27.211396;       # eV
E0 	= 510.9989;	   # rest mass of electron in keV

# conversion routines
def gamma(V):
  """  gamma = 1 / sqrt( 1-beta^2 ) """
  return 1./np.sqrt(1.-beta(V)**2);

def beta(V):
  """Calculates beta from acceleration voltage V [kV]
       beta^2 =  1 - ( E0/(eV+E0) )^2; E0=mc^2
  """
  return np.sqrt( 1- E0**2/(V+E0)**2 );

def k0(V):
  """Calculate wavenumber of the incident electron [1/Angstrom]
       k0 = p/hbar = (E beta)/(hbar c)
  """
  E =(V+E0)*1000/Hartree;                          # relativistic kinetic energy [a.u.]
  return  E * beta(V) / c / bohr;                  # in [1/Angstrom]
  # non-relativistic:
  #return 1000 * np.sqrt( 2*V*E0 ) / Hartree / c / bohr;
  # OK

def thetaE(dE,V):
  """Calculate angle theta_E [mrad] from energy loss dE [eV]
     and the acceleration voltage V [kV] in the small angle
     approximation:
     
       theta_E  =  dE / pv = dE / (E beta^2);   E = eV + E0
  """
  return np.asarray(dE)/(V+E0)/ beta(V)**2

def Qmin(dE,V):
  "minimal momentum transfer qmin=k0*thetaE [1/Angstrom]"
  return thetaE(dE,V) / 1000. * k0(V);


def thetaBethe(dE,V):
  """Calculate scattering angle corresponding to Bethe ridge 
     (classical scattering at a free electron at rest)
  
      theta_B = sqrt(2 theta_E)
  """
  return np.sqrt( 2 * thetaE(dE,V)*1000 );

def theta(Q,V):
  """Calculate angle theta [mrad] from perpendicular momentum
     transfer Q [1/Angstrom], and the acceleration voltage V [kV]
     in the small angle approximation:

       theta    =  ( c hbar Q ) / (E beta);     E = eV + E0;
  """
  E =(V+E0)*1000/Hartree;                      # relativistic kinetic energy [a.u.]
  Q =np.asarray(Q)*bohr;                       # in a.u. [1/bohr]
  return c*Q / E / beta(V) * 1000;             # in [mrad]
  
def Qperp(theta, V):
  #"Inverse of theta(Q,V)"
  return theta / 1000. * k0(V);


# -----------------------------------------------------------------
if __name__ == "__main__":

  V   =80;           # keV
  ang =1;            # mrad
  Qmax=Qperp(ang,V); # 1/A 
  print ang, Qmax, theta(Qmax,20)
  print k0(40);
  print thetaE(20,40)
  print Qmin(20,40)

