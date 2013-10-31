"""
  analysis of the energy dispersion using the distance between
  the zero-loss peak (ZLP) and a sharp plasmon peak (PL)

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
import matplotlib.pylab as plt

from TEMareels.tools.transformations import NonlinearDispersion
from TEMareels.ecal.fit_peak_pos import get_peak_pos

# plasmon energy for Al [H. Abe et.al., J Electron Microsc (41) 465 (1992)]
E_plasmon_Al = 15.23; # 15.23+-O.O19 @ room temperature 

class BeamEnergy:
  " correct BeamEnergy "
  def __init__(self): 
    from os.path import dirname, abspath, join
    self.path = dirname(abspath(__file__));
    self.filename = join(self.path,"beam_energy_correction_120210.dat");
    self.E, self.E_corr = np.loadtxt(self.filename).T;   

  def En(self,E_ref):
    """
    calculated actual beam energy for nominal value E_ref:
    
    RETURNS beam energy [eV], or 
            NaN if E_ref is not found in correction table
    """
    E = []; E_ref=np.atleast_1d(E_ref);
    for E0 in E_ref.flat:
      index = np.where( np.abs(self.E-E0) < 1e-6 )[0];
      E.append( E0+self.E_corr[index[0]] if len(index)==1 else np.nan );
    if len(E)==1: return E[0];
    else        : return np.reshape(E,E_ref.shape);


def get_dispersion(spectra, refname, DE=None, order=2, \
                      ret_data=False, verbosity=0, **kwargs):
  """
    calculate the position-dependent energy dispersion from 
    the distance between two peaks (ZLP and plasmon reference)

    spectra  ... file containing the spectrum image (Nspectra, Npx)
    refname  ... filename of reference spectrum for second peak
    DE       ... (opt) use energy width of shifted scale to calibrate 
                 the absolute energy axis, otherwise we use the nominal
                 distance of two ZLPs
    order    ... (opt) order of the fitting polynomial for the distances
    ret_data ... (opt) return (pos, diff)-points used for the fit
    verbosity... (opt) verbosity (0=silent, 2=plotting fit, 3=debug)

    RETURNS function object returning the energy dispersion for any 
            position on the detector
  """

  pos = []; diff = []; data = [];
  for filename in spectra:
   
    Eref,zl,pl  = get_peak_pos( filename, refname, verbosity=verbosity, **kwargs);
    # order zl and pl (zl should be left peak)
    if np.nansum(pl[:,0]-zl[:,0])<0:  zl,pl=pl,zl
   
    # remove NaN's in zl (fit failed) and BeamEnergy (no correction available)
    ind =~np.isnan(zl[:,0] + BeamEnergy().En(Eref));
    Eref=Eref[ind]; zl=zl[ind]; pl=pl[ind];

    # determine position-dependent peak distance
    data.append( [Eref,  zl, pl] );        # zl[:,0]=mean pos
    pos.append( (zl[:,0] + pl[:,0])/2 );   # average position on detector
    diff.append( pl[:,0] - zl[:,0] );      # distance between peaks dy
   
  # plotting
  if verbosity > 1:
    fig=plt.figure(); ax=plt.gca();
    plt.title("Position-dependent peak distance");
    for i in range(len(pos)):
      ax.plot(pos[i],diff[i], 'x', label="%s"%spectra[i].split('/')[-1].split('.dm3')[0]);
 
  # flatten lists (one level)
  pos  = np.asarray(sum(map(list,pos ),[])); 
  diff = np.asarray(sum(map(list,diff),[]));
  
  # fitting polynomial to the appearent size of the scale
  ind = ~np.isnan(pos);            # index for outliers
  disp= np.poly1d(np.polyfit(pos[ind],diff[ind],order));

  if verbosity > 1:
    plt.gca().plot(pos[~ind],diff[~ind], 'rx', label="outliers");
    plt.gca().plot(disp(np.arange(4096)), 'k-', label="fit");
    ax.set_xlabel("position on detector [px]");
    ax.set_ylabel("distance between ZLP and plasmon [px]");
    #ax.set_ylim(460,475);
    ax.legend(loc=0);
 
  # Calculate energy width of scale from the linear shift of the ZLP 
  # with (corrected) beam energy
  if DE is None:

    # for each data series with N shifted spectra, we calculate the
    # missing scaling factor DE from the expected shift of the ZLPs
    DE = [];
    for Eref,zl,pl in data:   #  data[series][quantity][Nspectra]
      
      # expected energy shift for n-th ZLP compared to initial pos.
      Ecorr=-BeamEnergy().En(Eref); # energy scale inversed wrt. E-q map
      Y    =Ecorr[0]-Ecorr[1:];
      # measured energy shift for scale DE=1
      e2x=NonlinearDispersion(disp,scale=1);
      en =e2x.inverse(zl[:,0],zl[:,0])[0]; 
      X  = en[0]-en[1:];
      #print Ecorr, Eref, Y, zl[:,0], en;
      # calculate scaling factor DE which minimizes residuals X*DE-Y
      # min(|X*DE-Y|^2) <=> 2(X*DE-Y)*X = 0 <=> DE=X*Y/X*X
      DE.append( np.dot(X,Y)/np.dot(X,X) );
      
    # ouput
    if verbosity>0:
      print "-- estimating scale DE ---------------------------"
      if verbosity>1:
        for i in range(len(spectra)):
          print "   optimal Eplasmon for '%s':   %8.5f eV " \
                  % (spectra[i].split('/')[-1], DE[i]);
      print "\n Average Eplasmon:  (%5.3f +- %5.3f) eV" \
                  % (np.mean(DE), np.std(DE));

  # calculate energy dispersion dE/dy
  e2x=NonlinearDispersion(disp,scale=np.mean(DE));

  # DEBUG: test energy calibration for positions of ZLP:
  if verbosity>2:
    fig = plt.figure(); ax=fig.add_subplot(111);
    # reference curve for beam energy
    Beam= BeamEnergy();
    ax.plot(Beam.E,Beam.E_corr,'k-',label='reference');

    # beam energy from ZLP position and energy dispersion      
    for i,(Eref,zl,pl) in enumerate(data):

      E,_    = e2x.inverse(zl[:,0],zl[:,0]); 
      dE     =-(E    - E[0]);         # shift of ZLP from calibrated E-axis
      E_corr = dE + Beam.En(Eref[0]); # corrected Beam energy for each ZLP
      ax.plot(Eref,E_corr-Eref,'x',label=spectra[i].split('/')[-1]);
      ax.set_xlabel("nominal beam energy offset $E_0$ [eV]");
      ax.set_ylabel("estimated deviation $E - E_0$ [eV]");
      ax.set_ylim(-0.3,0.3);
      ax.legend(loc=0);

  # output transformation
  if verbosity > 1:
    print "Transformation to linear energy axis e2x: \n", e2x.info(3);
  return e2x if not ret_data else (e2x, np.vstack((pos[ind],diff[ind])));

# -- main ----------------------------------------
if __name__ == '__main__':

  spectra  = ["../tests/Eseries%i.tif" %i for i in range(1,4)];
  refname  = "../tests/reference.msa";
  e2x      = get_dispersion(spectra,refname,verbosity=5,order=2,ampl_cut=0.5);

  # test calibration of energy axis (consistency with previous results)
  e=np.arange(60,dtype=float);
  x=e2x.transform(e,e)[0];
  assert np.allclose( x[0:10], [ 0.,72.4690362,144.93702359,
        217.40522332,289.87489654,362.34730453,434.82370874,
        507.30537091,579.79355313,652.28951796]) ;

  # test beam energy
  Beam=BeamEnergy();
  assert Beam.En(np.ones((3,4))).shape == (3,4);
  assert Beam.En(0)==0;
 
  plt.show();
