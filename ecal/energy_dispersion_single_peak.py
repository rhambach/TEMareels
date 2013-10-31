"""
  analysis of the energy dispersion using a single peak and the
  beam-energy correction data

  TODO
    allow uneven spacing between ZLP's!

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
import matplotlib.pylab as plt

import TEMareels.tools.transformations  as trafo
from   TEMareels.ecal.fit_peak_pos import get_peak_pos;
from   TEMareels.ecal.energy_dispersion import BeamEnergy;

def local_energy_dispersion(spectra, n=1, order=2, correctE=True, \
       refname=None, ret_data=False, verbosity=0, **kwargs):
  """
    calculate the position-dependent energy dispersion from 
    the shift of the ZLP with a change in beam energy

    spectra  ... file containing the spectrum image (Nspectra, Npx)
    n        ... (opt) use ZLP shift between spectrum i and i+n
    order    ... (opt) order of the fitting polynomial for the distances
    correctE ... (opt) if True, beam energy is corrected using BeamEnergy
    refname  ... (opt) filename of reference spectrum for second peak, 
                        which will be used instead of the ZLP
    ret_data ... (opt) return (pos, diff)-points used for the fit
    verbosity... (opt) verbosity (0=silent, 2=plotting fit, 3=debug)

    RETURNS - function object returning the energy dispersion for any 
                position on the detector
            - if ret_data is True, data used to create this function object 

  """
  # set up beam energy correction
  Beam = BeamEnergy();

  data = []; pos = []; diff = []; dE = []; 
  for filename in spectra:

    # determine zlp position for each beam energy
    if refname is not None:
      Eref,d,zl=get_peak_pos(filename,refname=refname, verbosity=verbosity-1, **kwargs);
    else:
      Eref,zl  = get_peak_pos( filename, verbosity=verbosity-1, **kwargs);

    # correct BeamEnergy
    E   = Beam.En(Eref) if correctE else Eref;

    # remove NaN's in zl (fit failed) and E (no correction available)
    ind =~np.isnan(zl[:,0] + E); 
    E   = E[ind]; zl=zl[ind]; 
    assert np.allclose( Eref[ind][1:] - Eref[ind][:-1], Eref[1]-Eref[0] )
                    # only even spacing in ZLP energy shifts allowed so far

    # calculate energy dispersion from difference in adjacent spectra
    data.append( [Eref[ind],  zl] );         # zl[:,0]=mean pos
    pos.append(  (zl[n:,0] + zl[:-n,0])/2 ); # average position on detector
    diff.append(-(zl[n:,0] - zl[:-n,0])   ); # distance between ZLP i and i+n 
    dE.append( E[n:] - E[:-n] );             # corresponding energy distance
  
  # TODO: allow for many spectra with uneven energy spacing
    dEref = Eref[n]-Eref[0];                 # nominal energy difference
    
  # plotting
  if verbosity > 1:
    fig=plt.figure(); ax=plt.gca();
    plt.title("Position-dependent peak distance"+\
              " (n=%d, $\Delta E_{ref}$=%4.1f eV)"%(n,dEref));
    for i in range(len(pos)):
      ax.plot(pos[i], diff[i], 'ko', alpha=0.1, label="uncorrected" if i==0 else "");
      ax.plot(pos[i], diff[i]*dEref/dE[i], 'o', markersize=6, 
                            label="%s"%spectra[i].split('/')[-1].split('.dm3')[0]);
 
  # flatten lists (one level)
  pos  = np.asarray(sum(map(list,pos ),[])); 
  diff = np.asarray(sum(map(list,diff),[]));
  dE   = np.asarray(sum(map(list,dE),[]));

  # fitting polynomial to the Energy-corrected appearent size of the scale
  ind = ~np.isnan(pos);            # index for outliers
  diff_corr = diff*dEref/dE;
  disp= np.poly1d(np.polyfit(pos[ind],diff_corr[ind],order));

  if verbosity > 1:
    plt.gca().plot(pos[~ind],diff[~ind], 'rx',  label="outliers");
    plt.gca().plot(disp(np.arange(4096)), 'k-', label="fit");
    ax.set_xlabel("position on detector [px]");
    if correctE: 
      ax.set_ylabel("corrected ZLP distance  $ \Delta x * (\Delta E_{ref} / \Delta E)$ [px]");
    else: 
      ax.set_ylabel("ZLP distance dx [px]");
    #ax.set_ylim(460,475);
    ax.legend(loc=0);

  # calculate energy dispersion dE/dy
  e2x=trafo.NonlinearDispersion(disp,scale=dEref,xrange=(0,4096));#

  # DEBUG: test energy calibration for positions of ZLP:
  if verbosity>2:
    fig = plt.figure(); ax=fig.add_subplot(111);
    # reference curve for beam energy
    Beam= BeamEnergy();
    ax.plot(Beam.E,Beam.E_corr,'k-',label='reference');

    # beam energy from ZLP position and energy dispersion      
    for i,(Eref,zl) in enumerate(data):

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
  return e2x if not ret_data else (e2x, np.vstack((pos[ind],diff_corr[ind])));




# -- main ----------------------------------------
if __name__ == '__main__':

  # test beam energy class
  Beam = BeamEnergy();
  assert(np.allclose(Beam.En(10),[ 10.04961]));
  assert(np.allclose(Beam.En([1,3,2]),[ 0.9465, 2.94671, 1.97699]));
  assert(np.all(np.isnan(Beam.En([3.5,60]))));

  # test dispersion
  spectra  = ["../tests/Eseries%i.tif" %i for i in range(1,4)];
  e2x      = local_energy_dispersion(spectra,order=2,n=5,verbosity=4); 
  #e2x      = local_energy_dispersion(spectra,order=2,n=5,correctE=False,verbosity=1); 
  plt.show();

