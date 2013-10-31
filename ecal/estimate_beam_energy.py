"""
  estimation of the beam energy from the position of the ZLP
  taking into account the energy dispersion of the detector

  IMPLEMENTATION

    - get the position-dependent energy dispersion from 
       the distance between two peaks
    - estimate the energy shift of the beam from the
       position of the ZLP (including dispersion correction)

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
import matplotlib.pylab as plt

from TEMareels.ecal.energy_dispersion import get_dispersion;
from TEMareels.ecal.fit_peak_pos import get_peak_pos;

# calculate dispersion
spectra  = ["../tests/Eseries%i.tif" %i for i in range(1,4)];
refname  = "../tests/reference.msa";
order    = 2;
DE       = None;  # energy width of shifted scale [eV]
e2x      = get_dispersion(spectra,refname,verbosity=0,order=order,ampl_cut=0.5);

# calculate energy of ZLP for each of the series
E_ref = []; E_zl = []; E_pl = [];
for i,filename in enumerate(spectra):

  E,zl,pl=get_peak_pos(filename, refname, border=100, \
              ampl_cut=0.1, verbosity=2);     # positions [px]
  ind=~np.isnan(pl[:,0]-zl[:,0]);             # identify NaN's
  E  = E[ind]; zl = zl[ind]; pl = pl[ind]; 
  zl = e2x.inverse( zl[:,0], zl[:,0] )[0];    # energies  [eV]
  pl = e2x.inverse( pl[:,0], pl[:,0] )[0];
  s0 = np.nanargmax(zl);                      # remove offset
  E_zl.append( E[s0]+zl[s0]-zl );
  plOffset = np.mean(pl-zl) \
           if DE is None else DE;
  E_pl.append( E[s0]+zl[s0]+plOffset-pl);
  E_ref.append( E );
  
  #sE = np.argmin(np.abs(E-0));               # align at Energy E0
  #print sE, E_zl[-1][sE];
  #E_zl[-1] += E[sE]-E_zl[-1][sE];
  #E_pl[-1] += E[sE]-E_pl[-1][sE];

  # write to output file
  if False: 
    out = open('%s_beam_energy.dat'%(filename.split('.dm3')[0]),'w');
    out.write("# Estimated beam energy \n");
    out.write("# energy calibration: Eplasmon = %5.3f,"%Eplasmon);
    out.write(                 " order = %d, using files:\n"%order);
    for filename in spectra+[refname]: out.write("#  %s\n"%filename);
    out.write("# Eref [eV] Ezl-Eref [eV] Epl-Eref [eV]\n");
    for e in range(len(E)):
      out.write("%f %f %f \n" % (E_ref[i][e], E_zl[i][e]-E_ref[i][e], E_pl[i][e]-E_ref[i][e]));
    out.close();

# plot results
fig=plt.figure(); ax=plt.gca();
plt.title("Estimated Offset in Beam Energy");
# reference curve for beam energy
E, E_corr= np.loadtxt("./beam_energy_correction_120210.dat").T; 
ax.plot(E,E_corr,'k-',label='reference');
for s in range(len(spectra)):
  c = ['b','g','r','c','m','y','k'];
  name = spectra[s].split('/')[-1].split('.dm3')[0]; 
  ax.plot(E_ref[s],E_zl[s]-E_ref[s], 'o', alpha=0.7, \
          color=c[s%7], label="%s, ZLP"%name);
  ax.plot(E_ref[s],E_pl[s]-E_ref[s], '.', \
          color=c[s%7], label="plasmon" if s==len(spectra)-1 else "");
#plt.legend(loc=0);
ax.set_xlabel("nominal beam energy offset $E_0$ [eV]");
ax.set_ylabel("estimated deviation $E - E_0$ [eV]");
ax.set_ylim((-0.3,0.3));
ax.legend(loc=0);
plt.show();

# print results
print "\n Estimated Beam Energy [eV] and standard deviation";
print "    from ZLP                 from plasmon peak";
E_zl = np.asarray(E_zl); E_pl = np.asarray(E_pl);
for e in range(len(E_zl[0])):
  print "%8.5f +- %8.5f, \t" % (np.mean(E_zl[:,e]), np.std(E_zl[:,e])), 
  print "%8.5f +- %8.5f"     % (np.mean(E_pl[:,e]), np.std(E_pl[:,e])) 

plt.show();
