"""
  Compare the energy dispersion estimated from the distance 
  between the ZLP and plasmon (energy_dispersion.py) and from
  the position of the ZLP or plasmon alone, if we correct the
  nominal beam energy (energy_dispersion_single_peak.py)
"""
### TODO: no longer working ! #########


# set package root dir (if not included in PYTHONPATH)
pkgdir = '../..'; 
import sys; sys.path.insert(0,pkgdir);

import numpy as np
import matplotlib.pylab as plt
import energy_dispersion_single_peak as ed_zlp
import energy_dispersion as ed

# FILENAMES
spectra  = ["test/series1.tif"];
refname  = "test/plasmon.msa";
Eplasmon=15.0;
dE      =1; # eV
order   =3;
correctE=False;

fig=plt.figure(); ax=plt.gca();

# reference dispersion
fit, data = ed.get_dispersion(spectra,refname,Eplasmon,order=order,\
                                border=100,ret_data=True,verbosity=0); 

x = np.arange(4096);
ax.plot(data[0],data[1]*1000,'k.',label="reference");
ax.plot(x,fit.inverse(x,x)[0]*1000,'k');
plt.show()

for n in (3,6,9):
  # ZLP
  print "Calculating energy dispersion for n=%d"%n;
  fit_zl, data_zl = ed_zlp.local_energy_dispersion(spectra,order=order,n=n,\
                               correctE=correctE,ret_data=True,verbosity=0); 
  fit_pl, data_pl = ed_zlp.local_energy_dispersion(spectra,order=order,n=n,\
               correctE=correctE,refname=refname,ret_data=True,verbosity=0); 
  pos = np.hstack((data_zl[0],data_pl[0]));
  disp= np.hstack((data_zl[1],data_pl[1]))*1000;
  ax.plot(pos,disp,'o',alpha=0.5,label="$\Delta E=%d$ eV"%(n*dE));
  
plt.title("Energy dispersion from the ZLP shift, with%s beam energy correction" \
                                             % ("" if correctE else "out"));
ax.set_xlabel("position on detector [px]");
ax.set_ylabel("energy dispersion dE/dx in [meV/px]");
ax.set_xlim(0,4096);
ax.set_ylim(15.6,16.6);
plt.legend(loc=0);
plt.show();

