"""
  fitting of peak positions in shifted EELS spectra for 
  energy-calibrations

  IMPLEMENTATION:
    - gauss fit for ZLP (highest peak in spectrum)
    - correlation with plasmon spectrum for second highest peak
      (The position corresponds to the center of the reference spectrum.)

  TODO:
    - make implementation more general: just fit left and right peak
       using either a reference spectrum or a model function

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig;
import scipy.optimize as opt;

from   TEMareels.tools.models import gauss;
from   TEMareels.tools.msa    import MSA;
import TEMareels.tools.tifffile as tiff;


def fit_zlp(spectra, medfilt_radius=5, verbosity=0, border=10, ampl_cut=0.5, sort=False):
  """ 
  fitting gauss to highest peak in spectrum 
  RETURNS
    (Nspectra,3)-array with fitting parameters (center, height, width)
  """

  if verbosity>2: print "-- fitting zero-loss peak ---------------------------";
  Nspectra, Npx = spectra.shape;
  x = np.arange(Npx);
  peaks = np.zeros((Nspectra,3));

  for s in xrange(Nspectra):

    line = sig.medfilt(spectra[s],medfilt_radius);
    imax = np.argmax(line); # initial guess for ZLP
    peaks[s], pconv = \
            opt.curve_fit(gauss,x,line,p0=(imax, np.sum(line[imax-5:imax+5]), 10));
    if verbosity>2: 
      print "#%03d:   "%s, "pos max: ", imax, ",  fit params: ", peaks[s]

  # remove outliers 
  # - peak height < 50% of max. amplitude in all spectra
  peaks = np.asarray(peaks);
  height = peaks[:,1];
  peaks[ height < ampl_cut*np.nanmax(height) ] = np.nan;
  # - peak pos close to the border (10px)
  pos   = peaks[:,0];
  peaks[ (pos<border) | (Npx - pos<border) ] = np.nan;

  # return sorted arrays
  if sort:
    i=np.argsort(peaks[:,0])[::-1]; 
    return peaks[i], spectra[i];
  else:
    return peaks, spectra;


def fit_plasmon(spectra, ref, xmin=None, xmax=None, medfilt_radius=5, border=10, ampl_cut=0.5, verbosity=0):
  """
    fitting reference peak by finding the best correlation with
    the original spectrum within a restricted range [xmin, xmax]

    NOTE: A gauss fit to the plasmon peak is rather poor due to 
          its assymetry. We need a precision of about 1px.

    RETURNS:
      (Nspectra,2)-array containing the position of the best overlap
          with respect to the center of the reference spectum and 
          the maximal intensity in the spectrum
  """
  if verbosity>2: print "-- fitting plasmon peak ------------------------------";
  Nspectra, Npx = spectra.shape;
  if xmin is None: xmin = np.zeros(Nspectra);
  else:            xmin = np.asarray(xmin,dtype=int);
  if xmax is None: xmax = np.ones(Nspectra)*Npx;
  else:            xmax = np.asarray(xmax,dtype=int);
  peaks = [[]]*Nspectra;


  
  for s in xrange(Nspectra):

    # skip lines, where no ZLP was found (nan is -2147483648 after conversion to int)
    if xmin[s]<0 or xmax[s]<0:
      peaks[s] = [np.nan, np.nan];
      continue;
    line = sig.medfilt(spectra[s],medfilt_radius);
    x    = np.arange(xmin[s],xmax[s],dtype=int);
    line = line[x];                   # region of interesst
    conv = sig.convolve(line,ref[::-1],'same');
    peaks[s] = [x[np.argmax(conv)], line.max() ];

    ## Alternatively: try to fit an (assymetric) model function
    #try:
    #  peaks[s], pconv = \
    #     opt.curve_fit(gauss,x,line,p0=(x[imax], line[imax], 50.));
    #except:                           # Catch any fitting errors
    #  peaks[s], cov = [np.nan,]*3, None
    #plt.plot(x,line); plt.plot(x,gauss(x,*peaks[s]));
    #plt.show();
    if verbosity>2:
      #print s, peaks[s]	
      print "#%03d:   pos max: %5s,  "%(s,peaks[0]), "fit params: ", peaks[s]

  # remove outliers 
  # - peak height < 50% of max. amplitude in all spectra
  peaks = np.asarray(peaks);
  height = peaks[:,1];
  peaks[ height < ampl_cut*np.nanmax(height) ] = np.nan;
  # - peak pos close to the border (10px)
  pos   = peaks[:,0];
  peaks[ (pos<border) | (Npx - pos<border) ] = np.nan;

  return peaks;

def plot_peaks(spectra, ref, zl, pl, filename=''):
  plt.figure();
  plt.title("Debug: Peak fitting for '%s'" % filename);
  plt.xlabel("y-position [px]");
  plt.ylabel("Intensity");

  Nspectra, Npx = spectra.shape;

  for s in xrange(Nspectra):
    scale = 1./spectra.max();
    offset= -s*0.1;

    # plot data
    plt.plot(spectra[s]*scale + offset,'k',linewidth=2);

    # plot first peak
    p,A,w = zl[s];
    x = np.arange(-2*w, 2*w) + p;
    plt.plot(x,gauss(x,*zl[s])*scale + offset,'r');
  
    # plot second peak
    if ref is not None:
      p,A   = pl[s];
      x = np.arange(len(ref)) - len(ref)/2 + p;
      plt.plot(x,ref/ref.max()*A*scale + offset,'g');

    #plt.xlim(xmin=0,xmax=Npx);


def get_peak_pos(filename, refname=None, medfilt_radius=5, sort=False, border=10, ampl_cut=0.5, verbosity=1):
  """
    calculate the position-dependent energy dispersion from 
    the distance between two peaks (ZLP and plasmon reference)

    filename ... file containing the spectrum image (Nspectra, Npx)
    refname  ... (opt) filename of reference spectrum for second peak
    medfilt_radius... (opt) median filter radius for smoothing of spectra
    sort     ... (opt) if True, sort spectra according to ZLP position
    border   ... (opt) skip peaks which are too close to the border (in pixel)
    ampl_cut ... (opt) skip peaks with amplitude smaller than ampl_cut*maximum
    verbosity... (opt) 0 (silent), 1 (minimal), 2 (plot), 3 (debug)

    RETURNS 
      x(N), zl(N) or 
      x(N), zl(N), pl(N) which are one-dimensional arrays of length N
       containing the x-value of the spectrum, the zero-loss and 
       plasmon-peak position.
       (N=Nspectra)
  """

  # 1. read EELS spectra of series
  if verbosity>0: print "Loading spectra from file '%s'"%filename;
  IN         = tiff.imread(filename); # Ny, Ns+1 
  data       = IN[:,:-1];
  x          = IN[:,-1];              # last line in image corresponds
                                      # to energie values
  
  # 2. fit ZLP to spectra
  zl,spectra = fit_zlp(data, border=border, medfilt_radius=medfilt_radius, 
                       ampl_cut=ampl_cut, verbosity=verbosity, sort=sort);

  if refname is None:
    if verbosity>2: plot_peaks(spectra, None, zl, None, filename=filename);
    return x,zl;

  # 3. fit second peak from correlation with reference spectrum
  spectra_noZLP=spectra.copy();
  for s in range(len(spectra)):  # for each spectrum, we remove the ZLP
    x0,I,fwhm = zl[s];           # parameters from ZLP
    xmin,xmax = max(0,x0-5*fwhm), min(len(spectra[s]),x0+5*fwhm);
    spectra_noZLP[s,xmin:xmax]=0;

  REF = MSA(refname).get_data();
  pl  = fit_plasmon(spectra_noZLP, REF, border=border,
      ampl_cut=ampl_cut, medfilt_radius=medfilt_radius, verbosity=verbosity);
  
  if verbosity>2: plot_peaks(spectra, REF, zl, pl, filename=filename);

  return x,zl,pl  




# -- main ----------------------------------------
if __name__ == '__main__':

  ref = "../tests/reference.msa"; # ref: maximum must be at the center !
  dat = "../tests/Eseries1.tif";  # spectra

  get_peak_pos(dat,ref, sort=False, border=80, verbosity=3);
  plt.show();
