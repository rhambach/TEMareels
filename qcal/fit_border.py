"""
  fitting border of aperture in wq-maps for momentum calibration

  IMPLEMENTATION:
    - gauss fit for ZLP (highest peak in spectrum)
    - correlation with plasmon spectrum for second highest peak
      (The position corresponds to the center of the reference spectrum.)

  TODO
    - improve fit of borders, e.g., use general function for
      intensity of projected round aperture

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
from scipy import ndimage
import matplotlib.pylab as plt

from TEMareels.tools.img_filter import medfilt1D


def threshold(line, threshold=0.1):
  """
    find smallest index in line > threshold 
  """
  ### TODO: improve !
  i=0;
  for i in range(len(line)):
    if line[i] > threshold: break;
  return i;

def outliers(x,f,iterations=2,dev=3):
  """ 
    remove outliers from a series of points by linear regression
    iterations ... number of regressions 
    dev        ... deviation from linear fit in units of the std. deviation

    RETURN boolean array indicating the outliers
  """
  bOut = np.isnan(f);
  for n in range(iterations):
    fit = np.poly1d(np.polyfit(x[~bOut],f[~bOut],1));
    dy  = (f - fit(x))**2;
    bOut[ dy > dev**2 * np.mean(dy[~bOut]) ] = True;
  return bOut;

def replace_nan(x,f,order=2):
  """
    replace NaNs from a series of points with polynomial fit
    order ... order of fitting polynomial (default: 2)
  """
  bOut = np.isnan(f);  # outliers
  if any(bOut):  
    fit  = np.poly1d(np.polyfit(x[~bOut],f[~bOut],order));
    f[bOut] = fit(x[bOut]);
  return f;

def get_border_points(img, rel_threshold=0.1, dev_bg=2, xmin=0, xmax=np.inf, ymin=0, ymax=np.inf, \
                      medfilt_radius=11, window_size=None, interp_out=False, verbosity=1, **kwargs):
  """
    Estimate x coordinates of the aperture border in wq-maps
    for each y-line of the (previously binned) image.

    img           ... 2D array image (Ny,Nx), binned along y
    rel_threshold ... (opt) relative threshold
    dev           ... (opt) deviation from linear fit w.r.t. std.dev.
    dev_bg        ... (opt) background noise level w.r.t. sdt.dev of noise
    medfilt_radius... (opt) filtering of input signal
    xmin,xmax     ... (opt) restricts image for fitting (px)
    ymin,ymax
    window_size   ... (opt) restrict to window around max intensity (px) 
    interp_out    ... (opt) bool indicating, if outliers (NaN's) shall be 
                            replaced with interpolated values ( 
    verbosity     ... (opt) verbosity (0=silent, 2=plotting fit, 3=debug)

    RETURNS left and right border as list of points (x,y) with length ymax-ymin 
  """
 
  # image axis order (y,x) 
  xmax=min(xmax,img.shape[1]); ymax=min(ymax,img.shape[0]);
  xmin=max(xmin,0);            ymin=max(ymin,0);
  Nx=xmax-xmin;                Ny=ymax-ymin; 
  assert Nx>0 and Ny>0;
  xleft=np.empty(Ny);          xright=np.empty(Ny);
  x = np.arange(xmin,xmax);

  # 1. try to find left and right border for each line y
  #    in the cropped image, i.e., ymin <= y < ymax, xmin <= x < xmax
  for iy,y in enumerate(np.arange(ymin,ymax)):

    # smoothing and background subtraction
    line = medfilt1D(img[y,xmin:xmax], medfilt_radius);
    out  = outliers(x,line,iterations=3,dev=2);  
    bg   = line[~out];      # linear background signal
    line-= np.mean(bg);

    # window around max intensity
    ixcenter= np.argmax(line);
    if window_size is not None:
      ixmin   = max( ixcenter-window_size/2, 0);
      ixmax   = min( ixcenter+window_size/2, Nx-1);
    else:
      ixmin = 0; ixmax = Nx-1;

    # find first point from left/right above threshold
    thresh = line[ixcenter] * rel_threshold;
    if thresh < dev_bg*np.std(bg):  # S/N too low (threshold below std.dev)
      xleft[iy] = xright[iy] = np.nan;
    else:
      # image coordinates !
      xleft[iy] = x[ ixmin + threshold(line[ixmin:ixcenter],   thresh)   ];
      xright[iy]= x[ ixmax - threshold(line[ixmax:ixcenter:-1],thresh) ];
    
    # DEBUG
    if verbosity > 10 and iy==(ymax+ymin)/2:
      plt.figure();
      plt.title("line at y = %d" % (Ny/2));
      plt.plot(img[y],'k-',label="intensity");
      plt.plot(x[~out],bg,'b-',label="background");
      plt.axhline(thresh+np.mean(bg),color='g');
      plt.axvline(xleft[iy],color='r');
      plt.axvline(xright[iy],color='r',label="border");
      plt.legend(loc=2);

  # 2. determine outliers
  y   = np.arange(ymin,ymax);
  iout_left  = outliers(y, xleft, **kwargs);
  iout_right = outliers(y, xright,**kwargs);

  # 3. plotting
  if verbosity>2:
    plt.figure();
    plt.title("Debug: Edge fitting");
    plt.xlabel("x [px]");
    plt.ylabel("y [px]");
    plt.imshow(np.log(1+np.abs(img)), cmap=plt.cm.gray, aspect='auto');
    #plt.imshow(img,vmin=0,vmax=np.mean(img)*10,cmap=plt.cm.gray, aspect='auto');
    plt.plot(xleft, y, 'g+', mew=2,label='border'); 
    plt.plot(xright,y, 'g+', mew=2);
    plt.plot(xleft[iout_left],  y[iout_left],'r+', mew=2,label='outliers');
    plt.plot(xright[iout_right],y[iout_right],'r+',mew=2);
    plt.gca().add_patch(plt.Rectangle((xmin,ymin),Nx,Ny,lw=3,ec='red',fc='0.5',alpha=0.2));
    plt.xlim(0,img.shape[1])
    plt.ylim(img.shape[0]-1,0)
    plt.legend(loc=2);

  # 4. calculate fitted points at outliers
  xleft[  iout_left ]  = np.nan;
  xright[ iout_right ] = np.nan;
  if interp_out:
    xleft = replace_nan(y,xleft);
    xright= replace_nan(y,xright);

  # 5. return border as list of (x,y)-points 
  return np.vstack((xleft,y)), np.vstack((xright,y));

# -- main ----------------------------------------
if __name__ == '__main__':
  from TEMareels.tools import tvips;

  datname = "../tests/qseries9.tif";
  image   = tvips.load_tiff(datname);  # img[iy,ix]
  get_border_points(image, rel_threshold=0.1, dev=3, xmin=222, ymin=-2, ymax=55, interp_out=True, verbosity=13);
  plt.show();
