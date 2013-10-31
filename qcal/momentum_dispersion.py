"""
  Analysis of the q/x dispersion from the borders of
  a small, round aperture in momentum space / real space.

  TODO
    - polynomial distortion fails if dispersion has no zero

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.

"""
import copy

import numpy as np
import scipy.optimize as opt;
import matplotlib.pylab as plt

import TEMareels.tools.tifffile as tif
import TEMareels.tools.transformations as trafo
from TEMareels.tools.img_filter import gaussfilt1D
from TEMareels.gui.wq_stack import WQBrowser, WQStackBrowser
from TEMareels.qcal import fit_border
from TEMareels import Release

class QDispersion:
  """
    Determine the distortions of the energy filter from a series
    calibration images.  To this end, we record a series of E-q maps
    of a small round aperture at different positions in the
    filter-entrance plane and fit the borders (iso-q-lines). In
    diffraction mode, the illumination of the aperture is very
    inhomogeneous and should be corrected by a reference.

    The q-dispersion is extracted from the change of the appearent
    size of the aperture (distance between left and right border).

    The following steps have to be performed:

     1. normalization of pixel coordinates (x,y) with respect to 
        spectrum magnification and shift, we use the (x,y)-positions 
        of the slit borders and the position of the direct beam 
        -> normalized coordinates (u,v)

     2. fitting of the measured iso-q-lines in order to correct
        the trapezoidal distortion in the image 
        -> slit coordinates (s,t)

     3. calculate change of appearent aperture size and linearize q-axis
        -> linearized q-coordinates (q,r)

  """

  def __init__(self,ap_series,illu_ref=None,reverse_y=False,N=4096,verbosity=1):
    """
    ap_series ... names of tif-images of the shifted aperture, shape (Nap,)
    illu_ref  ... (opt) name of reference to correct non-homogeneous illumination
    reverse_y ... (opt) True if y-axis should be inversed
    N         ... (opt) number of pixels of camera
    verbosity ... (opt) 0: silent, 1: minimal, 2: verbose, >10: debug
    """
    self.Npx = N;
    self.ap_names = ap_series;
    self.ref_name = illu_ref;
    self.verbosity= verbosity
    self.history  = [];

    # load image files
    self.ref_img  = tif.imread(illu_ref,verbosity=verbosity) \
                            if illu_ref is not None else None; # Ny,Nx
    self.ap_stack = tif.imread(ap_series,verbosity=verbosity); # Nap,Ny,Nx

    # reverse images
    if reverse_y:
      print "WARNING: in QDispersion: reverse_y==True";
      self.ref_img= self.ref_img[::-1];
      self.ap_stack=self.ap_stack[:,::-1];

    # set image parameters
    self.Nap, self.Ny, self.Nx = self.ap_stack.shape;  
    self.ybin,self.xbin        = self.Npx/float(self.Ny), self.Npx/float(self.Nx);
    self.crop_img();

    # transformations (identity by default)
    self.u2x = trafo.I();  # normalised coordinates
    self.s2u = trafo.I();  # slit coordinates = distortions of iso-q-lines
    self.q2s = trafo.I();  # non-linear dispersion on q-axis

    # History + DEBUG
    self.history  = ["momentum_dispersion.py, version %s (%s)" % 
                     (Release.version, Release.version_id)];
    self.__dbg_fig=[];     # list of figures

  def crop_img(self,xmin=0,xmax=np.inf,ymin=0,ymax=np.inf):
    """
    reduce image size for fitting and resampling 
    xmin,xmax,ymin,ymax are given in image pixels between 0 and N
    """
    xmax=min(xmax,self.Npx);
    ymax=min(ymax,self.Npx);
    self.crop={'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax};
    self.history.append("Crop");
    param = ", ".join([key+": %d"%val for key,val in self.crop.items()])
    self.history.append("|- "+param);
    assert xmax-xmin > self.xbin
    assert ymax-ymin > self.ybin

  def fit_aperture_borders(self,order=2,log=False,**kwargs):
    """
    Determine the left and right border of the aperture as polynom x(y) 
    for the series of E-q maps.     
  
    order     ... (opt) order of the polynomial to fit
    log       ... (opt) consider aperture images on log-scale
    for further options, see fit_border.get_border_points()
  
    RETURNS list of tuples(left,right) containing polynomials x = left(y)
    """
    self.ap_order =order;
    self.ap_log   =log;
  
    # illumination reference (smoothed version) 
    if self.ref_name is not None:
      ref = np.abs(gaussfilt1D(self.ref_img,11));      # gauss-filter
      #ref = np.abs(lorentzfit1D(self.ref_img,offset=offset));  # fit lorentz
      ref[ref<np.mean(ref)] = np.mean(ref);               # constant for low values 
                                                          # to avoid 0 devision
      if self.verbosity>9:  
        self.__dbg_fig.append(self.plot_reference(ref));  # draw figure and save in list 
    else:
      ref = 1;                                            # no reference given
   
    # iterate over all aperture images
    points = []; fit = []; lines=[]; stack=[]; info=[];
    for i,image in enumerate(self.ap_stack):

      # correct image by illu_ref
      filtimg = image/ref;                                
      if log: filtimg = np.log(np.abs(filtimg)+1);
  
      # get aperture border as point list in px (correct for binning!)
      c = self.crop;
      l,r   = fit_border.get_border_points(filtimg, interp_out=True,
                xmin=int(c['xmin']/self.xbin), xmax=int(c['xmax']/self.xbin),
                ymin=int(c['ymin']/self.ybin), ymax=int(c['ymax']/self.ybin), 
                verbosity=self.verbosity-10, **kwargs);
      l = ((l+0.5).T*(self.xbin,self.ybin)).T;    # convert to px, points (x,y)
      r = ((r+0.5).T*(self.xbin,self.ybin)).T;
      points.append([l,r]);  

      # fit polynom x=p(y) to left and right border
      polyl = np.poly1d(np.polyfit(l[1],l[0],order));
      polyr = np.poly1d(np.polyfit(r[1],r[0],order));
      fit.append((polyl,polyr));
  
      # DEBUG: drawing fit points and polynoms
      if self.verbosity>2:
        y  = np.arange(self.crop['ymin'],self.crop['ymax']);
        stack.append(filtimg);
        info.append({'desc': 'DEBUG: '+self.ap_names[i], 
                     'xperchan':self.xbin, 'yperchan':self.ybin});
        p1 = plt.Line2D(l[0],l[1],marker='x',ls='',c='b'); 
        p2 = plt.Line2D(r[0],r[1],marker='x',ls='',c='b')
        p3 = plt.Line2D(polyl(y),y,color='r');
        p4 = plt.Line2D(polyr(y),y,color='r');
        lines.append([p1,p2,p3,p4]);
    if self.verbosity>2:
      self.__dbg_fig.append( self.plot_aperture_images(stack,info,lines) );
    
    # store results
    self.ap_points=points;       # contains Nap (l,r) tuples; l,r are a lists of (x,y) points
    self.ap_poly  =fit;          # contains Nap (pl,pr) tuples; pl,pr are polynoms y(x)

    # comments
    self.history.append("Fit aperture borders");
    self.history.append("|- order=%d,  log=%s"%(order,log));
    params = ", ".join([key+": "+str(val) for key,val in kwargs.items()]);
    self.history.append("|- " + params);

    return fit;

  def plot_reference(self,ref=None):
    if self.ref_name is None: 
      print 'WARNING: in QDispersion.plot_reference(): no reference specified.';
      return;
    if ref is None: ref=self.ref_img;
    info = {'desc': 'DEBUG: illumination reference, '+self.ref_name,
            'xperchan':self.xbin, 'yperchan':self.ybin};
    return WQBrowser(ref,info,aspect='auto');
   
  def plot_aperture_images(self,stack=None,info=None,lines=None):
    if stack is None: stack=self.ap_stack;
    WQB = WQStackBrowser(self.ap_stack,info,lines,aspect='auto');
    c=self.crop; 
    WQB.axis.add_patch(plt.Rectangle((c['xmin'],c['ymin']), \
      c['xmax']-c['xmin'],c['ymax']-c['ymin'],lw=3,ec='red',fc='0.5',alpha=0.2));
    return WQB;

  def normalize_coordinates(self,x0,y0,xl,xr,aspect=None):
    """
    Transfrom from pixels (x,y) to normalised units (u,v):
      (x0,y0)->(0,0), (xl,y0)->(ul,0), (xr,y0)->(ur,0), ur-ul=1.  I.e.,
    to remove the influence of a spectrum shift on the camera, the new
    origin is set to the position of the direct beam; and remove an
    influence of the spectrum magnification, the length of the slit in
    the E-q map is normalized to 1.  By default, the y-axis is not
    scaled. Otherwise, the aspect dx/dy can be specified explicitly
    (use 1 for common magnification of x and y axis).

    x0    ... horizontal position of the direct beam [px]
    y0    ... vertical slit position [px]
    xl,xr ... horizontal left/right slit position [px]
    aspect... (opt) change of apect ratio (du/dv)/(dx/dy) 

    RETURNS Tranformation object transforming (x,y) to (u,v)
    """
    self.u2x = trafo.Normalize(x0,y0,xl,xr,aspect); # (u,v) -> (x,y)
    # save slit borders in normalized coordinates
    self.u2x.ul,_ = self.u2x.inverse(xl,y0);
    self.u2x.ur,_ = self.u2x.inverse(xl,y0); 

    # history
    self.history.extend(self.u2x.info(3).split("\n"));
    return trafo.Inv(self.u2x);

  def __resample_aperture_border(self,N):
    "  resample aperture border with n points and normalise "

    # sample points for aperture in image coordinates
    if N is None: # original points
      guides=np.reshape(self.ap_points,(2*self.Nap,2,-1));
      x   = guides[:,0];   
      y   = guides[:,1]; 
      assert np.allclose( y-y[0], 0);  # all y-postitions should be the same
    else:         # resample points from quadratic fit
      s   = np.linspace(self.crop['ymin'],self.crop['ymax'],N,endpoint=False); # sampling points
      x   = np.asarray([ q(s) for b in self.ap_poly for q in b ]); # x-position for iso-q-lines
      y   = np.tile(s,(x.shape[0],1));                             # y-position   "

    # convert to normalised coordinates    
    try:  
      u,v = self.u2x.inverse(x,y);    
    except NameError:
      print " ERROR: run fit_aperture_borders() first."; 
      raise;
    return u,v;  # shape (2*Nap,n)
 
  def __residuals(self,u,v,T,u0,leastsq=True):
    " residuals for fitting transformations to distorted lines "
    K,N  = u.shape; assert len(u0)==K; assert v.shape==(K,N)
    s,t  = np.tile(u0,(N,1)).T, v;          # undistorted coordinates, shape (K,N)
    uu,vv= T.transform(s,t);                # perform distortion trafo 
    if leastsq: return (u-uu).flatten();    # residuals u-u' for leastsq()
    return T, u0, u-uu;                     # return trafo, u0, residuals


  def __debug_distortion(self,T,u,v,u0,res,title=""):
    " debugging commands which are common for all fitting of distortions "
    K,N  = u.shape;
 
    if self.verbosity>2:                    # PRINT INFO
      print T.info(3);
    if self.verbosity>9:
      max_res = np.max(np.abs(res),axis=1);
      print ' line     u0-values    max. deviation ';
      for k in range(len(u)): 
        print '   %2d    %8.3g        %8.3g'%(k,u0[k],max_res[k]);

    if self.verbosity>2:                    # PLOT POINTS AND FIT
      vv  = np.linspace(-1000,18000,100);
      s,t = np.tile(u0,(len(vv),1)).T, vv;  # iso-q-lines
      U,V = T.transform(s,t);               # perform trafo
      fig = plt.figure(); plt.title(title);
      plt.plot(u.reshape(K/2,2*N).T,v.reshape(K/2,2*N).T,'x');   
                                            # input points (left and right together)
      plt.plot(U.T,V.T,'k-');               # fitted lines
      plt.xlabel('u');
      plt.ylabel('v');
      plt.xlim(-0.61,0.5);
      plt.ylim(vv.max(), vv.min());
      self.__dbg_fig.append(fig);


  def fit_trapezoidal_distortions(self, N=None, vp=None, u=None, v=None):
    """
    Least-square fitting of iso-q-lines (u,v) using rays with 
    common vanishing point (U,V), passing through points (u0,0)

    N  ... (opt) number of sampling points along y-direction for each image
    vp ... (opt) initial guess for the vanishing point (U,V)
  
    RETURN (trapz,u0)
    trapz... TrapezodialDistortion object 
    u0   ... 1d-array; slit coordinate for each iso-q-line
    """
    # data + initial parameters
    if u is not None and v is not None:       # data given explicitely
      u,v = np.atleast_2d(u,v); N=u.shape[1]; # K... number of iso-q-lines
    else:
      u,v = self.__resample_aperture_border(N)# aperture borders, shape (K,N)
    if vp is None: vp = (self.Npx/2,self.Npx);# set reasonable start value
    trapz = trafo.TrapezoidalDistortion(vp);  # initialize trafo
    param0= list(vp)+[0]*len(u);              # param: [vp, u0]
  
    # deviation of given trapezoidal distortion from observed values u,v
    def residuals(param,u,v,T,leastsq=True):
      T.vp = param[:2]; u0 = param[2:];       # fit parameters
      return self.__residuals(u,v,T,u0,leastsq);
  
    # perform fitting
    fit,_ = opt.leastsq(residuals, param0, args=(u,v,trapz));
    trapz,u0,res = residuals(fit,u,v,trapz,False);

    self.__debug_distortion(trapz,u,v,u0,res,
      title="DEBUG: fit_trapezoidal_distortions()");
    
    # save results, slit borders in slit-cooridinates
    self.s2u = trapz;
    self.s2u.sl=self.u2x.ul;
    self.s2u.sr=self.u2x.ur; # same as borders in normalized coords

    # history
    self.history.extend(self.s2u.info(3).split("\n"));

    return trapz,u0;

  def fit_polynomial_distortions(self,N=None,I=1,J=1,c0=None,const='fixed_slit'):
    """
      Least-square fitting of all iso-q-lines (u,v) by polynomial functions
      of order J along the energy axis t=v with the restriction, that the 
      coefficients of different iso-q-lines can be expressed as polynoms in q 
      of order I. This corresponds to a transformation T:(s,t)->(u,v) 

        u(s,t) = sum_ij C_ij s^i t^j;   v(s,t) = t.

      Note that also the exact position s_k of the k'th iso-q-line at E=0 
      (v=0) is not known exactly and included in the fit. As the fitting 
      parameters s_k and C_ij are not independent, we add further constraints 
      according to the parameter 'const'.

      N   ... (opt) number of sampling points along y-direction for each image
      J   ... (opt) degree of fit u = sum_j c_j v^j for a single aperture border
      I   ... (opt) degree of polynomial c_j = sum_i C_ij s^i for coefficients
      c0  ... (opt) initial guess for the coefficients C_ij, overwrites I,J 
      const.. (opt) constraints for fitting parameters:
              'fixed_slit': T(s,0) = (s,0), the trafo will not
                 change the coordinates at the slit position t=0;
              'aperture_calibration': T(0,0) = (0,0) to avoid shifts
                 and constant aperture size s[k/2+2]-s[k/2]=1 
  
      RETURN (poly,u)
        poly ... PolynomialDistortion object 
        s_k  ... 1d-array; slit coordinate for each iso-q-line
    """
    # data + initial parameters
    u,v = self.__resample_aperture_border(N);          # aperture borders, shape (K,N)

    # fit approximate trapezoidal distortions, used to
    # -> calculate initial parameters for polynomial fit
    # -> distinguish between multiple solutions in inverse() of PolynomialDistortion
    self.verbosity-=10;
    trapz,u0_trapz = self.fit_trapezoidal_distortions(self, u=u, v=v);
    self.verbosity+=10;

    # initial fit parameters
    if c0 is not None: c0 = np.asarray(c0,dtype=float);
    else: 
      c0 = np.zeros((I+1,J+1),dtype=float);    # i=0,...,I; j=0,...;J
      c0[0,:2] = [0, trapz.vp[0]/trapz.vp[1]];
      c0[1,:2] = [1, -1/trapz.vp[1]         ];
    I = c0.shape[0]-1; J = c0.shape[1]-1; 
    K = u.shape[0];        # K=2Nap (# of left+right aperture borders)

    # 1. FIXED-SLIT CONSTRAINTS
    #    restrictions for fitting C_ij and s_k: 
    #       T(s,0)=(s,0)   <=>   C_i0 = 1 if i==1 else 0
    #    remaining fit parameters:
    #       param[0:(I+1)*J]         ... C_ij for j=1,...,J
    #       param[(I+1)*J:(I+1)*J+K] ... s_k  for k=0,...,K-1
    if const=='fixed_slit':
      c0[:,0]= 0; c0[1,0] = 1;                           # => T(s,0) = (s,0)   
      poly   = trafo.PolynomialDistortion(c0,T=trapz);   # initialize trafo
      param0 = np.hstack((c0[:,1:].flatten(),u0_trapz)); # param: [c0, u0]
      
      def residuals(param,u,v,T,leastsq=True):
        T.coeff[:,1:] = param[:(I+1)*J].reshape(I+1,J);  
        s_k           = param[(I+1)*J:];
        return self.__residuals(u,v,T,s_k,leastsq);
   
    # 2. FIXED APERTURE SIZE 
    #    restrictions for fitting C_ij and s_k: 
    #       T(0,0)=(0,0)   <=>   C_00 = 0;
    #       s[k/2+1] - s[k/2] = 1 for all k 
    #       Note: k=0 mod K/2 for left border, k=1 mod K/2 for right border
    #    remaining fit parameters:
    #       param[0:Nc]      ... C_ij for all i,j except C_00, Nc=(I+1)(J+1)-1
    #       param[Nc:Nc+K/2] ... s_k for k=0,2,...,K
    elif const=='aperture_calibration':
      assert K%2==0;                             # even number of lines required
      poly   = trafo.PolynomialDistortion(c0,T=trapz);
      param0 = np.hstack((c0.flatten()[1:], u0_trapz[::2])); 
      DS     = np.mean(u0_trapz[1::2]-u0_trapz[::2]);
    
      def residuals(param,u,v,T,leastsq=True):
        T.coeff = np.insert(param[:(I+1)*(J+1)-1],0,0).reshape((I+1,J+1));   # C_00=0
        s_k     = np.asarray([[s,s+DS] for s in param[(I+1)*(J+1)-1:]]).flat;
        # set s[k+1]-s[k]=DS instead of 1 such that the total slit width remains close
        # to 1 like in the case of const='fixed slit' (exact value is not important)
        return self.__residuals(u,v,T,s_k,leastsq);
      
    else: raise ValueError("Parameter const='%s' is not allowed."%const); 

    # perform fitting
    fit,_ = opt.leastsq(residuals, param0, args=(u,v,poly));
    #fit = param0
    poly,s_k,res = residuals(fit,u,v,poly,False);
 
    self.__debug_distortion(poly,u,v,s_k,res,
          title="DEBUG: fit_polynomial_distortions(), %s, I=%d, J=%d"%(const,I,J));

    # save results and slit borders in slit coordinates
    self.s2u = poly;
    self.s2u.sl,_=poly.inverse(self.u2x.ul,0);
    self.s2u.sr,_=poly.inverse(self.u2x.ur,0); 

    # history
    self.history.extend(self.s2u.info(3).split("\n"));
    self.history.append("|- I=%d, J=%d, const=%s"%(I,J,const));
    return poly,s_k

  def linearize_qaxis(self,N=20,ord=2,dq=1):
    """
    Fit transformation
    N    ... (opt) number of sampling points along y-direction for each image
    ord  ... (opt) order of fitting polynomial
    dq   ... (opt) size of the aperture q-coordinates

    RETURNS aperture size and position in px, shape (k, n)
    """
    # 1. get undistorted coordinates of aperture borders
    u,v = self.__resample_aperture_border(N);          # aperture borders, shape (K,N)
    s,t = self.s2u.inverse(u,v);                       # correct distortions
  
    # 2. calculate apearent aperture size
    s   = s.reshape(self.Nap,2,N);     # shape (k,2,N)
    size = s[:,1] - s[:,0];            #  right-left
    pos  = 0.5*(s[:,1]+s[:,0]);        # (right+left)/2

    # 3. fit polynomial (common for all v-values)
    size_dispersion = np.poly1d(np.polyfit(pos.flatten(),size.flatten(),ord));
    if self.verbosity>2: # DEBUG: plot aperture size + quadratic fit
      smin,smax,slen = s.min(),s.max(),s.max()-s.min();
      x = np.mgrid[smin-0.1*slen:smax+0.1*slen:100j];
      fig=plt.figure(); 
      plt.title("DEBUG: Normalized aperture size for different y");
      plt.gca().set_color_cycle([plt.cm.winter(1.*i/N) for i in range(N)]); # continous colors
      plt.plot(pos,size,'o',alpha=0.5);
      plt.plot(x,size_dispersion(x),'k-');
      plt.xlabel("slit position s");
      plt.ylabel("appearent aperture size ds");
      self.__dbg_fig.append(fig);

    # 4. create transformation object (q,r) -> (s,t)
    self.q2s=trafo.NonlinearDispersion(size_dispersion,scale=dq);

    # 5. write history
    self.history.extend(self.q2s.info(3).split('\n'));

    # TEST: check positive dispersion within the slit
    if self.q2s.xrange[0]>=self.s2u.sl or self.q2s.xrange[1]<=self.s2u.sr: 
      print self.q2s.info(3);
      plt.show();
      raise ValueError("Unexpected xrange in QDispersion.linearize_qaxis().\n"\
         "Check polynomial fit of appearent aperture size using verbosity>2");
    if self.verbosity>2:
      print self.q2s.info(3); 
   
    # TEST: aperture size should be roughly dq in q coordinates
    q,r=self.q2s.inverse(s,t.reshape(self.Nap,2,N));
    qsize = np.mean(q[:,1]-q[:,0],axis=1); # average over energies

    # - deviation of single aperture from dq by >5%
    if not np.allclose(qsize,dq,rtol=0.05) and self.verbosity>0: 
      print "WARNING: in QDispersion.linearize_qaxis(): \n"+           \
            "  calculated aperture size deviates by more than 5% from scale dq: \n"+   \
            "  dq: %8.3f,  %8.3f < qsize < %8.3f \n  " % (dq,qsize.min(),qsize.max());
    # - variation of aperture size
    if np.std(qsize)/np.mean(qsize)>0.01 and self.verbosity>0:  # rel error > 1%
      print "WARNING: in QDispersion.linearize_qaxis(): \n"+           \
            "  calculated aperture size varies by more than 1%: \n"+   \
            "  mean(dq): %8.3g,  std(dq): %8.3g,  variation: %5.2f%%\n"\
              %(np.mean(qsize),np.std(qsize),100*np.std(qsize)/np.mean(qsize));
  
    return size,pos
  
  def get_q2u(self):
    """ 
      RETURN combined transformation from linearized coordinates 
         to normalized slit coordinates (q,r)->(s,t)->(u,v) 
    """
    return trafo.Seq(self.s2u,self.q2s);
 
  def get_absolute_qs(self,line,verbosity=3):
    """ 
      OLD!

      determine two points on y-axis with known q-distance
      (low-loss w-q reference with central spot and bragg spot)
      line ... 1D array with N-points containing two peaks
    """
    x  = np.arange(N,dtype='float');
    ref=gaussfilt1D(line, 5); peaks=[];
    for i in range(2):           # fit 2 peaks
      imax = np.argmax(ref);     # initial guess for peak
      p, pconv = \
        opt.curve_fit(models.gauss,x,ref,p0=(imax, np.sum(ref[imax-5:imax+5]), 10));
      peaks.append(p);           # gauss fit
      imin = max(p[0]-5*p[2],0);
      imax = min(p[0]+5*p[2],N);
      ref[imin:imax]=0;          # remove peak from line (5*fwhm around x0)
    
    if verbosity>2:
      plt.figure(); plt.title("DEBUG: Fit q-reference");
      plt.plot(x,line,'k');
      plt.plot(x,models.gauss(x,*peaks[0]),'r');
      plt.plot(x,models.gauss(x,*peaks[1]),'g');
  
    return peaks[0][0], peaks[1][0];

  def get_status(self):
    return "\n".join(self.history);




def calibrate_qaxis(q2s,sl,sr,G):
  """
    Calibration of q-axis with two symmetric Bragg spots -G,G.
    q2s   ... NonlinearDispersion object mapping linear. q to slit coordinates
    sl,sr ... slit coordinates of -G,G Bragg spot
    G     ... length of G in reciprocal units [1/A]
    Note: we test for consistency of sl and sr, the direct beam is at s=0;
   
    RETURNS: rescaled trafo q2s
  """
  # calculate linearized coordinates corresponding to u-values
  Q2s = copy.deepcopy(q2s);
  ql,_= q2s.inverse(sl,0);
  qr,_= q2s.inverse(sr,0);
  assert ql < 0 and qr > 0;
  assert np.allclose( (0,0), q2s.inverse(0,0) ); # direct beam at coordinate u=0=q
  
  # calculate scaling factor and check consistency
  q      =(qr-ql)/2.; 
  scale  = G/q;
  Q2s.scale_u(scale);   # change scale in NonlinearDispersion

  # check consistency (ql vs qr)
  rel_err=np.abs(qr-q)/q;
  if rel_err > 0.01 :                 # relative error of 1%
    print "WARNING in calibrate_qaxis(): left and right q-vector deviate:"
    print "  ql=%.3f, qr=%.3f, rel_err=%.1f%% " %(scale*ql,scale*qr, rel_err*100)

  return Q2s;


def fit_aperture_borders(ap_series,illu_ref=None,reverse_y=False,verbosity=1,offset=0,**kwargs):
  " wrapper for backward compatibility "
  QDisp=QDispersion(ap_series, illu_ref,reverse_y=reverse_y,verbosity=verbosity);
  QDisp.crop_img(ymin=offset);
  return QDisp.fit_aperture_borders(**kwargs);


# -- main ----------------------------------------
if __name__ == '__main__':
 try:
  # filenames
  aperture_files   = ["../tests/qseries%d.tif" % (i)  for i in range(1,10) if i<>2];
  ref_illumination = "../tests/qreference.tif";

  # fitting aperture borders + normalization
  QDisp=QDispersion(aperture_files, ref_illumination,verbosity=11);
  QDisp.crop_img(xmin=22, ymin=700);
  QDisp.fit_aperture_borders(rel_threshold=0.2);
  QDisp.normalize_coordinates(2371,1060,900,3850);

  # fit non-linear polynomial distortion + linearize q-axis
  poly,u0 = QDisp.fit_polynomial_distortions(I=3,J=2,const='fixed_slit');  
  QDisp.linearize_qaxis(ord=4);

  # ADDITIONAL TESTS
  print 'HISTORY';
  for l in QDisp.get_status().split("\n"): print "| "+l;
  QDisp.verbosity=0;

  # test normalisation to slit coordinates
  T = QDisp.u2x;
  assert np.allclose((2371,1060),T.transform(0,0));   # origin at direct beam
  assert np.allclose(1,T.inverse(3850,1060)[0]-T.inverse(900,1060)[0]); # slit lenght 1

  # test fitting of trapezoidal distortions
  uv      = np.reshape([ (u,k*u) for k in range(1,6) for u in range(100) ], (5,100,2));
  trapz,u0= QDisp.fit_trapezoidal_distortions(vp=(0,1),u=uv[...,0],v=uv[...,1]);
  assert( np.allclose( trapz.vp , (0,0), atol=1e-6 ) );

  # test coherence between trapezoidal and polynomial distortion
  trapz,u0= QDisp.fit_trapezoidal_distortions();
  poly,u0p= QDisp.fit_polynomial_distortions(const='fixed_slit');
  #print poly.coeff, [[ 0, trapz.vp[0]/trapz.vp[1]], [1, -1/trapz.vp[1]]]
  assert np.allclose(poly.coeff, [[ 0, trapz.vp[0]/trapz.vp[1]], [1, -1/trapz.vp[1]]]);
  assert np.allclose(u0,u0p);

  # test position of origin; q=0 should be at pos. of direct beam
  # TODO: polynomial distortion fails if dispersion has no zero
  for const in ('aperture_calibration','fixed_slit'):
    poly,u0 = QDisp.fit_polynomial_distortions(I=3,J=2,const=const);
    QDisp.linearize_qaxis(ord=2);
    T=QDisp.get_q2u();
    assert np.allclose((0,0),T.transform(0,0));
  plt.show();

 # uncomment to raise all figures before closing upon exception
 except Exception, e:
  print e;
  #plt.show();
  raise

