"""
   Rectangular off-axis aperture for a w-q mapping in a TEM.
   The slit is defined by the width dqy in the energy-dispersive
   direction and the bin width dqx for each spectrum; q0 denotes
   the center of the aperture.
   
   USAGE

     The aperture correction function for w-q mapping can be 
     calculated as follows (see also TEST2 at end of file)

     def APC(q0,dqx,dqy,E,E0):
       AP = TEM_wqslit(q0,dqx,dqy,E0);
       return AP.moment(0,E);

"""
import numpy as np;
from   aperture import Aperture;
import conversion as conv;


class TEM_wqslit(Aperture):
  """
    Rectangular aperture centered at q0 in the TEM
  """

  def __init__(self,q0,dqx,dqy,E0,verbosity=0):
    """
      q0 ... center of aperture [a.u.]
     dqx ... width of aperture along x-direction [a.u.]
     dqy ... height of aperture along y-direction (dispersive direction) [a.u.]
      E0 ... beam energy [keV]
    """
    self.q0 = q0;
    self.dqx= dqx;
    self.dqy= dqy;
    self.verbosity=verbosity;
    super(TEM_wqslit,self).__init__(E0);

  def __arc_segment(self,r,a,b):
    """
      Returns length of circle segment in a half-infinite slit.
        r   ... radius of circle which is centered at the origin
        a,b ... borders of slit containing (x,y) with x>a, b>y>0 
    """  
    #print r, a, b
    # scalar version
    if np.isscalar(r):
      if r<a:              # circle outside slit
        return 0;   
      elif r**2>a**2+b**2: # circle does not intersect left border at x=a
        return r*np.arcsin(b/r);
      else:                # circle only intersects left border at x=a
        return r*np.arccos(a/r);
  
    # parallel version
    else:
      r     = np.atleast_1d(r); 
      arc   = np.zeros(r.size);
      index = r>=a;              # select radii which intersect half-slit
      y2_a  = r[index]**2-a**2;  # intersection of circle with left border at x=a
      ymax  = np.where( y2_a > b**2, b, np.sqrt(y2_a) ); 
                                 # largest y value on circle segment in half-slit
      arc[index] = r[index] * np.arcsin( ymax / r[index] );
                                 # calculate arc length
      # DEBUG
      if self.verbosity>2:
        arc2 = [ self.__arc_segment(ir,a,b) for ir in r ];
        assert(np.allclose( arc, arc2 ));
      return arc;
  
    raise RuntimeError("Should never reach this point");
  
  def __arc_in_box(self, r, (x1,y1), (x2,y2)):
    """ 
      Returns the lenght of a circle segment in rectangular box.
        r       ... radius of circle which is centered at the origin
        (x1,y1) ... lower-left corner of box
        (x2,y2) ... upper-right corner of box
    """
    #print (x1,y1), (x2,y2);
  
    # use mirror symmetry about x,y-axis
    if x1<0 and x2<0: (x1,x2) = (-x1,-x2);
    if y1<0 and y2<0: (y1,y2) = (-y1,-y2);
  
    # ensure standard order
    if x1>x2: (x1,x2) = (x2,x1);   
    if y1>y2: (y1,y2) = (y2,y1);
  
    # if x1<0<x2: split box at x=0, summing left + right box
    if x1*x2<0:  
      return self.__arc_in_box(r, (0,y1),(-x1,y2)) + self.__arc_in_box(r, (0,y1), (x2,y2));
  
    # if y1<0<y2: split box at y=0, summing lower + upper box
    # if 0<y1<y2: represent box as difference of two boxes with y1=0
    if y1<>0:
      return self.__arc_in_box(r, (x1,0), (x2,y2)) - np.sign(y1) * \
             self.__arc_in_box(r, (x1,0), (x2, np.abs(y1)));
  
    assert(x2>x1>=0 and y2>0 and y1==0);
    return self.__arc_segment(r,x1,y2) - self.__arc_segment(r,x2,y2);
  
  def p(self,q):
    """
    Momentum probability distribution p(|q|) for electrons entering the 
    rectangular aperture of width dqx * dqy at momentum q0 [a.u.]
               / q0+dqx/2  / dqy/2
     p(|q|) =  |           |        delta(|q|-sqrt(qx^2+qy^2) dqx dqy
               / q0-dqx/2  / -dqy/2
    """
    return 2 * self.__arc_in_box(q,(self.q0-self.dqx/2., 0), \
                                   (self.q0+self.dqx/2., self.dqy/2.));

  def __compile_weave_func(self):
    """ 
    Fast version of the integrand function f() in moment().
    We write all functions called in f() in C-code and 
    compile it once during runtime. See scipy.weave for info.

    NOTE: remove TEM_wqslit_weave.so if C-code is changed
    """
    from scipy.weave import ext_tools;
    
    mod = ext_tools.ext_module('TEM_wqslit_weave'); # create module TEM_wqslit_weave
    r=a=b=x1=x2=y1=y2=q=qE=q0=dqx=dqy=n=1.;         # declaration of variable type

    # translated C-code for arc_segment() and arc_in_box()
    code="""
        double arc_segment(double r, double a, double b) {
          if      (r<a)          return 0; 
          else if (r*r>a*a+b*b)  return r*asin(b/r); 
               else              return r*acos(a/r); 
        };
  
        double arc_in_box(double r, double x1, double y1, double x2, double y2) {
          double x, y;
          if (x1<0 && x2<0) { x1=-x1; x2=-x2;};
          if (y1<0 && y2<0) { y1=-y1; y2=-y2;};
          if (x1>x2)        {x=x1; x1=x2; x2=x;};
          if (y1>y2)        {y=y1; y1=y2; y2=y;};
        
          if (x1*x2<0) return arc_in_box(r, 0, y1, -x1, y2) + arc_in_box( r, 0, y1, x2, y2); 
          if (y1<0)    return arc_in_box(r, x1, 0, x2, y2)  - arc_in_box( r, x1, 0, x2,-y1);
          if (y1>0)    return arc_in_box(r, x1, 0, x2, y2)  + arc_in_box( r, x1, 0, x2, y1);
    
          return arc_segment(r,x1,y2) - arc_segment(r,x2,y2);
        };
        """
    # translated C-code for integrand in moment(), skips p(), weight_q()
    main = "return_val = pow(q,n) / (q*q + qE*qE) * 2 \
                        * arc_in_box(q, q0-dqx/2., 0, q0+dqx/2., dqy/2.);"

    # compile module
    func = ext_tools.ext_function('f',main,['q','qE','n','q0','dqx','dqy']);
    func.customize.add_support_code(code);
    mod.add_function(func);
    mod.compile();



  def moment(self,n,E):
    """
    Calculate moment  <q^n> = Int d|q| w(|q|,qE) |q|^n [a.u.] 
    
    FASTER IMPLEMENTATION compared to function of base class
      using inline C-code for the calculation of the Integrand

    ToDo: speed-up is complicated
      1. using vectorized integration (like romberg) runs into
         accuracy problems and is only slightly faster
      2. lowering rtol might be sufficient
      3. trapz is very fast, but might be inaccurate
      4. is it possible to reuse some of the calculated
         quantities in a second call (parallel over E ?)
    """ 
    from   scipy.integrate import quad;
    try:                                # try loading module with precompiled integrand f()
        import TEM_wqslit_weave
    except ImportError:                 # module not yet compiled
        self.__compile_weave_func()
        import TEM_wqslit_weave

    qE = conv.Qmin(E,self.E0) * conv.bohr;        # [a.u.]
    F, err = quad(TEM_wqslit_weave.f,self.qmin(),self.qmax(),
                args=(qE,float(n),self.q0,self.dqx,self.dqy)); # C-function expects float arguments
    assert(abs(err)<1e-5);
    return F;
  
  
  def qmin(self):
    " minimal |q| with p(|q|) > 0 [a.u]" 
    return max(np.abs(self.q0)-self.dqx/2., 1e-5);  # avoid q=0

  def qmax(self):
    " maximal |q| with p(|q|) > 0 [a.u.]"
    return np.sqrt((np.abs(self.q0)+self.dqx/2.)**2 + (self.dqy/2.)**2);



# -- main ----------------------------------------
if __name__ == '__main__':
  import matplotlib.pylab as plt;
  from   scipy.integrate import quad;
  import conversion as conv;

  # TEM aperture
  q0=0.2*conv.bohr; dqx=0.1*conv.bohr; dqy=0.25*conv.bohr; # [a.u.]
  E=20; E0=40;                                            # eV / keV
  TEM = TEM_wqslit(q0,dqx,dqy,E0,verbosity=3);

  print " TEST1: circle segment in rectangular box "  # test for symmetry
  arc_in_box = TEM._TEM_wqslit__arc_in_box;    # access private function
  assert( np.allclose( arc_in_box(1,(-1,-1),(1,1)), 2*np.pi ) );
  assert( np.allclose( arc_in_box(1,(0.5,0.2),(0.7,1)), \
                       arc_in_box(1,(0.2,0.5),(1,0.7)) ));
  assert( np.allclose( arc_in_box(1,(-0.5,0.2),(0.7,-1)), \
                       arc_in_box(1,(0.2,-0.5),(-1,0.7)) ));

  print " TEST2: aperture correction (calculated in cartesian and polar coord.)"
  def aperture_correction(q0,dqx,dqy,qE):
    def f(qy):
      " Returns Int_{-dqx/2}^{dqx/2}  dqx / (qx^2 + qy^2 + qz^2) "
      q = np.sqrt(qy**2+qE**2);
      return 2 * np.arctan( dqy / 2. / q ) / q;
    F, err = quad(f,q0-dqx/2,q0+dqx/2);
    return F;

  for q0 in np.arange(-6,6)*0.3333*conv.bohr:   # [a.u.]
    Aperture = TEM_wqslit(q0,dqx,dqy,E0);
    assert( np.allclose( Aperture.moment(0,E), \
                 aperture_correction(q0,dqx,dqy,conv.Qmin(E,E0)*conv.bohr)) );

  fig=plt.figure();
  En=np.arange(0,40,0.5);
  for q0 in np.arange(0,0.3,0.05)*conv.bohr:
    Aperture = TEM_wqslit(q0,dqx,dqy,E0);
    APC = [Aperture.get_APC(_E) for _E in En];
    plt.plot(En,APC,label="q=%4.2f 1/A"%(q0/conv.bohr));
  plt.legend();
  plt.title("Angular correction for a rectangular aperture at different positions q");
  plt.xlabel("Energy [eV]");
  plt.ylabel("Angular Correction Factor");

  print " TEST3: internal consistency between fast calculation of moments and normal one"
  def moment(self,n,E): # copy of original function to calculate moment
    def f(q): return q**n * self.weight_q(q,E);
    F, err = quad(f,self.qmin(),self.qmax());
    return F;
  for q0 in np.arange(-6,6)*0.3333*conv.bohr: 
    for n in range(3):
      assert( np.allclose( Aperture.moment(0,E), moment(Aperture,0,E) ) );


  print " TEST4: statistics for contributing momentum transfers"
  fig=plt.figure(); ax=fig.add_subplot((111));
  plt.title("Contributions of different momentum transfers to EELS signal");
  q = np.arange(0.001,0.5,0.001);               # [1/A]
  w = TEM.weight_q(q*conv.bohr,E);
  ax.plot( q, w );
  plt.xlabel("Momentum transfer q [1/A]");
  plt.ylabel("Weight in EELS signal");
  plt.axvline(TEM.qmin()/conv.bohr,color='k');
  plt.axvline(TEM.qmax()/conv.bohr,color='k');
  q_mean = TEM.mean(E)/conv.bohr; q_dev= np.sqrt(TEM.var(E))/conv.bohr;
  plt.axvline(q_mean      ,color='b');
  plt.axvline(q_mean-q_dev,color='r');
  plt.axvline(q_mean+q_dev,color='r');


  print " TEST5: average momentum transfer"
  fig=plt.figure(); ax=fig.add_subplot((111));
  plt.title("Average momentum transfer for dqx=%5.3f 1/A, dqy=%5.3f 1/A, E=%d eV, E0=%d keV"%(dqx/conv.bohr,dqy/conv.bohr,E,E0));
  q0= np.arange(0.01,1,0.1); q = []; dq = []; # [1/A]
  for _q0 in q0: 
    Aperture = TEM_wqslit(_q0*conv.bohr,dqx,dqy,E0);
    q.append( Aperture.mean(E) / conv.bohr );
    dq.append( np.sqrt(Aperture.var(E)) / conv.bohr );
  q = np.asarray(q);
  ax.plot( q0, q,    'b', linewidth=2 );
  ax.plot( q0, q-dq, 'r');
  ax.plot( q0, q+dq, 'r');
  ax.plot( q0, q0,   'k--');
  plt.xlabel("Center of aperture q0 [1/A]");
  plt.ylabel("Average momentum transfer q [1/A]");
  

  plt.show();
