"""
  Simple GUI for visualising and analysing wq-maps

  USAGE
    An example can be found at the end of this file and can be executed
    using 'python wq_stack.py'

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
     
"""

import numpy as np
import matplotlib.pylab as plt
from matplotlib.widgets import Button, RadioButtons, Slider


class WQBrowser(object):
  """
  WQBrowser shows a grayscale image in a separate window
  """

  def __init__(self,image,imginfo={},verbosity=0,**kwargs):
    """
    image  ... 2D Array 
    imginfo... (opt) dictionary with parameters of the image:
               'desc'     ... image description (default: '')
               'filename' ... filename of the image (default: '')
               'xlabel'   ... name for x-axis    (default: x)
               'ylabel'   ... name for y-axis    (default: y)
               'xperchan' ... scaling for x-axis (default: 1)
               'yperchan' ... scaling for y-axis (default: 1)
               'xunits'   ... unit for x-axis (default: 'px')
               'yunits'   ... unit for y-axis (default: 'px')
               'xoffset'  ... first x-value   (default: 0)
               'yoffset'  ... first y-value   (default: 0)
    verbosity. (opt) quiet (0), verbose (3), debug (4)
    futher options are passed to the imshow method of matplotlib
    """

    self.image  = np.asarray(image);
    self.kwargs = kwargs;
    self.verbosity=verbosity;

    # set default values for imginfo and extent
    self._set_imginfo(imginfo);
         
    # open new figure and draw image
    self.fig  = plt.figure();
    self.axis = self.fig.add_subplot(111); 
    self.fig.subplots_adjust(right=0.85);
    self.AxesImage=None;
    self.axis.set_title(self.imginfo['desc']);
      
    # add Switches and Buttons
    axStyle = self.fig.add_axes([0.85, 0.1, 0.12, 0.12]);
    self.rbStyle = RadioButtons(axStyle,["linear","log","contour"]);
    self.rbStyle.on_clicked(self.set_style);
    self.currentstyle = "linear";
    axLine = self.fig.add_axes([0.85, 0.3, 0.12, 0.05]);    
    self.bLine = Button(axLine,'Line Profile');
    self.bLine.on_clicked(self._toggle_line_profile);
    self.LineProfile=None;
    self.bLine.color_activated='red';

    # key pressed
    self.fig.canvas.mpl_connect('key_press_event', self._key_press_callback)

    # finally draw image 
    self._reset_image();     


  def set_cmap(self,cmap):
    "Change the colormap"
    self.cmap=cmap;

  def set_style(self,label):
    """ 
    Change style of background image 
      label ... "linear", "log", or "contour"
    """
    from matplotlib.colors import Normalize, LogNorm, BoundaryNorm, Colormap
    from copy import deepcopy
    self.currentstyle=label;

    if label=="linear":
      self.AxesImage.set_cmap(self.cmap);
      self.AxesImage.set_interpolation("bilinear");
      self.AxesImage.set_norm(Normalize(self.vmin,self.vmax,clip=False));
    
    elif label=="log":
      self.AxesImage.set_cmap(self.cmap);
      self.AxesImage.set_interpolation("bilinear");
      posmin = np.min(self.image[self.image>0]); # smallest positive pixel value
      self.AxesImage.set_norm(LogNorm(max(self.vmin,posmin),self.vmax,clip=False));
    
    elif label=="contour":
      cmap = deepcopy(self.cmap);
      cmap.set_over('y'); cmap.set_under('b'); cmap.set_bad('r');
      self.AxesImage.set_cmap(cmap);
      self.AxesImage.set_interpolation("nearest");
      self.AxesImage.set_norm(BoundaryNorm(np.linspace(self.vmin,self.vmax,10),cmap.N,clip=False));
    self._update();


  def _key_press_callback(self, event):
    "whenever a key is pressed"
    if not event.inaxes: return

    if event.key=='+':
      self.vmin+=(self.vmean-self.vmin)/2.;
      self.vmax-=(self.vmax -self.vmean)/2.;
      self.AxesImage.set_clim(self.vmin,self.vmax);
      self._update();

    if event.key=='-':
      self.vmin-=(self.vmean-self.vmin);
      self.vmax+=(self.vmax -self.vmean);
      self.AxesImage.set_clim(self.vmin,self.vmax);
      self._update();

    if event.key.upper()=='R':
      self._reset_image();

  def _set_imginfo(self,imginfo):
    " set default values for imginfo and extent"

    # set default values for imginfo
    self.imginfo= {'desc': "", 'filename':"", 'xlabel': 'x', 'ylabel': 'y',
                   'xperchan':1., 'yperchan':1., 'xoffset':0., 'yoffset':0.,
                   'xunits':'px', 'yunits':'px'};
    self.imginfo.update(imginfo);

    # set extent
    Ny,Nx= self.image.shape;
    Imin = self._px2ic(-0.5,-0.5);
    Imax = self._px2ic(Nx-0.5, Ny-0.5); 
    self.imginfo['extent'] = [Imin[0],Imax[0], Imax[1],Imin[1]];

  def _toggle_line_profile(self,event):
    if self.LineProfile is not None:
      self.LineProfile = None;              # close window
    else:
      self.LineProfile = LineProfile(self); # open window
    # swap colors
    b=self.bLine; 
    b.color,b.color_activated=b.color_activated,b.color   
    b.hovercolor=b.color;

  def _reset_image(self):
    " redraw image "
    self.vmin = np.min(self.image);
    self.vmax = np.max(self.image);
    self.vmean= np.mean(self.image);
    self.cmap = plt.cm.gray;      

    if self.AxesImage is not None: self.AxesImage.remove();
    self.AxesImage = self.axis.imshow(self.image,cmap=plt.gray(),\
        extent=self.imginfo['extent'],**self.kwargs);
    self.axis.set_xlabel("%s [%s]" % (self.imginfo['xlabel'], self.imginfo['xunits']));
    self.axis.set_ylabel("%s [%s]" % (self.imginfo['ylabel'], self.imginfo['yunits']));
    self.axis.set_xlim(*self.imginfo['extent'][0:2]);
    self.axis.set_ylim(*self.imginfo['extent'][2:4]);
    self.set_style(self.currentstyle);

  def _update(self):
    self.fig.canvas.draw();

  def _ic2px(self,x,y):
    "convert image coordinates to pixel positions"
    return ((x-self.imginfo['xoffset'])/self.imginfo['xperchan'], 
            (y-self.imginfo['yoffset'])/self.imginfo['yperchan']);

  def _px2ic(self,x,y):
    "convert pixel positions to image coordinates"
    return (x*self.imginfo['xperchan']+self.imginfo['xoffset'], 
            y*self.imginfo['yperchan']+self.imginfo['yoffset']);


class WQStackBrowser(WQBrowser):
  """
  WQStackBrowser shows a series of grayscale images in a separate window
  """

  def __init__(self,imgstack, info=None, linestack=None, **kwargs):
    """
    imgstack ... list of 2D Arrays
    info     ... (opt) list of imageinfo dictionaries (see WQBrowser)
    linestack... (opt) list of lines from plt.plot() to be drawn with each image
    futher options are passed to the base class WQBrowser
    """
    self.N         = len(imgstack);  # number of images
    self.slice     = None;           # current slice (is set to 0 later on)
    self.imgstack  = [np.asarray(image) for image in imgstack];
    self.linestack = linestack if linestack is not None else [[]]*self.N;
    self.stackinfo = info if info is not None else [{}]*self.N;

    # check input parameters
    Ny,Nx = self.imgstack[0].shape;
    for image in self.imgstack: 
      assert image.shape==(Ny,Nx);    # all images should have the same shape
    assert len(self.linestack) == self.N
    assert len(self.stackinfo) == self.N

    # init ImageBrowser
    super(WQStackBrowser,self).__init__(self.imgstack[0],self.stackinfo[0],**kwargs);
    self.fig.subplots_adjust(right=0.85, bottom=0.2);
    self.set_slice(0);

    # add slider
    axcolor='lightgoldenrodyellow';
    self.axSlice = self.fig.add_axes([0.22, 0.08, 0.53, 0.03], axisbg=axcolor);
    self.slSlice = Slider(self.axSlice, "Slice", 0, self.N, valinit=0, closedmax=False, valfmt='%d');
    self.slSlice.on_changed(self.set_slice);


  def set_slice(self,val):
    " set slice to be shown "
    if    self.slice==int(val): return # still the same slice
    self.slice= int(val);

    # update image data
    self.image = self.imgstack[self.slice];
    self.AxesImage.set_data(self.image);

    # update line data
    while self.axis.lines: self.axis.lines.pop();  # remove all lines
    for line2D in np.atleast_1d(self.linestack[self.slice]):
      self.axis.add_line(line2D);                  # add lines for slice

    # set default values for imginfo
    self._set_imginfo(self.stackinfo[self.slice]);
    self.axis.set_title(self.imginfo['desc']);

    # update LinePlot, if present
    if self.LineProfile is not None: self.LineProfile.reset();    

    self._update();



class LineProfile():
  """
  LineProfile shows a gray-scale image and interactive line-scans
  """
  def __init__(self,WQBrowser):
    """
    further options are passed to the base class WQBrowser()
    """
    self.WQB = WQBrowser;
    self.y   = 0;

    # open new figure with line plot
    self.fig  = plt.figure(); 
    self.axis = self.fig.add_subplot(111, sharex=self.WQB.axis); 
    self.fig.subplots_adjust(right=0.85);
    self.axis.autoscale(tight=True);#set_xlim(x[0],x[-1]);
    self.axis.set_xlabel("%s [%s]" % (self.WQB.imginfo['xlabel'], self.WQB.imginfo['xunits']));
    self.axis.set_ylabel("counts");
    self.reset();

    # event handling
    self.WQB.fig.canvas.mpl_connect('button_press_event', self.__onclick);
    # self.fig.canvas.mpl_connect('close_event', self.__onclose); # does not work
    self.fig.show();

  def __del__(self):
    plt.close(self.fig);
    self.hline.remove();
    self.WQB._update()

  def reset(self):
    while self.axis.lines: self.axis.lines.pop();  # remove all lines

    # draw spectrum 
    x,y = self._get_line_profile();
    self.spectrum = self.axis.plot(x,y,'k-')[0];

    # draw markers for lines in WQB
    self.linemarkers = [];
    for Line2D in self.WQB.axis.lines:
      x=self._get_intersection(Line2D);
      l=self.axis.axvline(x,color=Line2D.get_color(),ls='-');
      self.linemarkers.append((Line2D,l));

    # update WQB
    self.axis.set_title('Line profile: '+self.WQB.imginfo['desc']);
    self.hline = self.WQB.axis.axhline(self.y,color='y',ls='--');
    self._update();

  def _get_intersection(self,Line2D):
    " return intersection point with lines in WQBrowser "
    dy     = np.abs(Line2D.get_ydata()-self.y);
    imin   = np.argmin(dy);               # closest point to line
    _,dymin= self.WQB._ic2px(0,dy[imin]); # distance in pixels
    if dymin>=1: return None;             # pixel precision
    else:        return Line2D.get_xdata()[imin];

  def _get_line_profile(self):
    " return current line-plot "
    Ny,Nx= self.WQB.image.shape;
    x,_  = self.WQB._px2ic(np.arange(Nx),0);
    _,iy = self.WQB._ic2px(0,self.y);
    y    = self.WQB.image[iy];
    self.axis.set_ylim(np.min(y), np.max(y));
    return x,y

  def __onclick(self,event):
    if event.inaxes!=self.WQB.axis: return   # wait for clicks in WQBrowser
    if plt.get_current_fig_manager().toolbar.mode!='': return # toolbar is active
    self.y = event.ydata;

    # update spectrum
    x,y = self._get_line_profile();
    self.spectrum.set_ydata(y);

    # update line views
    for Line2D,l in self.linemarkers:
      x=self._get_intersection(Line2D);
      l.set_xdata([x]*2);
   
    # update WQBrowser window
    self.hline.set_ydata([event.ydata,]*2);


    self._update();

  def _update(self):
    """
    draw line plot and line in connected Browser
    """
    self.WQB._update();
    self.fig.canvas.draw();


# -- main ----------------------------------------
if __name__ == '__main__':
  import calibration.tools.tifffile as tiff
  
  coeff = [
[[  8.53185497e-06, 5.74373869e-02, 6.63128635e+02],
[   7.69501010e-06, 4.65273936e-02, 7.89253445e+02]],
[[  6.45514206e-06, 4.10327089e-02, 9.00378881e+02],
[   6.12716673e-06, 2.82385482e-02, 1.07283189e+03]]];


  stack = []; info = []; lines = [];
  for i in (1,2):
    # read image from file
    filename = '../test/qseries%d.tif'%i;
    stack.append(tiff.imread(filename));

    # info for image
    info.append(  {'desc': 'WQStackBrowser: '+filename,
                   'filename':filename, 
                   'xperchan':4., 'yperchan':64.} );

    # additional line plots
    y  = np.arange(150,4096);
    l1 = plt.Line2D(np.poly1d(coeff[i-1][0])(y),y,color='r',ls='-');
    l2 = plt.Line2D(np.poly1d(coeff[i-1][1])(y),y,color='g',ls='-');
    lines.append([l1,l2]);

  # show single image
  IB =WQBrowser(stack[0],{'desc': 'WQBrowser'},aspect='auto',verbosity=4);

  # show image stack
  IB =WQStackBrowser(stack,info,lines,verbosity=4);

  plt.show();

