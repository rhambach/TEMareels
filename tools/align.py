"""
  align images using maximum correlation between 1D projections
     
  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np
import scipy.signal as sig;

def get_offset1D(line, ref, max_shift=0, deriv=False):
  """
    deriv ... (opt) use derivative of projection to align images
                   (useful for step-function)
    use max_shift>0 to avoid border effects (reduces size of line)
    TODO: this is not really true! Implement real max_shift, which
          is then even faster...!

    return index dx in line, which corresponds
    to the origin in the reference signal; line[dx:] ~ ref[0:]
  """
  # (opt) calculate derivative
  if deriv:
    line= np.diff(line)
    ref = np.diff(ref)

  conv = sig.convolve(line[max_shift:len(line)-max_shift],ref[::-1]);
  #plt.plot(np.arange(len(ref)), ref, 'g');
  #plt.plot(np.arange(len(line))- np.argmax(conv)+(len(ref)-1), line, 'r');
  return np.argmax(conv)-(len(ref)-1)+max_shift;  # origin: argmax=len(ref)-1

def get_offsetND(img, ref, **kwargs):
  """
    return index tuple for img, indicating the origin in 
    the reference array ref; img[dx:] ~ ref[0:]
  """
  dim  = img.ndim;
  shift= np.empty(dim, dtype=int);

  # calculate shift between ref and img for each axis separately
  for axis in range(dim):

    # project N-D array to axis
    others = range(0,axis)+range(axis+1,dim);
    line   = np.apply_over_axes(np.sum, img, axes=others).flatten();
    refline= np.apply_over_axes(np.sum, ref, axes=others).flatten();

    # 1D alignment
    shift[axis] = get_offset1D(line, refline, **kwargs);

  return shift;


def align1D(line, ref, **kwargs):
  """
    shift line such that it conincides with ref
    RETURNS 1D array of size len(ref), padded with 0's
  """
  N, M = len(line), len(ref);
  ret  = np.zeros(M);
  dx   = get_offset1D(line, ref, **kwargs);

  #  Notation for aligning arrays:
  #  [-----------]            line
  #       [-------------]     ref
  # -dx   0     N-dx    M
  #
  #       [------|000000]     ret
  #       lo     up     M
  lo = max( -dx,0);
  up = min(N-dx,M);
  ret[lo:up] = line[lo+dx:up+dx];

  return ret

def alignND(img, ref, shift=None, **kwargs):
  """
    shift N-D array along each axis such that
    it coincides with the reference image

    RETURNS N-D array with same shape as ref, padded with 0's
  """
  N, M = img.shape, ref.shape;
  dim  = img.ndim;
  ret  = np.zeros(ref.shape);

  # shift between img and ref
  if shift is None:  shift= get_offsetND(img,ref,**kwargs);

  # common limits for ref and aligned img ( see align1D )
  lo = np.maximum( -shift, 0);
  up = np.minimum(N-shift, M);

  # slicing for N-dimensions ( slice(a,b) <-> [a:b] )
  s1 = tuple([ slice(lo[i],up[i]) for i in range(dim) ]);
  s2 = tuple([ slice(lo[i]+shift[i], up[i]+shift[i]) for i in range(dim) ]);
  ret[s1] = img[s2];

  return ret


# -----------------------------------------------------------------
if __name__ == "__main__":
  import matplotlib.pylab as plt

  for M in (14,15,4,5):
    # test get_offset1D
    ref = np.zeros(M); ref[2]=1;
    for dx in range(M):
      assert( dx == get_offset1D( np.roll(ref,dx), ref) % M );
      assert( dx == get_offset1D( np.roll(ref,dx), ref, deriv=True) % M );
    
    # test align for 1D
    for N in (4,5,14,15):
      line  = np.zeros(N);  line[1]=1;
      assert(np.allclose(ref,align1D(line,ref)));
      assert(np.allclose(ref,alignND(line,ref)));
      assert(np.allclose(ref,alignND(line,ref,deriv=True)));

  # test align3D
  ref = np.zeros((14,15,16)); ref[3,2,10]=1;
  img = np.zeros(( 5, 4, 3)); img[1,2,1]=1;
  assert(np.allclose(ref,alignND(img,ref)));
  assert(np.allclose(ref,alignND(img,ref,max_shift=1)));

  # test max_shift
  #ref = np.zeros((4)); ref[1]=1;
  #img = np.zeros((10)); img[4]=1;
  #for m in range(7):
  #  print m
  #  assert np.allclose(get_offset1D(img,ref,max_shift=m),3);
  #  assert np.allclose(get_offset1D(ref,img,max_shift=m),-3);

  print "all tests passed ..."
