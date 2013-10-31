"""
  Wrapper for PIL for reading Tietz Tiff Files.

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import Image;
import numpy as np;

def load_TVIPS(filename, verbosity=1):
  """
    RETURNS 2D-array containing the image data
  """
  if verbosity>0: print "load file '%s'"%filename;
  img = Image.open(filename);
  A   = np.array(img.getdata()).reshape(img.size[::-1]);
  # set border to 0 (dead pixel)
  A[0:5,:] = A[:,0:5] = A[-5:,:] = A[:,-5:] = 0;
  return A;

def load_tiff(filename, verbosity=1):
  """
    RETURNS 2D-array containing the image data
  """ 
  if verbosity>0: print "load file '%s'"%filename;
  img = Image.open(filename);
  A   = np.array(img.getdata(),dtype=float).reshape(img.size[::-1]); 
  return A; 

def write_tiff(A, filename):
  """
    write Tiff file
  """
  Image.fromarray(A).save(filename);  

# -- main ----------------------------------------
if __name__ == '__main__':
  
  A=load_tiff('../tests/qseries1.tif');
  write_tiff(A,'tmp.tif');
  B=load_tiff('tmp.tif');
  assert np.allclose( A, B );

  print 'all tests passed'
