"""
  Unique place for all runscripts to set the module path.
  (i.e. if you have a copy of TEMareels in some place on your
   hard disk but not in PYTHONPATH). 

  NOTE: it is recommended to keep a copy of all TEMareels
    module files with your data/analysis, as future versions 
    will be not necessarily backwards compatible.

  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""

# location of the TEMareels package on the hard disk
# (if not specified in PYTHONPATH)
pkgdir = '../..'; 
import sys
from   os.path import abspath;
sys.path.insert(0,abspath(pkgdir));
