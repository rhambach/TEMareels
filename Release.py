"""
  Copyright (c) 2013, rhambach. 
    This file is part of the TEMareels package and released
    under the MIT-Licence. See LICENCE file for details.
"""

# describe release
name        = 'TEMareels'
version     = '1.0.0'
version_id  = 'none'
description = "Analysis scripts for momentum-resolved energy-loss spectra taken in a transmission electron microscope."
license     = 'MIT Licence'
authors     = ("Ralf Hambach", "Philipp Wachsmuth")

# describe current state of TEMareels working directory
def git_describe(GITdir=None):
  """ 
  try to determine accurate version_id by running git
  http://stackoverflow.com/questions/14989858
  """
  from os import getcwd, chdir
  from os.path import dirname, abspath
  import subprocess

  if GITdir is None: GITdir = __file__;
  module_path = dirname(abspath(GITdir));   # module directory
  working_dir = getcwd();                   # working directory
  chdir(module_path);
  version_id = subprocess.check_output(["git", "describe","--long","--dirty"]);
  version_id = version_id.rstrip();         # remove trailing newlines
  chdir(working_dir);                       # switch back to old directory
  return version_id;

if version_id=='none':
  try:
    version_id = git_describe()
  except:
    pass

