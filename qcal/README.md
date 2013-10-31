TEMareels.qdisp
===============

Python module for momentum calibration.

- [fit_border.py](./fit_border.py):
      fit left and right border of aperture in wq-maps for 
      momentum calibration

- [momentum_dispersion.py](./momentum_dispersion.py): 
      analyse the q/x dispersion from the borders of a small, 
      round aperture in momentum/real space.
      
### HowTo use these modules

The modules should be used via the scripts in the 
[TEMareels/runscripts](../runscripts) directory or by importing 
them into your own scripts. Additionally, they can be executed 
using, e.g.,

```python momentum_dispersion.py```

to execute self-test. For a documentation, use pydoc

```pydoc ./momentum_dispersion.py```
