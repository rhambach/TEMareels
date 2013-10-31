TEMareels.edisp
===============

Python module for energy calibration.

- [fit_peak_pos.py](./fit_peak_pos.py):
      fit one or two peaks in shifted energy-loss spectra

- [energy_dispersion.py](./energy_dispersion.py): 
      calculate the energy dispersion using the distance between 
      two sharp peaks in the shifted energy-loss spectra


- [estimate_beam_energy.py](./estimate_beam_energy.py):
      script to estimate the actual beam energy in the TEM
- [energy_dispersion_single_peak.py](./energy_dispersion_single_peak.py):
      calculate the energy dispersion using a single peak in the
      shifted energy-loss spectra and the beam-energy correction data

- [find_zlp.py](./find_zlp.py):
      fit the ZLP peak positions in a WQmap
      
### HowTo use these modules

The modules should be used via the scripts in the [TEMareels/runscripts](../runscripts)
directory or by importing them into your own scripts. Additionally, they
can be executed using, e.g.,

```python energy_dispersion.py```

to execute self-test. For a documentation, use pydoc

```pydoc ./energy_dispersion.py```
