TEMareels.runscripts
====================

Runscripts for performing energy- and momentum calibration of WQmaps. 

The following list gives a short overview on the scripts and their tasks.
More detailed information can be found in each of the runscripts and
the related module. 

### Energy Calibration

1. **Ecal_image_preprocess.py**: 
      extract spectra from series of images with different energy offset
2. **Ecal_run_calibration.py**:
      perform energy calibrations and write results to *EDisp.pkl*

### Momentum Calibration

1. **Qcal_image_preprocess.py**:
      rebin raw images of q-calibration series along energy axis
2. **Qcal_run_calibration.py**:
      perform momentum calibration and write results to *QDisp.pkl*

### Apply Calibration to measured WQmap

0. **(EQmap_manual_calibration.py)**: 
      manual energy- and momentum calibration instead of the above
1. **EQmap_image_preprocess.py**:
      align and filter raw WQmaps
2. **EQmap_remove_qdistortion.py**:
      apply q-calibration to WQmap (reads *QDisp.pkl*), writes 
      undistorted WQmap as 32bit-Tif and pkl-file for next step
3. **EQmap_extract_spectrum.py**:
      extract spectra from calibrated WQmap and apply E-calibration
      and an angular correction to each spectrum (reads *EDisp.pkl*),
      writes one MSA-file for each q-value
      
### HowTo use these scripts

If you want to use these scripts with your own data, it is recommended
to make a copy of the entire TEMareels package for later reference 
(backward compatibility will not be not guaranteed) and to set this
install path in the file ```_set_pkgdir.py```. This allows you to copy the 
entire runscript directory and adjust the parameters in the runscripts
in the specified order.
