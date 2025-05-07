## Reg

Module for analysis the data after brain registration

- `main_roi_quant.py`: After 2d registration by [allenCCF](https://github.com/cortex-lab/allenCCF), and roi selection by
  Fiji. using the output .csv for roi identify (based on `roi_classifier_pl.RoiClassifierPL`)
- `main_roi_map.py`: annotate roi after transformation of the raw images (dev)
- `run_3d_proj.py`: use the allenCCF and reconstruct the 3d brain
- `np_localize.py`: reconstruct the 4 shanks neuropixel based on histology (interp.)
- `main_expression_level.py`: check the expression of rois in 1d or 2d space
- `main_cp_man_cmp.py`: compare roi selection using manually approach or cellpose automated pipeline (seealso in `cpose`
  module)

## reference source

- [allen API](http://help.brain-map.org/display/mouseconnectivity/API)
- [Cell atlas](https://bbp.epfl.ch/nexus/cell-atlas/)

## Data structure for the registration data and ROI labeled

    YW001_reg/
        ├── raw/ (optional) -- (1)
        │
        ├── zproj/ -- (2)
        │    └── YW001_g*_s*_{channel}.tif
        │
        ├── roi/ -- (3)
        │    └── YW001_g*_s*_{channel}.roi  
        │
        ├── roi_cpose/ -- (3')
        │    └── YW001_g*_s*_{channel}.roi          
        │
        ├── resize/ (src for the allenccf) 
        │    ├── YW001_g*_s*_resize.tif -- (4)
        │    │ 
        │    └── processed/
        │           ├── YW001_g*_s*_resize_processed.tif -- (5)
        │           └── transformations/
        │                 ├── YW001_g*_s*_resize_processed_transformed.tif -- (6)
        │                 │
        │                 ├── YW001_g*_s*_resize_processed_transform_data.mat -- (7)
        │                 │
        │                 │ 
        │                 └── labelled_regions/
        │                       ├── {*channel}_roitable.csv -- (8) 
        │                       └── parsed_data / 
        │                             └── parsed_csv_merge.csv -- (9)
        │
        ├── resize_overlap/* (optional) -- (10)
        └── output_files/ (for generate output fig)

* (1). raw data for confocal (i.e., .lsm, .czi or .tiff)
* (2). z projection stacks, should be `RGB` format and save per channel (r, g, b and o)
* (3). roi file after imageJ selection
* (3'). roi file using cellpose pipeline (dev)
* (4). merged ROI, scaled RGB tif file for registration. normally use **blue (DAPI)** channel as a reference.
  for example, **green roi + red roi + DAPI channel**. Since limited channel numbers, overlap channel need to save to
  another file using pseudo-color, then used the same transformation matrix to do the registration procedure
* (5) contrast or rotated image after process
* (6) image after transformation
* (7) transformation matrix for each slices
* (8) allenccf output (per ROI)
* (9) csv after parsed and classification (used for data visualization)
* (10) if there is overlap channel, create the same folder structure as r/g channels. ** Note that use the same
  transformation matrix
