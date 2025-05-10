Data Structure for Anatomy
=================================================

Directory Overview
------------------

.. code-block:: text

    hist/
       ├── ANIMAL_1/
       │     ├── raw/ (optional)             # (1)
       │     ├── zproj/                      # (2)
       │     ├── roi/                        # (3)
       │     ├── roi_cpose/                  # (3')
       │     ├── resize/                     # (4)
       │     │    ├── processed/
       │     │    │    └── transformations/
       │     │    │          └── labelled_regions/
       │     │    │                └── parsed_data/
       │     ├── resize_overlap/ (optional)  # (10)
       │     └── output_files/               # (for figure generation)
       │
       ├── OTHER_ANIMAL_2/
       ├── OTHER_ANIMAL_3/
       ├── cache/                             # (11)
       ├── population_analysis_figure1.pdf    # (12)
       ├── population_analysis_figure2.png    # (12)
       ├── ...                                # (12)
       └── (more population analysis result files)

Folder Details
--------------

- (1) ``raw/``
  Raw confocal data (e.g., `.lsm`, `.czi`, or `.tiff` formats). (Optional)

- (2) ``zproj/``
  Z-projection stacks in **RGB** format, saved separately per channel (`r`, `g`, `b`, `o`).

- (3) ``roi/``
  ROI files generated from manual selections using **ImageJ**.

- (3') ``roi_cpose/``
  ROI files generated using the **Cellpose** segmentation pipeline (developmental version).

- (4) ``resize/``
  Merged ROI stacks, rescaled into an RGB `.tif` file for registration.
  Typically use the **blue (DAPI)** channel as the registration reference.
  For example, merging **green ROI + red ROI + DAPI**.
  If overlap channels exist, pseudo-coloring is used, and the same transformation matrix must be applied.

- (5) ``processed/``
  Processed images after contrast adjustment or rotation.

- (6) ``transformations/`` (Transformed Images)
  Registered images after applying transformations.

- (7) ``transformations/`` (Transformation Matrices)
  `.mat` files containing transformation matrices for each slice.

- (8) ``labelled_regions/``
  Output per ROI generated using the **Allen CCF** registration pipeline.

- (9) ``labelled_regions/parsed_data/``
  Parsed and classified ROI tables (merged into one CSV file) for downstream analysis.

- (10) ``resize_overlap/``
  Folder for overlap channels (pseudo-color).
  Must use the **same transformation matrix** as the main channels.

- (11) ``cache/``
  Stores cached files for population data analysis.

- (12) Population analysis figures (files under ``hist/``)
  Files containing population analysis results (e.g., ``.pdf``, ``.png``) are saved directly under ``hist/``, **not inside a separate folder**.

Notes
-----

- Overlap channels should be processed carefully to ensure consistent registration and transformation.
- Population-level analysis figures are placed directly under ``hist/`` without additional folder structure.

