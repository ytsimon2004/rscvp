# rscvp2

by Yu-Ting Wei (ytsimon2004@gmail.com)

- Pipeline for RSC visuo-spatial project analysis
- Data are acquired using customized package [Stimpy](https://bitbucket.org/activision/stimpy/src/master)
- Core analysis largely depending on the customized
  package [neuralib](https://neuralib.readthedocs.io/en/latest/index.html)

------------------------------

# How to set up in a local machine?

## 1. Clone or download the project locally

## 2. Create conda environment

```bash
conda create -n rscvp python~=3.10.0 -y
conda activate rscvp
```

## 3. Install required packages

- First `cd` to the directory with [pyproject.toml](pyproject.toml)

- light package install

```bash
pip install -e .[light]
```

- all package dependencies (depending on the analysis purpose, see also [requirements-opt.txt](requirements-opt.txt))

```bash
pip install -e .[all]
```

------------------------------

## Project structure

- `bash`: Shell scripts for command line interface(CLI)-based automatic pipeline
- `config`: Configuration related to DAQ in experimental rig, or data preprocessing
- `notebook`: Notebook example for visualization
- `res`: Resource/references files related to the project
- `src`: Source code for python scripts
- `test`: Unittest/Function test pipeline

------------------------------

## Module scripts

- **[publish]** is the analysis used for Wei. et al., xxx
- **[optional]** is the optional analysis related to the project
- See also the CLI options by

```bash
python -m [MODULE_PATH] -h
```

### [atlas](src/rscvp/atlas)

**[publish]**

- `main_roi_atlas`: Plot the local maxima roi selection and atlas annotation based on transformed matrix & images
- `main_roi_top_view`: Plot all the ROIs (given animals, fluorescence channel, allen family) on the top view
- `main_roi_query`: Plot the fraction of subregion / hierarchical classification information in a queried area. Check
  `@DispatchOption.dispatch('stacked')` for the visualization
- `main_roi_quant_batch`: ROIs distribution analysis for batch animals. Check `@DispatchOption.dispatch()` decorated on
  methods for different visualization

**[optional]**

- `main_roi_query_batch`: Dotplot for subareas from a single area (foreach channel, animal)
- `main_ternary`: Ternary plot for each region in triangle channel space
- `main_expr_range`: Plot the roi distribution range in AP/ML axis for batch animals
- `main_expr_level`: Plot all the ROI expression histogram along the certain anatomical axi
- `main_roi_view`: TODO clean

### [behavioral](src/rscvp/behavioral)

**[publish]**

- `main_batch`: Plot multiple (batch) animals for treadmill behavioral analysis. Check
  `@DispatchOption.dispatch()` decorated on methods for different analysis types

**[optional]**

- `main_summary`: Plot single animal in treadmill behavioral overview

### [model](src/rscvp/model)

**[publish]**

- `bayes_decoding.cache_bayes`: Build the cache for bayes decoding animal's position, used for further
  analysis/plotting
- `bayes_decoding.main_decode_analysis`: Decoding analysis based on the existing cache. Check
  `@DispatchOption.dispatch()` decorated on methods for different analysis types

### [retinotopic](src/rscvp/retinotopic)

**[publish]**

- `cache_retinotopic`: Build the cache for plot the retinotopic
- `main_retinotopic_map`: Plot the retinotopic map based on the existing cache

### [selection](src/rscvp/selection)

**[publish]**

- `main_neuropil_error`: Check if any error in neuropil extraction
- `main_trial_reliability`: See fraction of active trials in the linear treadmill task
- `main_cls_summary`: Quantification of proportion of visual/spatial/overlap/unclassified RSC neurons

**[optional]**

- `rastermap.main_rastermap_2p`: TODO clean
- `rastermap.main_rastermap_wfield`: TODO clean
- `glm.main_eval`: TODO clean

### [signal](src/rscvp/signal)

**[publish]**

- `main_dff_session`: Calculate the mean/median/percentile/max dff in every recording sessions

### [spatial](src/rscvp/spatial)

**[publish]**

- `main_si`: Calculate the spatial information, and plot with shuffle activity foreach cell
- `main_position_map`: Plot normalized position binned calcium activity across trial
- `main_place_field`: Place field properties calculations, including place field width, peak location, numbers
- `main_slb`: Calculate the spatial lower bound activity, and binned the shuffled activity to see the given percentile
  threshold
- `main_trial_corr`: Calculate median value of pairwise trial to trial activity correlation

**[optional]**

- `main_sparsity`: Calculate spatial sparsity
- `main_ev_pos`: Calculate explained variance of the position in a single trial
- `main_population_matrix`: TODO clean
- `main_align_map`: TODO clean

### [statistic](src/rscvp/statistic)

- TODO clean, together with statistic module

### [topology](src/rscvp/topology)

**[publish]**

- `main_fov`: Plot the recording FOV in both PMT channels and suite2p registered somata
- `main_roi_loc`: Get the xy actual coordinates (relative to Bregma, in um) for each cell
- `cache_ctype_cord`: Build cache for collection of coordinates, cell type information. AFTER PRESELECTION, and plot
  each cell type fraction histogram in ap/ml space
- `main_visual_topo`: Plot topographical distribution for visual metrics
- `main_spatial_topo`: Plot topographical distribution for spatial metrics

**[optional]**

- `main_cls`: TODO clean

### [track](src/rscvp/track)

- TODO see if included

### [visual](src/rscvp/visual)

**[publish]**

- `main_tuning`: Plot the calcium transients traces across different condition of visual stimulation(12 dir, 2tf, 3sf).
  Also make persistence cache for other analysis usage.
- `main_reliability`: Concatenate the calcium traces per visual-stimuli epoch, and compute the pairwise
  cross-correlation (reliability)
  to determine if is a visually-responsive cell
- `main_polar`: Plot the neural responses in different combination of sf-tf.
  Different direction of visual stimulus represents as polar plot.
- `main_sftf_pref`: Plot the visual sftf preference in dot plot (either fraction of cell, or activity amplitude).
  NOTE that this script should run `main_polar` first.

**[optional]**

- `main_sftf_map`: Plot the neural activity heatmap of pre/post visual stimulation windows in selected neurons
- `main_sftf_fit`: TODO clean

### [util](src/rscvp/util)

- `position`: Calculate the position binned signal. `Array[float, [N, L, B]]`.

-----------------------------

## Annotation for published figures

- Checkout class/function with `@publish_annotation('main' | 'sup',project='rscvp', figure=[FIGURE_NUMBER])` decorator

-----------------------------

## Example Colab (For publication)

- Only for `@publish_annotation('main')` and `@publish_annotation('sup')`

-----------------------------

## Data Folder structure

## DATA STRUCTURE (See also in [io.py](./src/rscvp/util/io.py))

### Physiological dataset (per recording date/mouse id, i.e., 2P data)

    Analysis/
        ├── summary/ -- (1)
        └── [EXP_DATE]_[ANIMAL_ID]__[EXP_TYPE]_[USER]
              ├── plane[*NUM]/ -- (2)
              ├── behavior/ -- (3)
              ├── cache/ -- (4)
              ├── track/ -- (5)
              └── *optional/ --(6)

* (1) For statistical and population visualization purposes
* (2) Different optical planes physiological data
* (3) Animal's behavior-related data
* (4) Cache files for avoid duplicated computing (e.g., .pkl, .npy., etc.)
* (5) Camera tracking data (e.g., pupil, running, location., etc.)
* (6) Preprocessing data storage (e.g., suite2p, kilosort., etc.)

### Anatomical dataset (per animal, i.e., posthoc histology)

     histology/
            ├── [ANIMAL_ID]/
            └── summary/

- Detail can be found in [histology readme](src/rscvp/atlas/README.md)

---

### Example of cli usage

  ```bash
  python -m <module.py> <subparser> \
  -S $STIMPY_ROOT \
  -S $S2P_ROOT \
  -D $EXP_DATE \
  -A $ANIMAL_ID \ 
  <options>
  ```

### GPU required installation

- Run **cellpose**, **stardist**, **rastemap**, **facemap**
- torch
    - Install nvidia driver, check by run `nvcc --version`, `nvidia-smi` for detail information
    - [pip install recommanded](https://pytorch.org/get-started/locally/)
    - `python -c 'import torch; print(torch.cuda.is_available())'`
    - For linux os `sudo apt install libxcb-cursor0` if qt dependencies error
    - Mac OS (mps backend) is under development...
  