# rscvp

- Pipeline for RSC visuo-spatial project (rscvp) analysis
- Data are acquired using internal customized package [Stimpy](https://bitbucket.org/activision/stimpy/src/master)
- Core analysis largely depending on the customized
  package [neuralib](https://neuralib.readthedocs.io/en/latest/index.html) and cli
  pipeline [argclz](https://argp.readthedocs.io/en/latest/)

------------------------------

## How to set up in a local machine?

### 1. Clone or download the project locally

### 2. Create conda environment

```bash
conda create -n rscvp python~=3.10.0 -y
conda activate rscvp
```

### 3. Install required packages

- First `cd` to the directory with [pyproject.toml](pyproject.toml)

```bash
pip install -e .[all]
```

------------------------------

## See the [documentation](https://rscvp.readthedocs.io/en/latest/) for more information


------------------------------

## Annotation for published figures

- Checkout class/function with `@publish_annotation('main' | 'sup',project='rscvp', figure=[FIGURE_NUMBER])` decorator

```python
from neuralib.util.verbose import publish_annotation


@publish_annotation('main', project='rscvp', figure=['fig.1A-1D', 'fig.2A-2F'], as_doc=True)
class FigureClass:

  @publish_annotation('main', project='rscvp', figure='fig.1B', as_doc=True)
  def run_figure1(self):
    ...

  @publish_annotation('main', project='rscvp', figure='fig.2C', as_doc=True)
  def run_figure2(self):
    ...

  def _prepare_data(self):
    pass


```

-----------------------------

## Data Folder structure (See also in [io.py](./src/rscvp/util/io.py))

### Physiological dataset (per recording date/mouse id, i.e., 2P data)

    phys/
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

     hist/
        ├── [ANIMAL_ID]/
        └── summary/

- Detail can be found in [histology readme](src/rscvp/atlas/README.md)

---------------------------

## Contact

Yu-Ting Wei (ytsimon2004@gmail.com)
