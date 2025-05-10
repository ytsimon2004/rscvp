# rscvp

[![Documentation Status](https://readthedocs.org/projects/rscvp/badge/?version=latest)](https://rscvp.readthedocs.io/en/latest/)

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



## See the detail information and Demo in [documentation](https://rscvp.readthedocs.io/en/latest/)




## Annotation for published figures in the source code

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



## Contact

Yu-Ting Wei (ytsimon2004@gmail.com)
