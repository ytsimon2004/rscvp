Welcome to rscvp's documentation!
====================================


Installation
-------------------------

- Clone the repo in from `rscvp <https://github.com/ytsimon2004/rscvp>`_

- Create conda env

.. prompt:: bash $

    conda create -n rscvp python=3.10 -y


- Create conda env

.. prompt:: bash $

    conda activate rscvp


- Installation of package

.. code-block:: bash

    pip install .[all]


Figure Instruction
-------------------------

.. toctree::
   :maxdepth: 2

   fig_instruction/index


Demo Run
-------------------------

.. toctree::
   :maxdepth: 2

   demo/index



API Reference
-------------------------

.. toctree::
   :maxdepth: 1
   :caption: Modules

   api/rscvp.atlas
   api/rscvp.model
   api/rscvp.retinotopic
   api/rscvp.selection
   api/rscvp.spatial
   api/rscvp.statistic
   api/rscvp.topology
   api/rscvp.visual
   api/rscvp.util