Demo Run
========

Token Request
-------------------------

Since the paper is not yet published, the example dataset is provided via a private link.
Please contact the author, Yu-Ting Wei (ytsimon2004@gmail.com), to request an access token.

- Replace ``TOKEN = ...`` in the example scripts or colab notebook

.. code-block:: python

    TOKEN = "your_token_here"
    #        ^^^^^^^^^^^^^^^ Replace with your token string (e.g., "mY1NSIsImRhdGEiOnt")


Run on Local Module Path
-------------------------

- ``cd`` to the project root

.. code-block:: bash

    $ python -m demo.[EXAMPLE_SCRIPT]

.. seealso::

    `Demo examples on GitHub <https://github.com/ytsimon2004/rscvp/tree/main/demo>`_


Run on Google Colab
-------------------------

Copy the entire example script from `Demo examples on GitHub <https://github.com/ytsimon2004/rscvp/tree/main/demo>`_
and paste on the colab block

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1m7_3tHrPOjX5TRdMGJoor68RlySgTVbt
   :alt: Open In Colab
   :width: 150px



Physiological Figure Demo
-------------------------

**Dataset:** Calcium Imaging

.. list-table:: **Physiological Figure Demo**
   :widths: 30 60
   :header-rows: 1

   * - **reST Page**
     - **Associated Figures**
   * - :doc:`position_map`
     - ``Fig. 2A``, ``Fig. S1``, ``Fig. S5B–S5C``
   * - :doc:`position_decoding`
     - ``Fig. 3A–3C``
   * - :doc:`visual_reliability`
     - ``Fig. 4A``, ``Fig. S4``, ``Fig. S5B–S5C``
   * - :doc:`visual_tuning`
     - ``Fig. 5A``, ``Fig. 5E–5H``, ``Fig. S4``
   * - :doc:`visual_polars`
     - ``Fig. 5B``, ``Fig. 5E–5H (upper)``
   * - :doc:`visual_sftf`
     - ``Fig. 5C``


Anatomical Figure Demo
-----------------------

**Dataset:** Anatomical Tracing

.. note::

    Requires ``allen_mouse_10um`` atlas data from `BrainGlobeAtlas <https://brainglobe.info/index.html>`_.
    First-time setup may take additional time.

.. list-table:: **Anatomical Figure Demo**
   :widths: 30 60
   :header-rows: 1

   * - **reST Page**
     - **Associated Figures**
   * - :doc:`slice_registration`
     - ``Fig. 6E``
   * - :doc:`dorsal_top_view`
     - ``Fig. 6G``
   * - :doc:`region_reconstruction`
     - ``Fig. 7A–7G``
