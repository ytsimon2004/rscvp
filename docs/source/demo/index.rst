Demo Run
========

Since the paper is not yet published, the example dataset is provided via a private link.
Please contact the author, Yu-Ting Wei (ytsimon2004@gmail.com), to request the access token.
Once received, replace the placeholder in ``demo/util.py``:

.. code-block:: python

    TOKEN = "your_token_here"
    #         ^^^^^^^^^^^^^^^ Replace with your token string (e.g., "mY1NSIsImRhdGEiOnt")


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
