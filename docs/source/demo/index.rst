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

- Physiological Dataset

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/19CfLB2izsMFZvaanJwkDSIzG8LZNhoAh#scrollTo=voF1jwXdepZ7
   :alt: Open In Colab
   :width: 150px


- Anatomical Dataset

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1Xf8Ukc0PwpyllUyZtD6zhQgzJT40XmQo#scrollTo=eiijT14vYQjs
   :alt: Open In Colab
   :width: 150px


.. note::

    For anatomical data, requires ``allen_mouse_10um`` atlas data from `BrainGlobeAtlas <https://brainglobe.info/index.html>`_.
    First-time setup may take additional time.
