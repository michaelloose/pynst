PyNST
=====

PyNST is a generic toolkit for planning and executing nested parameter sweeps,
storing crash-consistent HDF5 chunks, safely resuming interrupted runs and
exploring multidimensional results through a domain-independent dataset API.

The library deliberately stays independent of instruments and measurement
domains. Hardware safety rules, calibration identities and domain-specific
validation belong in the measurement wrapper or in a specialised dataset
subclass.

Quick installation
------------------

Install PyNST from PyPI:

.. code-block:: console

   python -m pip install pynst

For a first complete sweep, continue with :doc:`getting_started`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   planning
   concepts
   persistence
   metadata
   visualization
   tutorials/index
   migration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Maintainers

   releasing

Project links
-------------

* `Source code <https://github.com/michaelloose/pynst>`_
* `Issue tracker <https://github.com/michaelloose/pynst/issues>`_
* `MIT license <https://github.com/michaelloose/pynst/blob/main/LICENSE>`_
