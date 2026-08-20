MultiIndex visualization
========================

Visualization support is optional so storage and sweep execution do not pull a
graphical stack into headless systems:

.. code-block:: console

   python -m pip install "pynst[visualization]"

Headless selection
------------------

:class:`pynst.visualization.MultiIndexSelector` contains the deterministic
selection logic. It can inspect observed level values, select single or coupled
levels, operate on either DataFrame axis, and optionally drop selected levels.
It requires pandas but neither a notebook nor Matplotlib.

Interactive notebooks
---------------------

:class:`pynst.visualization.InteractiveMultiIndexPlotter` builds ipywidgets on
top of the selector. ``widget()`` constructs and returns the widget without
displaying it; ``show()`` performs the explicit display. This avoids duplicate
figures during notebook execution and documentation builds.

Matplotlib
----------

:func:`pynst.visualization.matplotlib.plot_mi` plots each MultiIndex-labelled
column as a separate trace on an ordinary Matplotlib axis and generates labels
from selected level names. PyNST does not register a custom Matplotlib
projection and does not modify global plotting state.

See :doc:`tutorials/03_visualization` for an executable example.
