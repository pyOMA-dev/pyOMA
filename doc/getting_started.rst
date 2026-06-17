Getting Started
===============

This page explains the core pyOMA workflow step by step.  After reading it you
should be able to set up your own analysis without looking at the source code.

.. contents:: On this page
   :local:
   :depth: 2


Installation
------------

Download and install anaconda or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

If you haven't setup your environment already:
.. code-block:: bash
   conda create --name pyoma
   conda activate pyoma
   conda install git pip


Activate the environment and clone the repository:

.. code-block:: bash
   conda activate pyoma
   git clone https://github.com/pyOMA-dev/pyOMA.git
   cd pyOMA

For Jupyter notebook support (interactive stabilisation and mode-shape widgets):

.. code-block:: bash

   pip install -e ".[jupyter]"

For the desktop PyQt5 GUI:

.. code-block:: bash

   pip install -e ".[gui]"

Or with both GUI options:

.. code-block:: bash

   pip install -e ".[jupyter, gui]"

The five-step workflow
----------------------

Every OMA analysis follows the same five stages:

.. list-table::
   :header-rows: 1
   :widths: 5 25 70

   * - Step
     - Class / method
     - What happens
   * - 1
     - :class:`~pyOMA.core.PreProcessingTools.GeometryProcessor`
     - Load node coordinates and structural connectivity
   * - 2
     - :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals`
     - Load the time-series data; assign sampling rate, channel types, and reference channels
   * - 3
     - :meth:`~pyOMA.core.PreProcessingTools.PreProcessSignals.correlation`
     - Decimate, filter, and compute cross-correlation functions
   * - 4
     - :class:`~pyOMA.core.SSICovRef.BRSSICovRef` *or other method*
     - Identify modal parameters at multiple model orders
   * - 5
     - :class:`~pyOMA.core.StabilDiagram.StabilCalc`
     - Compute the stabilisation diagram and select physical modes


Step 1 – Structural geometry
-----------------------------

.. code-block:: python

   from pyOMA.core import GeometryProcessor

   geometry_data = GeometryProcessor.load_geometry(
       nodes_file='grid.txt',
       lines_file='lines.txt',               # optional
       parent_childs_file='parent_childs.txt',  # optional
   )

``GeometryProcessor`` stores node coordinates and connectivity.  It is only
required for mode-shape visualisation; you can skip it for a numbers-only
analysis.

File formats are described on the :doc:`input_file_formats` page.


Step 2 – Loading measurement signals
-------------------------------------

Providing a file loader
~~~~~~~~~~~~~~~~~~~~~~~

pyOMA is format-agnostic.  Before calling ``init_from_config`` (or
constructing :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals`
directly), you must assign a callable to the class attribute
``load_measurement_file``.  It receives the file path and must return an
``(n_samples, n_channels)`` NumPy array:

.. code-block:: python

   import numpy as np
   from pyOMA.core import PreProcessSignals

   # NumPy .npy
   PreProcessSignals.load_measurement_file = np.load

   # Whitespace-separated ASCII
   PreProcessSignals.load_measurement_file = lambda f, **kw: np.loadtxt(f)

   # Custom binary
   PreProcessSignals.load_measurement_file = my_loader

Loading signals directly in Python (recommended for notebooks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals`
directly from a NumPy array.  This is the preferred approach when working
interactively — all parameters stay in the notebook and there is no need to
maintain separate text files:

.. code-block:: python

   import numpy as np
   from pyOMA.core import PreProcessSignals

   signals = np.load('my_measurement.npy')   # shape (n_samples, n_channels)

   prep_signals = PreProcessSignals(
       signals,
       sampling_rate=256,           # Hz
       ref_channels=[2, 3],         # column indices of reference sensors
       accel_channels=[0, 1, 2, 3], # columns measured with accelerometers
       velo_channels=[],            # columns measured with velocimeters
       disp_channels=[],            # columns measured with displacement sensors
       setup_name='my_setup',
   )

   # Assign channel-DOF information for mode-shape plotting
   # (chan, node_name, azimuth_deg, elevation_deg)
   prep_signals.chan_dofs = [
       (0, '5',  28.9, -8.7, 'vib_l'),
       (1, '5',  81.0, -7.3, 'vib_r'),
       (2, '24',  0.0, 180.0, 'ref_x'),
       (3, '24', -90.0,  0.0, 'ref_y'),
   ]


Step 3 – Pre-processing
------------------------

Decimation
~~~~~~~~~~

:meth:`~pyOMA.core.PreProcessingTools.PreProcessSignals.decimate_signals`
reduces the sampling rate by an integer factor.  An anti-aliasing filter is
applied automatically before down-sampling.  Call it multiple times to achieve
large total reduction factors while keeping each step moderate:

.. code-block:: python

   prep_signals.decimate_signals(3)   # 256 Hz → 85.3 Hz
   prep_signals.decimate_signals(3)   # 85.3 Hz → 28.4 Hz

Filtering
~~~~~~~~~

An optional explicit bandpass filter can be applied before computing
correlations if you want to restrict the analysis to a specific frequency band:

.. code-block:: python

   prep_signals.filter_signals(lowpass=10.0, highpass=0.1)  # Hz

Correlation functions
~~~~~~~~~~~~~~~~~~~~~

Covariance-driven SSI and PLSCF need the cross-correlation matrix.  The
Blackman–Tukey method generally gives better frequency resolution:

.. code-block:: python

   prep_signals.corr_blackman_tukey(m_lags=200)  # or corr_welch(m_lags=200)

``m_lags`` must satisfy ``m_lags > num_block_columns + num_block_rows``
(see Step 4 below).


Step 4 – System identification
-------------------------------

All identification methods share the same interface.  Pick the class, set the
parameters, call the two core methods:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Class
     - What it needs
     - Best for
   * - :class:`~pyOMA.core.SSICovRef.BRSSICovRef`
     - Correlations (Step 3)
     - General ambient vibration, most widely used
   * - :class:`~pyOMA.core.SSIData.SSIData`
     - Raw signals
     - Shorter records; avoids explicit correlation step
   * - :class:`~pyOMA.core.VarSSIRef.VarSSIRef`
     - Correlations
     - When uncertainty (variance) of modal parameters is required
   * - :class:`~pyOMA.core.PLSCF.PLSCF`
     - Correlations
     - Frequency-domain alternative to SSI
   * - :class:`~pyOMA.core.ERA.ERA`
     - Impulse response data
     - Impact-hammer and FRF tests

Example using **SSI-cov** (recommended — all parameters visible in the notebook):

.. code-block:: python

   from pyOMA.core import BRSSICovRef

   modal_data = BRSSICovRef(prep_signals)
   modal_data.build_toeplitz_cov(num_block_columns=100)  # must be < m_lags
   modal_data.compute_modal_params(max_model_order=40)

Example using **SSI-data** directly:

.. code-block:: python

   from pyOMA.core import SSIData

   ssi = SSIData(prep_signals)
   ssi.build_block_hankel(num_block_rows=100)
   ssi.compute_modal_params(max_model_order=40)

Example using **PLSCF** directly:

.. code-block:: python

   from pyOMA.core import PLSCF

   plscf = PLSCF(prep_signals)
   plscf.build_half_spectra(
       nperseg=200,            # correlation lag length (same as m_lags)
       begin_frequency=0.0,   # Hz
       end_frequency=12.0,    # Hz
   )
   plscf.compute_modal_params(max_model_order=40)

Key parameters
~~~~~~~~~~~~~~

``num_block_columns`` / ``num_block_rows``
   Number of block rows/columns in the block-Toeplitz or Hankel matrix.  Higher
   values capture longer time correlations but increase computation time.
   Typical range: 50–200.  Must satisfy
   ``num_block_columns + num_block_rows < m_lags``.

``max_model_order``
   The algorithm estimates modal parameters at every order from 1 up to this
   value.  Stable physical modes appear across many orders.  Typical range:
   20–100.  Setting it to ``2 × expected_number_of_modes`` is a safe starting
   point.


Step 5 – Stabilisation diagram
--------------------------------

.. code-block:: python

   from pyOMA.core import StabilCluster, StabilPlot

   stabil_calc = StabilCluster(modal_data)

   # Compute hard-criteria masks
   stabil_calc.calculate_stabilization_masks(
       d_range=(0, 0.10),   # damping 0–10 %
       df_max=0.01,          # max relative frequency change between orders
       dd_max=0.05,          # max relative damping change between orders
       dmac_max=0.05,        # max MAC change between orders
   )

   # Static plot
   stabil_plot = StabilPlot(stabil_calc)
   stabil_plot.plot()

   # Automated mode selection
   stabil_calc.automatic_clearing()
   stabil_calc.automatic_classification()

   # Export results to a tab-separated text file
   stabil_calc.export_results('modes.txt')

Pole stability criteria
~~~~~~~~~~~~~~~~~~~~~~~

``d_range``
   Absolute damping ratio limits ``(min, max)``.

``df_max``, ``dd_max``
   Maximum *relative* change in frequency / damping between consecutive model
   orders.  A pole that shifts by more than ``df_max × 100 %`` between order
   *n* and *n+1* is considered new/unstable.

``dmac_max``
   Maximum MAC difference between mode shapes at consecutive orders.  Ensures
   that the shape is consistent and not just a numerical artefact.


Choosing thresholds
~~~~~~~~~~~~~~~~~~~

There is no universal set of thresholds.  As a starting point:

- For lightly damped civil structures: ``df_max=0.01``, ``dd_max=0.05``,
  ``dmac_max=0.05``, ``d_range=(0, 0.05)``.
- For more heavily damped mechanical systems: loosen ``d_range`` and
  ``dd_max`` accordingly.
- Always inspect the raw stabilisation diagram visually before accepting
  automatically selected modes.


Example scripts and notebooks
------------------------------

Three ready-to-run examples are included in the ``scripts/`` directory.
Each is available as a plain Python script (requires ``pip install "pyOMA[gui]"``)
and as an interactive Jupyter notebook (requires ``pip install "pyOMA[jupyter]"``):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Scenario
     - Script
     - Notebook
   * - Single setup
     - ``scripts/single_setup_analysis.py``
     - :doc:`_collections/single_setup_analysis`
   * - Multi-setup — PoSER (post-identification merging)
     - ``scripts/multi_setup_analysis.py``
     - :doc:`_collections/multi_setup_analysis`
   * - Multi-setup — PoGER (pre-identification merging)
     - ``scripts/multi_setup_analysis_poger.py``
     - :doc:`_collections/multi_setup_analysis_poger`

The PoSER workflow runs SSI independently on each measurement setup and then
merges the estimated modal parameters using
:class:`~pyOMA.core.PostProcessingTools.MergePoSER`.  The PoGER workflow stacks
correlation functions from all setups into a joint Hankel matrix before a single
SSI run, yielding global frequencies, damping ratios, and re-scaled mode shapes
directly via :class:`~pyOMA.core.SSICovRef.PogerSSICovRef`.
