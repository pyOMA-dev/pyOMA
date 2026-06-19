Input File Formats
==================

All pyOMA configuration files are plain-text, line-by-line key–value files.
Every key appears on one line and its value on the *next* line.  Lines starting
with ``#`` are comments.

.. contents:: On this page
   :local:
   :depth: 2

.. note::
   These files are only needed when using the ``init_from_config`` helper
   methods.  If you prefer to configure everything in Python code (recommended
   for Jupyter notebooks) you can pass parameters directly — see
   :doc:`getting_started`.


Measurement signals
-------------------

pyOMA is format-agnostic for the actual time-series data.  Before calling
``PreProcessSignals.init_from_config`` you must register a loader function that
reads your file format and returns a ``(n_samples, n_channels)`` NumPy array:

.. code-block:: python

   import numpy as np
   from pyOMA.core import PreProcessSignals

   # NumPy binary (.npy) — used by the bundled example data
   PreProcessSignals.load_measurement_file = np.load

   # Whitespace-separated ASCII (e.g. .txt, .dat)
   PreProcessSignals.load_measurement_file = lambda f, **kw: np.loadtxt(f)

   # MATLAB .mat file (requires scipy)
   import scipy.io
   PreProcessSignals.load_measurement_file = lambda f, **kw: scipy.io.loadmat(f)['data']

   # Any custom binary or proprietary format
   PreProcessSignals.load_measurement_file = my_loader_function

The loader receives the file path (``str`` or ``pathlib.Path``) and must return
a 2-D NumPy array of shape ``(n_samples, n_channels)``.  Columns correspond to
channels; the mapping from column index to channel type is defined in
``setup_info.txt`` (see below).

When working in Jupyter notebooks or in Python scripts, you can also bypass the
file loader entirely and construct
:class:`~pyOMA.core.PreProcessingTools.PreProcessSignals` directly from an array
— see :doc:`getting_started` for the direct-construction workflow.


Geometry files
--------------

grid.txt
~~~~~~~~

Node coordinates.  One header line, then one node per line.  Columns are
separated by whitespace (spaces or tabs).

.. code-block:: text

   node_name    x             y             z
   1    1.8100000e+01  -1.8100000e+01   0.0000000e+00
   2    1.8100000e+01   1.8100000e+01   0.0000000e+00
   3   -1.8100000e+01   1.8100000e+01   0.0000000e+00
   4   -1.8100000e+01  -1.8100000e+01   0.0000000e+00
   5    1.8100000e+01  -1.8100000e+01   1.9200000e+01

- ``node_name`` — a string label (any alphanumeric identifier)
- ``x``, ``y``, ``z`` — coordinates in any consistent unit (metres, cm, etc.)

lines.txt
~~~~~~~~~

Structural connectivity.  One header line, then one edge per line.

.. code-block:: text

   node_start   node_end
   1            2
   2            3
   3            4
   4            1

Node names must match exactly those in ``grid.txt``.

parent_child_assignments.txt (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines sensor-to-node rigid-body offsets for sensors that are not located
exactly on a structural node.  One header line, then one assignment per line.

.. code-block:: text

   node_parent  x_m  y_m  z_m  node_child  x_sl  y_sl  z_sl
   24           0.0  0.0  1.0  5           1.0   0.0  0.0

Each entry defines a parent node with its local-frame axis directions and a
child node.  Displacements are interpolated from parent to child during
mode-shape animation.


Measurement configuration
--------------------------

setup_info.txt
~~~~~~~~~~~~~~

Describes one measurement setup: sampling rate, channel classification, and
optional channel deletion.

.. code-block:: text

   Setup Name:
   my_setup
   Sampling Rate [Hz]:
   256
   Reference Channels:
   2 3
   Delete Channels:
   5
   Accel. Channels:
   2 3 4
   Velo. Channels:
   0 1
   Disp. Channels:

   ####
   Note:
   Channel numbering starts with 0.
   Numbers are space-separated; leave a line blank for "none".

- **Setup Name** — arbitrary string; used to label saved results.
- **Sampling Rate** — in Hz (float or integer).
- **Reference Channels** — column indices of the permanently installed sensors
  used as references for cross-spectrum / cross-correlation estimation.
- **Delete Channels** — column indices to discard at load time.  Remaining
  channels are renumbered consecutively starting from 0.
- **Accel. / Velo. / Disp. Channels** — every remaining channel must appear in
  exactly one of these three lists.  Used for unit conversion when computing
  mode shapes in displacement.

channel_dofs.txt
~~~~~~~~~~~~~~~~

Maps channel numbers to structural degrees of freedom for mode-shape plotting.

.. code-block:: text

   Channel-Nr.   Node   Azimuth   Elevation   Channel Name
   0             5       28.87     -8.70       vib_l
   1             5       80.98     -7.31       vib_r
   2             5       55.88     29.35       vib_t
   3             24       0.00    180.00       ref_x
   4             24     -90.00      0.00       ref_y

- **Channel-Nr.** — 0-based channel index (after deletions from
  ``setup_info.txt``).
- **Node** — node name matching an entry in ``grid.txt``.
- **Azimuth**, **Elevation** — sensor orientation in degrees.
  Azimuth is measured counter-clockwise from the x-axis in the x-y plane;
  elevation is measured up from the x-y plane.
- **Channel Name** — arbitrary label used in plots.


Method configuration files
---------------------------

ssi_config.txt (BRSSICovRef / SSIData / SSIDataMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Number of Block-Columns:
   200
   Maximum Model Order:
   40
   Use Multiprocessing:
   yes

- **Number of Block-Columns** — number of block columns in the Toeplitz /
  Hankel matrix.  Must satisfy
  ``block_columns + block_rows < m_lags`` where ``block_rows`` defaults to
  ``block_columns`` and ``m_lags`` is the lag length used in
  ``corr_blackman_tukey`` / ``corr_welch``.
- **Maximum Model Order** — model orders 1 … max_order are evaluated.
- **Use Multiprocessing** — ``yes`` / ``no`` (currently not used by all
  methods).

plscf_config.txt (PLSCF)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Begin Frequency:
   0
   End Frequency:
   20
   Samples per time segment:
   4096
   Maximum Model Order:
   50

- **Begin / End Frequency** — restrict the identified frequency band (Hz).
  Set ``Begin Frequency: 0`` and ``End Frequency`` to the Nyquist frequency to
  include everything.
- **Samples per time segment** — length of the correlation sequence used to
  build the half-spectra (equivalent to ``m_lags`` for SSI-cov).

varssi_config.txt (VarSSIRef)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Number of Block-Columns:
   200
   Maximum Model Order:
   100
   Use Multiprocessing:
   yes
   Number of Blocks:
   30
   Subspace Method (projection/covariance):
   covariance
   LSQ Method for A (pinv/qr):
   pinv
   Variance Algorithm (fast/slow):
   fast

Additional parameters compared to ``ssi_config.txt``:

- **Number of Blocks** — the record is split into this many non-overlapping
  blocks for Monte-Carlo variance estimation.
- **Subspace Method** — ``covariance`` uses block-Toeplitz covariance matrices
  (recommended); ``projection`` uses a direct subspace projection.
- **LSQ Method for A** — least-squares solver for the state matrix:
  ``pinv`` (Moore–Penrose pseudo-inverse) or ``qr`` (QR decomposition).
- **Variance Algorithm** — ``fast`` uses a linearisation approximation
  (recommended); ``slow`` uses the full sensitivity computation.


Configuring without files
--------------------------

All parameters that can be set via configuration files can also be set
directly in Python, which is the recommended approach in Jupyter notebooks.
See :doc:`getting_started` for the equivalent Python-only workflow.
