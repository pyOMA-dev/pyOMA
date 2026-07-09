Signal processing
=================

.. currentmodule:: pyOMA.core

.. autosummary::
   :recursive:
   :toctree: core
   :template: custom-module-template.rst

   PreProcessingTools


Importing external measurement data
------------------------------------

The desktop GUI's :guilabel:`File > Import Signals...` action
(:class:`~pyOMA.GUI.PreProcessSignalsGUI`) opens either a pyOMA state file
(``.npz``, as written by :meth:`~pyOMA.core.PreProcessingTools.PreProcessSignals.save_state`)
or a bare ``.npy`` array of shape ``(n_samples, n_channels)`` — for the
latter you are prompted for the sampling rate, since a plain array carries
no such metadata.

For measurement formats other than ``.npy``/``.npz``, ``scripts/converters/``
contains small, self-contained example scripts to copy and adapt — each one
reads a specific file format and writes a ``.npz`` in the same schema
``PreProcessSignals.load_state()`` expects, so the result loads directly via
the GUI's "Import Signals" action or in a script. These are not a claim of
broad format support; they are a starting point — add an adapter for your
own DAQ export by copying the closest script.

============================ ======================================
Script                       Format
============================ ======================================
``csv_converter.py``         Delimited ASCII/CSV (via ``pandas``)
``mat_converter.py``         MATLAB ``.mat`` (via ``scipy.io``)
``wav_converter.py``         ``.wav`` (via ``scipy.io.wavfile``)
``tdms_converter.py``        National Instruments ``.tdms`` (via ``nptdms``, optional dependency)
============================ ======================================