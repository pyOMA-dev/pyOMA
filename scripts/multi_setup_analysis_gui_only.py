"""
pyOMA – Multi-Setup OMA, (almost) GUI-only workflow
=====================================================

Run this script from the repository root::

    python scripts/multi_setup_analysis_gui_only.py

Requirements: pip install "pyOMA[gui]"

This is the multi-setup counterpart of ``single_setup_analysis_gui_only.py``:
it pushes every analysis decision - PoSER vs. PoGER, geometry, how many
setups, decimation, correlation/PSD parameters, the system-identification
method and its build/compute steps, stabilization thresholds/pole selection,
and the PoGER block-column/model-order settings - into
:class:`~pyOMA.GUI.MultiSetupGUI.MultiSetupGUI` itself. What's left below is
only what that GUI has no equivalent for (yet): telling PreProcessSignals how
to read the measurement file format. Everything else, including picking each
setup's config/measurement/channel-DOF files, happens interactively once the
window opens - for each setup, add a tab via "Add Setup", then in that tab
click "Pre-process Signals...", which opens
:class:`~pyOMA.GUI.PreProcessSignalsGUI.PreProcessSignalsGUI` empty; its
File -> Load Config... prompts for a config file, then a measurement file.

Suggested inputs, from this repository's bundled example data
(``tests/files/``), one setup per tab:

- Config file:      ``tests/files/measurement_<n>/setup_info.txt``
- Measurement file:  ``tests/files/measurement_<n>/measurement_<n>.npy``
- Channel-DOF file:  ``tests/files/measurement_<n>/channel_dofs.txt``
  (optional - load via that same window's File -> Load Channel DOFs...;
  shown here because this example's geometry is known upfront, but channel
  deletion in setup_info.txt no longer requires it)

Geometry (optional, shared across all setups, loaded via "Load Geometry..."):

- Nodes:               ``tests/files/grid.txt``
- Lines:                ``tests/files/lines.txt``
- Parent-child assignments: ``tests/files/parent_child_assignments.txt``

For a workflow that needs none of the files above - no config file,
no chan_dofs file, not even this script - see "Quickest start" in
doc/gui_usage.rst: run ``pyoma``, pick Single Setup mode, and use
"Import Signals..." to load a bare .npy/.npz file directly.
"""
import numpy as np

from pyOMA.core import PreProcessSignals
from pyOMA.GUI.MultiSetupGUI import start_multi_setup_gui

# Tell pyOMA how to read .npy files (replace for other formats)
PreProcessSignals.load_measurement_file = np.load

# Starts with no setups and no geometry - add setups, pick PoSER/PoGER, and
# optionally load geometry from the window itself. The chosen mode's merge
# result (MergePoSER or PogerSSICovRef) is returned once the window closes.
merged_data = start_multi_setup_gui()
