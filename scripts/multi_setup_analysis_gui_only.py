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
window opens.

Suggested inputs, from this repository's bundled example data
(``tests/files/``), one setup per tab added via "Add Setup":

- Config file:      ``tests/files/measurement_<n>/setup_info.txt``
- Measurement file:  ``tests/files/measurement_<n>/measurement_<n>.npy``
- Channel-DOF file:  ``tests/files/measurement_<n>/channel_dofs.txt``
  (required here - this example's setup_info.txt deletes a channel, and
  doing so currently requires chan_dofs to already be known, a pre-existing
  core limitation, not a GUI one)

Geometry (optional, shared across all setups, loaded via "Load Geometry..."):

- Nodes:               ``tests/files/grid.txt``
- Lines:                ``tests/files/lines.txt``
- Parent-child assignments: ``tests/files/parent_child_assignments.txt``
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
