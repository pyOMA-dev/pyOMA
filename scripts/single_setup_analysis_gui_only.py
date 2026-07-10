"""
pyOMA – Single-Setup OMA, (almost) GUI-only workflow
=====================================================

Run this script from the repository root::

    python scripts/single_setup_analysis_gui_only.py

Requirements: pip install "pyOMA[gui]"

This is a variant of ``single_setup_analysis.py`` that pushes every analysis
decision into the GUIs themselves instead of scripting it - no decimation
factor, correlation/PSD parameters, system-identification method, or
stabilization thresholds are chosen in this file. What's left below is only
what has no GUI equivalent (yet):

- telling PreProcessSignals how to read the measurement file format
- pointing at the raw setup/measurement/channel-DOF files: there is no
  "Load Measurement" dialog, unlike geometry, which is loaded interactively
  below via GeometryProcessorGUI's own file-loading buttons
- three constructor calls that just wire one GUI's output into the next
  GUI's input (StabilCluster / StabilPlot / ModeShapePlot) - these make no
  analysis choices, they only hand data from one window to the next

Everything else - geometry, channel-DOF assignment, filtering, decimation,
correlation/PSD parameters, the system-identification method and its
build/compute steps, stabilization thresholds, and mode selection - is done
interactively in the GUIs that open one after another below.

For a workflow that needs none of the files above - no config file,
no chan_dofs file, not even this script - see "Quickest start" in
doc/gui_usage.rst: run ``pyoma``, pick Single Setup mode, and use
"Import Signals..." to load a bare .npy/.npz file directly.
"""
from pathlib import Path
import numpy as np

from pyOMA.core import (
    GeometryProcessor,
    PreProcessSignals,
    StabilCluster,
    StabilPlot,
    ModeShapePlot,
)
from pyOMA.GUI.GeometryProcessorGUI import start_geometry_processor_gui
from pyOMA.GUI.PreProcessSignalsGUI import start_preprocess_gui
from pyOMA.GUI.ModalAnalysisGUI import start_modal_analysis_gui
from pyOMA.GUI.StabilGUI import start_stabil_gui
from pyOMA.GUI.PlotMSHGUI import start_msh_gui

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / 'tests' / 'files'
SETUP_DIR    = EXAMPLE_DATA / 'measurement_1'
MEAS_NAME    = 'measurement_1'
# ─────────────────────────────────────────────────────────────────────────────

# Tell pyOMA how to read .npy files (replace for other formats)
PreProcessSignals.load_measurement_file = np.load

# ── Step 1: Geometry ──────────────────────────────────────────────────────────
# Starts empty - use the window's File menu to load:
#   tests/files/grid.txt                      (Load nodes)
#   tests/files/lines.txt                     (Load lines)
#   tests/files/parent_child_assignments.txt  (Load parent-child assignments)
# Every load/edit action mutates this same instance in place, so geometry_data
# already holds everything loaded through the GUI once the window is closed.
geometry_data = GeometryProcessor()
start_geometry_processor_gui(geometry_data)

# ── Step 2: Signal pre-processing ─────────────────────────────────────────────
# The initial load has no GUI equivalent, so it stays scripted.
# chan_dofs_file is passed here only because this example's geometry is
# known upfront and it's convenient to assign DOFs at load time; it is no
# longer required just because setup_info.txt deletes a channel - pass
# chan_dofs_file=None and assign DOFs interactively via the GUI's "Add DOF"
# button instead, if you don't have geometry ready ahead of time.
# prep_signals = PreProcessSignals.init_from_config(
#     conf_file=SETUP_DIR / 'setup_info.txt',
#     meas_file=SETUP_DIR / (MEAS_NAME + '.npy'),
#     chan_dofs_file=SETUP_DIR / 'channel_dofs.txt',
# )
prep_signals=None

# Decimation, filtering, correlation/PSD parameters, and channel-DOF/name
# edits all happen interactively here - nothing is precomputed beforehand.
prep_signals = start_preprocess_gui(prep_signals, geometry_data)

# ── Step 3: System identification ─────────────────────────────────────────────
# No method or config file is chosen here - pick one from the combo box in the
# GUI and step through its Build/Compute buttons. modal_data ends up being
# whichever page's instance was active when the window closes.
modal_data = start_modal_analysis_gui(prep_signals)

# ── Step 4: Stabilisation diagram ─────────────────────────────────────────────
# StabilGUI computes default stabilization masks itself on first open;
# thresholds are then adjusted interactively from there.
stabil_calc = StabilCluster(modal_data)
stabil_plot = StabilPlot(stabil_calc)
start_stabil_gui(stabil_plot, modal_data, geometry_data)

# ── Step 5: Mode shape visualisation ─────────────────────────────────────────
mode_shape_plot = ModeShapePlot(
    geometry_data=geometry_data,
    stabil_calc=stabil_calc,
    modal_data=modal_data,
    prep_signals=prep_signals,
    amplitude=20,
)
start_msh_gui(mode_shape_plot)
