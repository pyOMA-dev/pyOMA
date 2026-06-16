"""
pyOMA – Single-Setup OMA with PyQt5 desktop GUI
================================================

Run this script from the repository root::

    python scripts/single_setup_analysis.py

Requirements: pip install "pyOMA[gui]"

The script uses the example data bundled with the repository (tests/files/).
To analyse your own data adjust the path variables in the "Configuration"
section below and set PreProcessSignals.load_measurement_file to a callable
that reads your measurement file format.
"""
from pathlib import Path
import numpy as np

from pyOMA.core import (
    GeometryProcessor,
    PreProcessSignals,
    BRSSICovRef,
    SSIData,
    PLSCF,
    PRCE,
    VarSSIRef,
    StabilCluster,
    StabilPlot,
    ModeShapePlot,
)
from pyOMA.GUI.StabilGUI import start_stabil_gui
from pyOMA.GUI.PlotMSHGUI import start_msh_gui

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / 'tests' / 'files'
SETUP_DIR    = EXAMPLE_DATA / 'measurement_1'
MEAS_NAME    = 'measurement_1'

# OMA method – change to SSIData, PLSCF, VarSSIRef, etc.
METHOD    = VarSSIRef

_CONF_FILES = {
    BRSSICovRef: 'ssi_config.txt',
    SSIData:     'ssi_config.txt',
    PLSCF:       'plscf_config.txt',
    PRCE:        'prce_config.txt',
    VarSSIRef:   'varssi_config.txt',
}
CONF_FILE = EXAMPLE_DATA / _CONF_FILES[METHOD]

# Set to True to skip recomputation when saved results exist
SKIP_EXISTING = False
SAVE_RESULTS  = False
# ─────────────────────────────────────────────────────────────────────────────

# Tell pyOMA how to read .npy files (replace for other formats)
PreProcessSignals.load_measurement_file = np.load

# ── Step 1: Geometry ──────────────────────────────────────────────────────────
geometry_data = GeometryProcessor.load_geometry(
    nodes_file=EXAMPLE_DATA / 'grid.txt',
    lines_file=EXAMPLE_DATA / 'lines.txt',
    parent_childs_file=EXAMPLE_DATA / 'parent_child_assignments.txt',
)

# ── Step 2: Signal pre-processing ─────────────────────────────────────────────
_prep_state = SETUP_DIR / 'prep_signals.npz'
if _prep_state.exists() and SKIP_EXISTING:
    prep_signals = PreProcessSignals.load_state(_prep_state)
else:
    prep_signals = PreProcessSignals.init_from_config(
        conf_file=SETUP_DIR / 'setup_info.txt',
        meas_file=SETUP_DIR / (MEAS_NAME + '.npy'),
        chan_dofs_file=SETUP_DIR / 'channel_dofs.txt',
    )
    # Decimate 256 Hz → 28.4 Hz (two passes of ×3)
    prep_signals.decimate_signals(3)
    prep_signals.decimate_signals(3)
    # Compute cross-correlation functions required by SSI-cov
    prep_signals.correlation(m_lags=200)
    # Compute spectral densities required by PLSCF from the correlations
    prep_signals.psd()
    if SAVE_RESULTS:
        prep_signals.save_state(_prep_state)

# ── Step 3: System identification ─────────────────────────────────────────────
_modal_state = SETUP_DIR / 'modal_data.npz'
if _modal_state.exists() and SKIP_EXISTING:
    modal_data = METHOD.load_state(_modal_state, prep_signals)
else:
    modal_data = METHOD.init_from_config(CONF_FILE, prep_signals)
    if SAVE_RESULTS:
        modal_data.save_state(_modal_state)

# ── Step 4: Stabilisation diagram ─────────────────────────────────────────────
_stabil_state = SETUP_DIR / 'stabil_data.npz'
if _stabil_state.exists() and SKIP_EXISTING:
    from pyOMA.core import StabilCalc
    stabil_calc = StabilCalc.load_state(_stabil_state, modal_data)
else:
    stabil_calc = StabilCluster(modal_data)
    stabil_calc.calculate_stabilization_masks(
        d_range=(0, 0.10),
        df_max=0.01,
        dd_max=0.05,
        dmac_max=0.05,
    )

# ── Step 5: Interactive GUI ───────────────────────────────────────────────────
stabil_plot = StabilPlot(stabil_calc)
start_stabil_gui(stabil_plot, modal_data, geometry_data)

if SAVE_RESULTS:
    stabil_calc.save_state(_stabil_state)

# ── Step 6: Mode shape visualisation ─────────────────────────────────────────
mode_shape_plot = ModeShapePlot(
    geometry_data=geometry_data,
    stabil_calc=stabil_calc,
    modal_data=modal_data,
    prep_signals=prep_signals,
    amplitude=20,
)
start_msh_gui(mode_shape_plot)
