"""
pyOMA – Multi-Setup OMA with PreGER merging
============================================

Run this script from the repository root::

    python scripts/multi_setup_analysis_preger.py

Requirements: pip install "pyOMA[gui]"

PreGER (Pre-Global-Estimation-Re-scaling, Döhler, Lam & Mevel 2013) merges
setups *before* modal identification: each setup's observability is
normalised to a common modal basis before a single joint identification, in
contrast to PoGER's post-estimation re-scaling (see
multi_setup_analysis_poger.py). Unlike that script's manual
add_setup/pair_channels/build_merged_subspace_matrix/compute_modal_params
sequence, PreGERSSI (and VarPreGERSSI) expose a one-call
``init_from_config(conf_file, prep_signals_list)`` classmethod that runs the
whole merge+identification pipeline from a config file and a list of
already-preprocessed setups.

Set SHOW_GEOMETRY_GUI/SHOW_PREPROCESS_GUI=True to inspect the shared
geometry and each setup's pre-processed signals interactively as the loop
runs (mirrors multi_setup_analysis_poger.py).
Set MANUAL_POLE_SELECTION=True to open a StabilGUI for interactive pole
selection. Set SHOW_MODE_SHAPES=True to open PlotMSHGUI after identification.
Set USE_VARIANCE=True to identify with VarPreGERSSI instead of PreGERSSI,
propagating first-order modal-parameter uncertainties (requires block-wise
correlations per setup, computed below via corr_blackman_tukey's
n_segments).
"""
from pathlib import Path
import numpy as np

from pyOMA.core import (
    GeometryProcessor,
    PreProcessSignals,
    PreGERSSI,
    VarPreGERSSI,
    StabilCluster,
    StabilPlot,
    ModeShapePlot,
)
from pyOMA.GUI.StabilGUI import start_stabil_gui
from pyOMA.GUI.PlotMSHGUI import start_msh_gui
from pyOMA.GUI.PreProcessSignalsGUI import start_preprocess_gui
from pyOMA.GUI.GeometryProcessorGUI import start_geometry_processor_gui

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / 'tests' / 'files'

CONF_FILE = EXAMPLE_DATA / 'preger_config.txt'

# Number of blocks each setup's correlation is split into for the variance
# estimate; only used when USE_VARIANCE=True.
NUM_BLOCKS = 15

# Set to True to identify with VarPreGERSSI instead of PreGERSSI, adding
# first-order standard deviations of the modal parameters.
USE_VARIANCE = True

# Set to True to open StabilGUI for manual pole selection.
# When False, automatic_selection() is used instead.
MANUAL_POLE_SELECTION = True

# Set to True to open PlotMSHGUI after identification for mode shape inspection.
SHOW_MODE_SHAPES = True

# Set to True to inspect/edit the shared geometry interactively (nodes,
# lines, parent-child assignments) before it is used downstream.
SHOW_GEOMETRY_GUI = False

# Set to True to inspect/adjust each setup's pre-processed signals
# interactively (channel selection, filtering, decimation, PSD/correlation
# diagrams, ...) before it is pooled into the joint PreGER identification.
SHOW_PREPROCESS_GUI = False
# ─────────────────────────────────────────────────────────────────────────────

PreProcessSignals.load_measurement_file = np.load

# ── Step 1: Shared geometry ───────────────────────────────────────────────────
geometry_data = GeometryProcessor.load_geometry(
    nodes_file=EXAMPLE_DATA / 'grid.txt',
    lines_file=EXAMPLE_DATA / 'lines.txt',
    parent_childs_file=EXAMPLE_DATA / 'parent_child_assignments.txt',
)

if SHOW_GEOMETRY_GUI:
    start_geometry_processor_gui(geometry_data)

# ── Step 2: Pre-process each setup ────────────────────────────────────────────
setup_dirs = sorted(
    EXAMPLE_DATA.glob('measurement_*'),
    key=lambda p: int(p.name.split('_')[1]),
)

prep_signals_list = []
for setup_dir in setup_dirs:
    meas_name = setup_dir.name
    print(f'\n── {meas_name} ──')

    prep_signals = PreProcessSignals.init_from_config(
        conf_file=setup_dir / 'setup_info.txt',
        meas_file=setup_dir / f'{meas_name}.npy',
        chan_dofs_file=setup_dir / 'channel_dofs.txt',
    )
    prep_signals.decimate_signals(3)
    prep_signals.decimate_signals(3)
    
    if USE_VARIANCE:
        # VarPreGERSSI.add_setup requires block-wise correlations.
        prep_signals.corr_blackman_tukey(m_lags=200, n_segments=NUM_BLOCKS)
    else:
        prep_signals.correlation(m_lags=200)
    prep_signals.psd()

    if SHOW_PREPROCESS_GUI:
        start_preprocess_gui(prep_signals, geometry_data)

    prep_signals_list.append(prep_signals)

# ── Step 3: Joint identification via PreGER (one call) ────────────────────────
print('\n── Joint PreGER identification ──')
METHOD = VarPreGERSSI if USE_VARIANCE else PreGERSSI
preger = METHOD.init_from_config(CONF_FILE, prep_signals_list)

# ── Step 4: Stabilisation and pole selection ──────────────────────────────────
stabil_calc = StabilCluster(preger)
stabil_calc.calculate_stabilization_masks(
    d_range=(0, 0.50),
    df_max=0.01,
    dd_max=0.05,
    dmac_max=0.05,
)

if MANUAL_POLE_SELECTION:
    stabil_plot = StabilPlot(stabil_calc)
    start_stabil_gui(stabil_plot, preger, geometry_data, preger.prep_signals)
else:
    stabil_calc.automatic_selection()

n_selected = len(stabil_calc.select_modes)
verb = 'manually' if MANUAL_POLE_SELECTION else 'automatically'
print(f'\n   → {n_selected} mode(s) {verb} selected')

# ── Step 5: Print results ─────────────────────────────────────────────────────
selected_freqs = [preger.modal_frequencies[i] for i in stabil_calc.select_modes]
selected_damps = [preger.modal_damping[i]      for i in stabil_calc.select_modes]

print(f'\nPreGER results  ({len(preger.setups)} setup(s), {n_selected} mode(s))')
print('─' * 46)
print(f'  {"#":>3}  {"Freq [Hz]":>10}  {"Damp [%]":>10}')
print('─' * 46)
for i, (f, d) in enumerate(zip(selected_freqs, selected_damps), 1):
    print(f'  {i:>3}  {f:>10.3f}  {d:>10.3f}')
print('─' * 46)

# ── Step 6: Mode shape visualisation ─────────────────────────────────────────
if SHOW_MODE_SHAPES:
    mode_shape_plot = ModeShapePlot(
        geometry_data=geometry_data,
        stabil_calc=stabil_calc,
        modal_data=preger,
        prep_signals=preger.prep_signals,
        amplitude=20,
    )
    start_msh_gui(mode_shape_plot)
