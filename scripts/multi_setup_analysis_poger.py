"""
pyOMA – Multi-Setup OMA with PoGER merging
==========================================

Run this script from the repository root::

    python scripts/multi_setup_analysis_poger.py

Requirements: pip install "pyOMA[gui]"

PoGER (Post Global Estimation Re-scaling) merges setups *before* modal
identification: correlation functions from all setups are stacked into a
joint Hankel matrix, and a single SSI run yields global frequencies,
damping ratios, and re-scaled mode shapes.  This contrasts with PoSER
(see multi_setup_analysis.py), where SSI is run per setup and the modal
parameters are merged afterwards.

Set MANUAL_POLE_SELECTION=True to open a StabilGUI for interactive pole
selection.  Set SHOW_MODE_SHAPES=True to open PlotMSHGUI after identification.
"""
from pathlib import Path
import numpy as np

from pyOMA.core import (
    GeometryProcessor,
    PreProcessSignals,
    PogerSSICovRef,
    StabilCluster,
    StabilPlot,
    ModeShapePlot,
)
from pyOMA.core.Helpers import ConfigFile
from pyOMA.GUI.StabilGUI import start_stabil_gui
from pyOMA.GUI.PlotMSHGUI import start_msh_gui

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / 'tests' / 'files'

CONF_FILE = EXAMPLE_DATA / 'ssi_config.txt'

# Stabilisation thresholds
STABIL_KWARGS = dict(
    d_range=(0, 0.50),
    df_max=0.01,
    dd_max=0.05,
    dmac_max=0.05,
)

# Set to True to reload saved intermediate results instead of recomputing
SKIP_EXISTING = False
SAVE_RESULTS  = False

# Set to True to open StabilGUI for manual pole selection.
# When False, automatic_selection() is used instead.
MANUAL_POLE_SELECTION = True

# Set to True to open PlotMSHGUI after identification for mode shape inspection.
SHOW_MODE_SHAPES = True
# ─────────────────────────────────────────────────────────────────────────────

PreProcessSignals.load_measurement_file = np.load

# ── Step 1: Shared geometry ───────────────────────────────────────────────────
geometry_data = GeometryProcessor.load_geometry(
    nodes_file=EXAMPLE_DATA / 'grid.txt',
    lines_file=EXAMPLE_DATA / 'lines.txt',
    parent_childs_file=EXAMPLE_DATA / 'parent_child_assignments.txt',
)

# ── Step 2: Pre-process each setup and add to PoGER object ───────────────────
setup_dirs = sorted(
    EXAMPLE_DATA.glob('measurement_*'),
    key=lambda p: int(p.name.split('_')[1]),
)

poger = PogerSSICovRef()

for setup_dir in setup_dirs:
    meas_name = setup_dir.name
    print(f'\n── {meas_name} ──')

    _prep_state = setup_dir / 'prep_data.npz'
    if _prep_state.exists() and SKIP_EXISTING:
        prep_signals = PreProcessSignals.load_state(_prep_state)
    else:
        prep_signals = PreProcessSignals.init_from_config(
            conf_file=setup_dir / 'setup_info.txt',
            meas_file=setup_dir / f'{meas_name}.npy',
            chan_dofs_file=setup_dir / 'channel_dofs.txt',
        )
        prep_signals.decimate_signals(3)
        prep_signals.decimate_signals(3)
        prep_signals.correlation(m_lags=401)
        prep_signals.psd()
        if SAVE_RESULTS:
            prep_signals.save_state(_prep_state)

    poger.add_setup(prep_signals)

# ── Step 3: Joint identification via PoGER ────────────────────────────────────
print('\n── Joint PoGER identification ──')
poger.pair_channels()

cfg = ConfigFile(CONF_FILE)
num_block_columns = cfg.int('Number of Block-Columns')
max_model_order   = cfg.int('Maximum Model Order')

_poger_state = EXAMPLE_DATA / 'poger_modal_data.npz'
if _poger_state.exists() and SKIP_EXISTING:
    poger = PogerSSICovRef.load_state(_poger_state)
else:
    poger.build_merged_subspace_matrix(num_block_columns)
    poger.compute_modal_params(max_model_order)
    if SAVE_RESULTS:
        poger.save_state(_poger_state)

# ── Step 4: Stabilisation and pole selection ──────────────────────────────────
stabil_calc = StabilCluster(poger)
stabil_calc.calculate_stabilization_masks(**STABIL_KWARGS)

if MANUAL_POLE_SELECTION:
    stabil_plot = StabilPlot(stabil_calc)
    start_stabil_gui(stabil_plot, poger, geometry_data, poger.prep_signals)
else:
    stabil_calc.automatic_selection()

n_selected = len(stabil_calc.select_modes)
verb = 'manually' if MANUAL_POLE_SELECTION else 'automatically'
print(f'\n   → {n_selected} mode(s) {verb} selected')

# ── Step 5: Print results ─────────────────────────────────────────────────────
selected_freqs = [poger.modal_frequencies[i] for i in stabil_calc.select_modes]
selected_damps = [poger.modal_damping[i]      for i in stabil_calc.select_modes]

print(f'\nPoGER results  ({len(poger.setups)} setup(s), {n_selected} mode(s))')
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
        modal_data=poger,
        prep_signals=poger.prep_signals,
        amplitude=20,
    )
    start_msh_gui(mode_shape_plot)
