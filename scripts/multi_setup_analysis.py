"""
pyOMA – Multi-Setup OMA with PoSER merging
==========================================

Run this script from the repository root::

    python scripts/multi_setup_analysis.py

Requirements: pip install "pyOMA[gui]"

Each sub-directory matching tests/files/measurement_* is treated as one
measurement setup.  Modal analysis is performed on all setups, pole
selection is carried out via StabilCluster (automatically or interactively),
and results are merged across setups using MergePoSER.

Set MANUAL_POLE_SELECTION=True to open a StabilGUI for each setup so that
poles can be picked manually.  Set SHOW_MODE_SHAPES=True to open PlotMSHGUI
after merging to inspect the merged mode shapes.
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
from pyOMA.core.PostProcessingTools import MergePoSER
from pyOMA.GUI.StabilGUI import start_stabil_gui
from pyOMA.GUI.PlotMSHGUI import start_msh_gui

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / 'tests' / 'files'

# OMA method – change to SSIData, PLSCF, VarSSIRef, etc.
METHOD = BRSSICovRef

_CONF_FILES = {
    BRSSICovRef: 'ssi_config.txt',
    SSIData:     'ssi_config.txt',
    PLSCF:       'plscf_config.txt',
    PRCE:        'prce_config.txt',
    VarSSIRef:   'varssi_config.txt',
}
CONF_FILE = EXAMPLE_DATA / _CONF_FILES[METHOD]

# Stabilisation thresholds (same as single_setup_analysis.py)
STABIL_KWARGS = dict(
    d_range=(0, 0.50),
    df_max=0.01,
    dd_max=0.05,
    dmac_max=0.05,
)

# Set to True to reload saved intermediate results instead of recomputing
SKIP_EXISTING = False
SAVE_RESULTS  = False

# Set to True to open StabilGUI per setup for manual pole selection.
# When False, automatic_selection() is used instead.
MANUAL_POLE_SELECTION = False

# Set to True to open PlotMSHGUI after merging for mode shape inspection.
SHOW_MODE_SHAPES = True
# ─────────────────────────────────────────────────────────────────────────────

PreProcessSignals.load_measurement_file = np.load

# ── Step 1: Shared geometry ───────────────────────────────────────────────────
geometry_data = GeometryProcessor.load_geometry(
    nodes_file=EXAMPLE_DATA / 'grid.txt',
    lines_file=EXAMPLE_DATA / 'lines.txt',
    parent_childs_file=EXAMPLE_DATA / 'parent_child_assignments.txt',
)

# ── Step 2–4: Per-setup loop ──────────────────────────────────────────────────
merger = MergePoSER()

setup_dirs = sorted(
    EXAMPLE_DATA.glob('measurement_*'),
    key=lambda p: int(p.name.split('_')[1]),
)

for setup_dir in setup_dirs:
    meas_name = setup_dir.name
    print(f'\n── {meas_name} ──')

    # ── Step 2: Signal pre-processing ────────────────────────────────────────
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
        prep_signals.correlation(m_lags=200)
        prep_signals.psd()
        if SAVE_RESULTS:
            prep_signals.save_state(_prep_state)

    # ── Step 3: System identification ────────────────────────────────────────
    _modal_state = setup_dir / 'modal_data.npz'
    if _modal_state.exists() and SKIP_EXISTING:
        modal_data = METHOD.load_state(_modal_state, prep_signals)
    else:
        modal_data = METHOD.init_from_config(CONF_FILE, prep_signals)
        if SAVE_RESULTS:
            modal_data.save_state(_modal_state)

    # ── Step 4: Pole selection via StabilCluster ──────────────────────────────
    _stabil_state = setup_dir / 'stabil_data.npz'
    if _stabil_state.exists() and SKIP_EXISTING:
        from pyOMA.core import StabilCalc
        stabil_calc = StabilCalc.load_state(_stabil_state, modal_data)
    else:
        stabil_calc = StabilCluster(modal_data)
        stabil_calc.calculate_stabilization_masks(**STABIL_KWARGS)
        if MANUAL_POLE_SELECTION:
            stabil_plot = StabilPlot(stabil_calc)
            start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_signals)
        else:
            stabil_calc.automatic_selection()
        if SAVE_RESULTS:
            stabil_calc.save_state(_stabil_state)

    n_selected = len(stabil_calc.select_modes)
    verb = 'manually' if MANUAL_POLE_SELECTION else 'automatically'
    print(f'   → {n_selected} mode(s) {verb} selected')

    merger.add_setup(prep_signals, modal_data, stabil_calc)

# ── Step 5: Merge across setups ───────────────────────────────────────────────
print('\n── Merging setups ──')
merger.merge()

_merged_state = EXAMPLE_DATA / 'measurement_15' / 'merged_setups.npz'
if SAVE_RESULTS:
    merger.save_state(_merged_state)

# ── Step 6: Print results ─────────────────────────────────────────────────────
freqs = merger.mean_frequencies[:, 0]    # shape (n_modes,)
damps = merger.mean_damping[:, 0]
std_f = merger.std_frequencies[:, 0]
std_d = merger.std_damping[:, 0]
n_modes  = len(freqs)
n_setups = len(merger.setups)

print(f'\nMerged PoSER results  ({n_setups} setup(s), {n_modes} mode(s))')
print('─' * 56)
print(f'  {"#":>3}  {"Freq [Hz]":>10}  {"Damp [%]":>10}  {"σ_f [Hz]":>10}  {"σ_d [%]":>10}')
print('─' * 56)
for i, (f, d, sf, sd) in enumerate(zip(freqs, damps, std_f, std_d), 1):
    print(f'  {i:>3}  {f:>10.3f}  {d:>10.3f}  {sf:>10.4f}  {sd:>10.4f}')
print('─' * 56)

# ── Step 7: Mode shape visualisation ─────────────────────────────────────────
if SHOW_MODE_SHAPES:
    mode_shape_plot = ModeShapePlot(
        geometry_data=geometry_data,
        merged_data=merger,
        amplitude=20,
    )
    start_msh_gui(mode_shape_plot)
