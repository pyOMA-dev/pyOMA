#!/usr/bin/env python
"""
Generate reference screenshots of pyOMA's PyQt6 desktop GUI windows, used by
doc/gui_usage.rst.

Run this after changing any GUI code (a .ui file, its generated/ui_*.py, or
the hand-written GUI module) so the committed screenshots reflect the current
UI, then commit the updated PNGs under doc/_static/gui/::

    python scripts/generate_gui_screenshots.py

Use --check in CI / pre-commit to verify the committed screenshots are not
older than the GUI sources they depict, without regenerating anything::

    python scripts/generate_gui_screenshots.py --check
"""
import argparse
import hashlib
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')  # headless Qt – must precede any Qt import
# The GUIs pick their mode-shape backend via resolve_mode_shape_backend(); pin
# matplotlib so screenshots stay consistent and headless (the pyvista backend
# cannot render into an offscreen-QPA QWidget). Must precede any pyOMA import.
os.environ.setdefault('PYOMA_MSH_BACKEND', 'matplotlib')

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless backend – must precede any other mpl import

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# These are disposable screenshot widgets, not real editing sessions - answer
# UnsavedChangesMixin's save-on-close prompt with "Discard"/"Continue" so
# widget.close() in _capture() below never blocks on a QMessageBox that
# nothing can click under the offscreen QPA platform.
# UnsavedChangesMixin builds its own QMessageBox instance and calls .exec()
# on it (needed to relabel the "Discard" button to "Continue"), so it's
# .exec that's patched here, not the static .question() convenience method.
QMessageBox.exec = lambda self: QMessageBox.StandardButton.Discard

from pyOMA.core.PreProcessingTools import GeometryProcessor, PreProcessSignals
from pyOMA.core.SSICovRef import BRSSICovRef
from pyOMA.core.StabilDiagram import StabilCluster, StabilPlot
from pyOMA.core.PlotMSH import ModeShapePlot

REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_SRC_DIR = REPO_ROOT / 'pyOMA' / 'GUI'
EXAMPLE_DATA = REPO_ROOT / 'tests' / 'files'
SETUP_DIR = EXAMPLE_DATA / 'measurement_1'
OUT_DIR = REPO_ROOT / 'doc' / '_static' / 'gui'
SOURCE_HASH_FILE = OUT_DIR / 'gui_source.sha256'

# Fixed capture size per top-level widget class – keeps screenshots
# reproducible across machines instead of relying on sizeHint(). Values for
# classes whose .ui already bakes in a size mirror that size; the rest
# (HistoPlot, ModeShapeGUI) have no baked-in size and are chosen here.
_SIZES = {
    'GeometryProcessorGUI': (1200, 800),
    'PreProcessSignalsGUI': (1250, 820),
    'ChanDofEditorGUI': (420, 560),
    'ModalAnalysisGUI': (700, 800),
    'StabilGUI': (966, 901),
    'ModeShapeGUI': (1200, 800),
    'MultiSetupGUI': (720, 820),
}

SCREENSHOT_NAMES = (
    'gui_geometry_processor',
    'gui_preprocess_signals',
    'gui_chan_dof_editor',
    'gui_modal_analysis',
    'gui_stabil_diagram',
    'gui_mode_shape',
    'gui_multi_setup',
)


def _build_geometry_data():
    return GeometryProcessor.load_geometry(
        nodes_file=EXAMPLE_DATA / 'grid.txt',
        lines_file=EXAMPLE_DATA / 'lines.txt',
        parent_childs_file=EXAMPLE_DATA / 'parent_child_assignments.txt',
    )


def _build_prep_signals():
    PreProcessSignals.load_measurement_file = np.load
    ps = PreProcessSignals.init_from_config(
        conf_file=SETUP_DIR / 'setup_info.txt',
        meas_file=SETUP_DIR / 'measurement_1.npy',
        chan_dofs_file=SETUP_DIR / 'channel_dofs.txt',
    )
    # Same pipeline as scripts/single_setup_analysis.py's Step 2.
    ps.decimate_signals(3)
    ps.decimate_signals(3)
    ps.correlation(m_lags=200)
    ps.psd()
    return ps


def _build_modal_data(prep_signals):
    return BRSSICovRef.init_from_config(EXAMPLE_DATA / 'ssi_config.txt', prep_signals)


def _build_stabil_calc(modal_data):
    sc = StabilCluster(modal_data)
    sc.calculate_stabilization_masks(
        d_range=(0, 0.10), df_max=0.01, dd_max=0.05, dmac_max=0.05)
    sc.automatic_clearing()
    sc.automatic_classification()
    sc.automatic_selection()
    return sc


def _build_multi_setup_gui(geometry_data, mode='PoSER'):
    """Two loaded (but not yet identified) setups in *mode*, so the tab bar,
    per-setup pipeline buttons, and merge row are all visible.

    *mode* is set explicitly rather than relying on the combo box's default
    selection - ``_MODES`` lists ``'Single Setup'`` first (it's the
    recommended default for interactive use), so leaving this implicit
    would silently change which mode this screenshot depicts."""
    from pyOMA.GUI.MultiSetupGUI import MultiSetupGUI

    form = MultiSetupGUI(geometry_data)
    form.combo_mode.setCurrentText(mode)
    for setup_dir in (EXAMPLE_DATA / 'measurement_1', EXAMPLE_DATA / 'measurement_2'):
        form._on_add_setup()
        tab = form._tabs[-1]
        # Loading itself now happens inside PreProcessSignalsGUI (opened via
        # "Pre-process Signals..."), not in this tab - build prep_signals
        # directly instead of driving that (now-modal) window here.
        tab.prep_signals = PreProcessSignals.init_from_config(
            conf_file=setup_dir / 'setup_info.txt',
            meas_file=setup_dir / f'{setup_dir.name}.npy',
            chan_dofs_file=setup_dir / 'channel_dofs.txt',
        )
        tab._refresh_status()
    return form


def _capture(widget, name):
    """Resize, force-draw, grab, and save one top-level widget as a PNG."""
    size = _SIZES[type(widget).__name__]
    widget.resize(*size)
    widget.show()
    for canvas in widget.findChildren(FigureCanvasQTAgg):
        canvas.draw()  # synchronous – draw_idle() may not have fired before grab()
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()  # let layout/paint events settle
    out_path = OUT_DIR / f'{name}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = widget.grab().save(str(out_path), 'PNG')
    widget.close()
    if not ok:
        raise RuntimeError(f"QPixmap.save() failed for {name!r} -> {out_path}")
    return out_path


def _capture_all():
    from pyOMA.GUI.GeometryProcessorGUI import GeometryProcessorGUI
    from pyOMA.GUI.PreProcessSignalsGUI import PreProcessSignalsGUI
    from pyOMA.GUI.ChanDofEditorGUI import ChanDofEditorGUI
    from pyOMA.GUI.ModalAnalysisGUI import ModalAnalysisGUI
    from pyOMA.GUI.StabilGUI import StabilGUI, ComplexPlot
    from pyOMA.GUI.PlotMSHGUI import ModeShapeGUI

    # Held alive for the duration of _capture_all() – Qt widgets require a
    # live QApplication, and nothing keeps this instance referenced otherwise.
    _app = QApplication.instance() or QApplication(sys.argv)

    geometry_data = _build_geometry_data()
    prep_signals = _build_prep_signals()
    modal_data = _build_modal_data(prep_signals)
    stabil_calc = _build_stabil_calc(modal_data)

    failures = []

    def _run(label, build_widget):
        try:
            path = _capture(build_widget(), label)
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        except Exception:
            print(f"FAILED capturing {label}:", file=sys.stderr)
            traceback.print_exc()
            failures.append(label)

    _run('gui_geometry_processor',
         lambda: GeometryProcessorGUI(geometry_data))
    _run('gui_preprocess_signals',
         lambda: PreProcessSignalsGUI(prep_signals, geometry_data))
    _run('gui_chan_dof_editor',
         lambda: ChanDofEditorGUI(prep_signals, geometry_data, channel=0))
    _run('gui_modal_analysis',
         lambda: ModalAnalysisGUI(prep_signals, modal_data))

    # StabilGUI wires cmpl_plot's diagram + the first auto-selected mode's
    # scatter plot during __init__ (same as real interactive use), so both
    # are already populated by the time construction returns.
    cmpl_plot = ComplexPlot()
    stabil_plot = StabilPlot(stabil_calc)
    stabil_gui = StabilGUI(stabil_plot, cmpl_plot)

    _run('gui_stabil_diagram', lambda: stabil_gui)

    mode_shape_plot = ModeShapePlot(
        geometry_data=geometry_data, stabil_calc=stabil_calc,
        modal_data=modal_data, prep_signals=prep_signals)
    _run('gui_mode_shape', lambda: ModeShapeGUI(mode_shape_plot))

    _run('gui_multi_setup', lambda: _build_multi_setup_gui(geometry_data))

    return failures


def _source_hash(root, patterns=('*.py', '*.ui')):
    """Stable content hash of every GUI source file.

    This used to compare filesystem mtimes, but `actions/checkout` stamps
    every file with the checkout's wall-clock time instead of preserving
    history, so on a fresh CI checkout the screenshots and the GUI source
    end up with essentially arbitrary mtimes relative to each other -
    making that comparison fail (or pass) at random. Hashing file contents
    instead is independent of checkout time, machine clock, and git history
    depth.
    """
    h = hashlib.sha256()
    for path in sorted(p for pattern in patterns for p in root.rglob(pattern)):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _check() -> int:
    existing = [OUT_DIR / f'{name}.png' for name in SCREENSHOT_NAMES]
    missing = [p.name for p in existing if not p.exists()]
    if missing:
        print(f"Missing screenshots: {', '.join(missing)}\n"
              "Run `python scripts/generate_gui_screenshots.py` and commit the result.",
              file=sys.stderr)
        return 1

    current_hash = _source_hash(GUI_SRC_DIR)
    stored_hash = (
        SOURCE_HASH_FILE.read_text().strip()
        if SOURCE_HASH_FILE.exists() else None)
    if current_hash != stored_hash:
        print("GUI screenshots are older than the GUI source they depict.\n"
              "Run `python scripts/generate_gui_screenshots.py` and commit the result.",
              file=sys.stderr)
        return 1

    print("All GUI screenshots are up to date.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check', action='store_true',
        help="don't regenerate; exit non-zero if screenshots are missing or stale")
    args = parser.parse_args()

    if args.check:
        return _check()

    failures = _capture_all()
    if failures:
        print(f"Failed to capture: {', '.join(failures)}", file=sys.stderr)
        return 1
    SOURCE_HASH_FILE.write_text(_source_hash(GUI_SRC_DIR) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
