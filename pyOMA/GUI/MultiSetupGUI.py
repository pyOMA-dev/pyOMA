# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 GUI orchestrating multi-setup OMA (PoSER, PoGER, PreGER).

Each measurement setup gets its own tab (:class:`SetupTabWidget`), which
pops open the existing single-setup dialogs
(:func:`~pyOMA.GUI.PreProcessSignalsGUI.start_preprocess_gui`,
:func:`~pyOMA.GUI.ModalAnalysisGUI.start_modal_analysis_gui`,
:func:`~pyOMA.GUI.StabilGUI.start_stabil_gui`) rather than re-implementing
those steps - this module only orchestrates, mirroring what
``scripts/multi_setup_analysis.py``, ``scripts/multi_setup_analysis_poger.py``
and ``scripts/multi_setup_analysis_preger.py`` already do as a script loop,
made interactive.

PoSER (:class:`~pyOMA.core.PostProcessingTools.MergePoSER`) identifies and
pole-selects each setup independently, then merges; every setup tab needs
its own modal-ID + pole-selection pass, so :class:`SetupTabWidget` shows
those rows. PoGER (:class:`~pyOMA.core.SSICovRef.PogerSSICovRef`) and PreGER
(:class:`~pyOMA.core.MultiSetupSSI.PreGERSSI`, plus its variance-propagating
subclass :class:`~pyOMA.core.MultiSetupSSI.VarPreGERSSI` selected via the
"Compute variances" checkbox) both pool the pre-processed setups *before*
identification and identify/pole-select the joint data once -
:class:`SetupTabWidget` hides its modal-ID/pole rows in these modes, and
:meth:`MultiSetupGUI._merge_poger`/:meth:`MultiSetupGUI._merge_preger` run the
single shared identification + pole-selection pass instead.

Widget layout lives in ``ui/multi_setup.ui`` and ``ui/setup_tab.ui``
(compiled to ``generated/ui_multi_setup.py`` / ``generated/ui_setup_tab.py``
by ``scripts/build_ui.py``); this module only wires signals/slots.
"""
import sys
import logging

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QFileDialog
from PyQt6.QtCore import QEventLoop, pyqtSignal

from .generated.ui_multi_setup import Ui_MultiSetupGUI
from .generated.ui_setup_tab import Ui_SetupTabWidget
from .HelpersGUI import UnsavedChangesMixin
from .PreProcessSignalsGUI import start_preprocess_gui
from .ModalAnalysisGUI import start_modal_analysis_gui
from .StabilGUI import start_stabil_gui
from .PlotMSHGUI import start_msh_gui
from .GeometryProcessorGUI import start_geometry_processor_gui
from ..core.PreProcessingTools import GeometryProcessor
from ..core.SSICovRef import PogerSSICovRef
from ..core.MultiSetupSSI import PreGERSSI, VarPreGERSSI
from ..core.StabilDiagram import StabilCluster, StabilPlot
from ..core.PostProcessingTools import MergePoSER
from ..core import resolve_mode_shape_backend

logger = logging.getLogger(__name__)

app = None

_STATE_FILE_FILTER = "NumPy archive (*.npz);;All files (*)"

_MODES = ['Single Setup', 'PoSER', 'PoGER', 'PreGER', 'Var-PreGER']

# mode -> class used by _on_load to pick the right load_state(); PoSER/Single
# Setup are handled separately in _on_load (MergePoSER / no load support).
_MERGE_CLASSES = {
    'PoGER': PogerSSICovRef,
    'PreGER': PreGERSSI,
    'Var-PreGER': VarPreGERSSI,
}


class SetupTabWidget(QWidget, Ui_SetupTabWidget):
    """One tab of :class:`MultiSetupGUI`: runs the pre-processing, and (in
    PoSER mode) modal identification and pole selection, of a single
    measurement setup. Loading itself happens inside the pre-processing
    window (``File -> Load Config...`` / ``Import Signals...``), opened via
    "Pre-process Signals...".

    Parameters
    ----------
    mode : str
        One of :data:`_MODES` - controls whether the modal-ID/pole-selection
        rows are shown (``'PoSER'``/``'Single Setup'`` only; every pooled mode
        - ``'PoGER'``, ``'PreGER'``, ``'Var-PreGER'`` - hides them).
    geometry_data : GeometryProcessor, optional
        Shared geometry, passed through to per-setup sub-dialogs.
    parent : QWidget, optional
    """

    #: Emitted whenever this tab's lifecycle status changes (loaded,
    #: identified, poles selected, ...) - MultiSetupGUI connects this to
    #: its own _refresh_merge_status() so the "Merge Setups"/"Continue"
    #: button reflects per-tab progress without needing a mode switch to
    #: force a refresh.
    status_changed = pyqtSignal()

    def __init__(self, mode, geometry_data=None, parent=None):
        super().__init__(parent)
        self.geometry_data = geometry_data
        self.prep_signals = None
        self.modal_data = None
        self.stabil_calc = None

        self.setupUi(self)
        self._wire_buttons()
        self.set_mode(mode)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.btn_preprocess.clicked.connect(self._on_preprocess)
        self.btn_modal_id.clicked.connect(self._on_modal_id)
        self.btn_select_poles.clicked.connect(self._on_select_poles)

    def set_mode(self, mode):
        """Show/hide the per-setup modal-ID and pole-selection rows.

        PoSER and Single Setup both identify and pole-select this setup
        individually (Single Setup just skips the merge step afterwards);
        PoGER/PreGER/Var-PreGER pool the pre-processed setups and
        identify/pole-select once on the joint data (handled by
        ``MultiSetupGUI``, not here), so those two rows are irrelevant in
        every pooled mode.
        """
        self.mode = mode
        is_poser = mode in ('PoSER', 'Single Setup')
        self.btn_modal_id.setVisible(is_poser)
        self.btn_select_poles.setVisible(is_poser)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _on_preprocess(self):
        """Open PreProcessSignalsGUI - the sole place setups are now loaded
        (via its File -> Load Config.../Import Signals... actions) or
        further edited. Whatever it reports back on close becomes this
        tab's prep_signals; downstream modal data/pole selection is only
        reset if that's a genuinely new object (a fresh load), not just the
        same signals mutated further."""
        prep_signals = start_preprocess_gui(self.prep_signals, self.geometry_data)
        if prep_signals is not None and prep_signals is not self.prep_signals:
            self.prep_signals = prep_signals
            self.modal_data = None
            self.stabil_calc = None
        self._refresh_status()

    def _on_modal_id(self):
        if self.prep_signals is None:
            return
        self.modal_data = start_modal_analysis_gui(self.prep_signals, self.modal_data)
        self.stabil_calc = None
        self._refresh_status()

    def _on_select_poles(self):
        if self.modal_data is None:
            QMessageBox.warning(
                self, "No modal data", "Run modal analysis before selecting poles.")
            return
        if self.stabil_calc is None:
            self.stabil_calc = StabilCluster(self.modal_data, self.prep_signals)
            self.stabil_calc.calculate_stabilization_masks()
        stabil_plot = StabilPlot(self.stabil_calc)
        start_stabil_gui(stabil_plot, self.modal_data, self.geometry_data, self.prep_signals)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _refresh_status(self):
        self.btn_modal_id.setEnabled(self.prep_signals is not None)
        self.btn_select_poles.setEnabled(self.modal_data is not None)

        if self.prep_signals is None:
            self.lbl_status.setText("Not loaded.")
        else:
            name = self.prep_signals.setup_name or '-'
            if self.mode in ('PoSER', 'Single Setup'):
                if self.stabil_calc is not None and self.stabil_calc.select_modes:
                    text = f"{name}: {len(self.stabil_calc.select_modes)} mode(s) selected."
                elif self.modal_data is not None:
                    text = f"{name}: modal analysis done, poles not selected yet."
                else:
                    text = f"{name}: loaded, not identified yet."
            else:
                text = f"{name}: loaded."
            self.lbl_status.setText(text)
        self.status_changed.emit()

    # ------------------------------------------------------------------
    # Readiness checks used by MultiSetupGUI
    # ------------------------------------------------------------------
    @property
    def is_ready_poser(self):
        return self.stabil_calc is not None and bool(self.stabil_calc.select_modes)

    @property
    def is_ready_pooled(self):
        """Readiness for any pooled mode (PoGER, PreGER, Var-PreGER): just
        needs pre-processed signals, since identification runs once, jointly,
        after merging - see :meth:`MultiSetupGUI._merge_poger`/
        :meth:`MultiSetupGUI._merge_preger`."""
        return self.prep_signals is not None


class MultiSetupGUI(UnsavedChangesMixin, QMainWindow, Ui_MultiSetupGUI):
    """Interactive GUI orchestrating multi-setup OMA merging via PoSER, PoGER or PreGER.

    Setups are added as tabs; "Merge Setups" then combines them via
    :class:`~pyOMA.core.PostProcessingTools.MergePoSER` (PoSER),
    :class:`~pyOMA.core.SSICovRef.PogerSSICovRef` (PoGER) or
    :class:`~pyOMA.core.MultiSetupSSI.PreGERSSI`/
    :class:`~pyOMA.core.MultiSetupSSI.VarPreGERSSI` (PreGER/Var-PreGER),
    mirroring ``scripts/multi_setup_analysis.py`` /
    ``scripts/multi_setup_analysis_poger.py`` /
    ``scripts/multi_setup_analysis_preger.py`` interactively.

    Parameters
    ----------
    geometry_data : GeometryProcessor, optional
        Shared geometry object. Created (and mutated in place) via the
        geometry-loading dialog if not given.
    parent : QWidget, optional
    """

    def __init__(self, geometry_data=None, parent=None):
        super().__init__(parent)
        self.geometry_data = geometry_data
        self.merged_data = None
        self._pooled_stabil_calc = None
        self._single_setup_tab = None
        self._dirty = False

        self.setupUi(self)
        self.combo_mode.addItems(_MODES)
        self._wire_buttons()
        self._refresh_geometry_status()
        self._on_mode_changed(0)
        self.show()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_load_geometry.clicked.connect(self._on_load_geometry)
        self.btn_add_setup.clicked.connect(self._on_add_setup)
        self.btn_remove_setup.clicked.connect(self._on_remove_setup)
        self.btn_merge.clicked.connect(self._on_merge)
        self.btn_show_mode_shapes.clicked.connect(self._on_show_mode_shapes)
        self.actionSave_State.triggered.connect(self._on_save)
        self.actionLoad_State.triggered.connect(self._on_load)
        self.actionQuit.triggered.connect(self.close)

    @property
    def mode(self):
        return self.combo_mode.currentText()

    @property
    def _tabs(self):
        return [self.tabs_setups.widget(i) for i in range(self.tabs_setups.count())]

    # ------------------------------------------------------------------
    # Mode switch
    # ------------------------------------------------------------------
    def _on_mode_changed(self, _index):
        mode = self.mode
        if mode == 'PoGER':
            settings_page = 1
        elif mode in ('PreGER', 'Var-PreGER'):
            settings_page = 2
        else:
            settings_page = 0
        self.stack_settings.setCurrentIndex(settings_page)
        self.btn_merge.setText("Continue" if mode == 'Single Setup' else "Merge Setups")
        for tab in self._tabs:
            tab.set_mode(mode)
        self.merged_data = None
        self._single_setup_tab = None
        self._dirty = False
        self.btn_show_mode_shapes.setEnabled(False)
        self._refresh_add_setup_enabled()
        self._refresh_merge_status()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _on_load_geometry(self):
        if self.geometry_data is None:
            self.geometry_data = GeometryProcessor()
        start_geometry_processor_gui(self.geometry_data)
        for tab in self._tabs:
            tab.geometry_data = self.geometry_data
        self._refresh_geometry_status()

    def _refresh_geometry_status(self):
        if self.geometry_data is None:
            self.lbl_geometry_status.setText("Geometry: not loaded")
        else:
            self.lbl_geometry_status.setText(
                f"Geometry: {len(self.geometry_data.nodes)} node(s)")

    # ------------------------------------------------------------------
    # Setup tabs
    # ------------------------------------------------------------------
    def _on_add_setup(self):
        tab = SetupTabWidget(self.mode, self.geometry_data)
        tab.status_changed.connect(self._refresh_merge_status)
        index = self.tabs_setups.addTab(tab, f"Setup {self.tabs_setups.count() + 1}")
        self.tabs_setups.setCurrentIndex(index)
        self._refresh_add_setup_enabled()
        self._refresh_merge_status()

    def _on_remove_setup(self):
        index = self.tabs_setups.currentIndex()
        if index < 0:
            return
        answer = QMessageBox.question(
            self, "Remove Setup",
            "Remove the current setup tab? Unsaved progress on this tab will be lost.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        widget = self.tabs_setups.widget(index)
        self.tabs_setups.removeTab(index)
        widget.deleteLater()
        self._refresh_add_setup_enabled()
        self._refresh_merge_status()

    def _refresh_add_setup_enabled(self):
        # Single Setup mode is capped at one tab - there is nothing to merge.
        self.btn_add_setup.setEnabled(not (self.mode == 'Single Setup' and self._tabs))

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _refresh_merge_status(self):
        tabs = self._tabs
        if not tabs:
            self.btn_merge.setEnabled(False)
            self.lbl_merge_status.setText("Add at least one setup.")
            return
        if self.mode == 'Single Setup':
            ready = tabs[0].is_ready_poser
            hint = "poles selected" if ready else "select poles first"
            self.btn_merge.setEnabled(ready)
            self.lbl_merge_status.setText(f"1 setup, {hint}.")
            return
        ready, hint = self._merge_readiness(tabs)
        self.btn_merge.setEnabled(ready and len(tabs) >= 2)
        if len(tabs) < 2:
            self.lbl_merge_status.setText("Add at least two setups to merge.")
        else:
            self.lbl_merge_status.setText(f"{len(tabs)} setup(s), {hint}.")

    def _merge_readiness(self, tabs):
        '''Return ``(ready, hint)`` for the multi-setup modes.

        PoSER merges selected poles, so every setup must have a selection;
        the pooled modes (PoGER/PreGER/Var-PreGER) only need the setups loaded.
        '''
        if self.mode == 'PoSER':
            ready = all(tab.is_ready_poser for tab in tabs)
            return ready, "poles selected" if ready else "select poles on every setup first"
        ready = all(tab.is_ready_pooled for tab in tabs)
        return ready, "setups loaded" if ready else "load every setup first"

    def _on_merge(self):
        try:
            if self.mode == 'Single Setup':
                self._single_setup_tab = self._tabs[0]
                self.merged_data = None  # nothing to merge - the tab's own results are shown
            elif self.mode == 'PoSER':
                self._merge_poser()
            elif self.mode == 'PoGER':
                self._merge_poger()
            else:
                self._merge_preger(use_variance=(self.mode == 'Var-PreGER'))
        except Exception as exc:
            logger.exception("Merge failed")
            QMessageBox.warning(self, "Merge failed", str(exc))
            return
        self._dirty = True
        self.btn_show_mode_shapes.setEnabled(True)
        self.lbl_merge_status.setText(
            "Ready." if self.mode == 'Single Setup' else "Merged successfully.")

    def _merge_poser(self):
        merger = MergePoSER()
        for tab in self._tabs:
            merger.add_setup(tab.prep_signals, tab.modal_data, tab.stabil_calc)
        merger.merge()
        self.merged_data = merger

    def _merge_poger(self):
        poger = PogerSSICovRef()
        for tab in self._tabs:
            if tab.prep_signals.m_lags is None:
                name = tab.prep_signals.setup_name or 'A setup'
                raise ValueError(
                    f"{name} has no correlation function computed yet. Open "
                    "\"Pre-process Signals...\" for that setup and compute "
                    "correlation (e.g. via the Correlation time-domain diagram) "
                    "before merging with PoGER.")
            poger.add_setup(tab.prep_signals)
        poger.pair_channels()

        num_block_columns = self.spin_num_block_columns.value() or None
        max_model_order = self.spin_max_model_order.value() or None
        poger.build_merged_subspace_matrix(num_block_columns)
        poger.compute_modal_params(max_model_order)

        stabil_calc = StabilCluster(poger, poger.prep_signals)
        stabil_calc.calculate_stabilization_masks()
        stabil_plot = StabilPlot(stabil_calc)
        start_stabil_gui(stabil_plot, poger, self.geometry_data, poger.prep_signals)

        self.merged_data = poger
        self._pooled_stabil_calc = stabil_calc

    @staticmethod
    def _check_preger_prerequisites(prep_signals, use_variance):
        '''Raise ValueError if *prep_signals* lacks what a PreGER merge needs.

        PreGER consumes the correlation function of each setup; Var-PreGER
        additionally needs it block-wise, to estimate its covariance.
        '''
        name = prep_signals.setup_name or 'A setup'
        if prep_signals.m_lags is None:
            raise ValueError(
                f"{name} has no correlation function computed yet. Open "
                "\"Pre-process Signals...\" for that setup and compute "
                "correlation (e.g. via the Correlation time-domain diagram) "
                "before merging with PreGER.")
        if use_variance and (prep_signals.n_segments is None
                             or prep_signals.n_segments < 2):
            raise ValueError(
                f"{name} has no block-wise correlation function computed yet. "
                "Recompute correlation with n_segments >= 2 (e.g. "
                "corr_blackman_tukey(m_lags, n_segments=...)) before merging "
                "with Var-PreGER.")

    def _merge_preger(self, use_variance):
        """PreGER (or, with *use_variance*, Var-PreGER) merge - mirrors
        :meth:`_merge_poger`, but through the classes' own ``add_setup``
        (Ns) -> ``pair_channels`` -> ``build_subspace_matrices`` ->
        ``compute_modal_params`` sequence rather than PoGER's
        ``build_merged_subspace_matrix``.
        """
        cls = VarPreGERSSI if use_variance else PreGERSSI
        preger = cls()
        for tab in self._tabs:
            self._check_preger_prerequisites(tab.prep_signals, use_variance)
            preger.add_setup(tab.prep_signals)
        preger.pair_channels()

        num_block_columns = self.spin_preger_num_block_columns.value() or None
        num_block_rows = self.spin_preger_num_block_rows.value() or None
        subspace_method = self.combo_preger_subspace_method.currentText()
        num_blocks = self.spin_preger_num_blocks.value() or None
        max_model_order = self.spin_preger_max_model_order.value() or None
        preger.build_subspace_matrices(
            num_block_columns, num_block_rows=num_block_rows,
            subspace_method=subspace_method, num_blocks=num_blocks)
        preger.compute_modal_params(max_model_order)

        stabil_calc = StabilCluster(preger, preger.prep_signals)
        stabil_calc.calculate_stabilization_masks()
        stabil_plot = StabilPlot(stabil_calc)
        start_stabil_gui(stabil_plot, preger, self.geometry_data, preger.prep_signals)

        self.merged_data = preger
        self._pooled_stabil_calc = stabil_calc

    # ------------------------------------------------------------------
    # Mode shapes
    # ------------------------------------------------------------------
    def _on_show_mode_shapes(self):
        if self.geometry_data is None:
            return
        if self.mode == 'Single Setup':
            if self._single_setup_tab is None:
                return
            tab = self._single_setup_tab
            mode_shape_plot = resolve_mode_shape_backend()(
                geometry_data=self.geometry_data,
                stabil_calc=tab.stabil_calc,
                modal_data=tab.modal_data,
                prep_signals=tab.prep_signals)
        elif self.merged_data is None:
            return
        elif isinstance(self.merged_data, MergePoSER):
            mode_shape_plot = resolve_mode_shape_backend()(
                geometry_data=self.geometry_data, merged_data=self.merged_data)
        else:
            mode_shape_plot = resolve_mode_shape_backend()(
                geometry_data=self.geometry_data,
                stabil_calc=self._pooled_stabil_calc,
                modal_data=self.merged_data,
                prep_signals=self.merged_data.prep_signals)
        start_msh_gui(mode_shape_plot)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def _on_save(self):
        if self.merged_data is None:
            QMessageBox.warning(self, "Nothing to save", "Merge setups first.")
            return
        fname, _filter = QFileDialog.getSaveFileName(
            self, "Save Merged State", "", _STATE_FILE_FILTER)
        if not fname:
            return
        try:
            self.merged_data.save_state(fname)
        except Exception as exc:
            logger.exception("save_state failed")
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._dirty = False

    def _do_save(self):
        self._on_save()

    def _on_load(self):
        fname, _filter = QFileDialog.getOpenFileName(
            self, "Load Merged State", "", _STATE_FILE_FILTER)
        if not fname:
            return
        cls = MergePoSER if self.mode == 'PoSER' else _MERGE_CLASSES.get(self.mode, PogerSSICovRef)
        try:
            loaded = cls.load_state(fname)
        except Exception as exc:
            logger.exception("load_state failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.merged_data = loaded
        self._pooled_stabil_calc = None
        self._dirty = False  # freshly loaded from disk - nothing unsaved yet
        self.btn_show_mode_shapes.setEnabled(True)
        self.lbl_merge_status.setText("Loaded merged state.")

    def closeEvent(self, event):
        if not self._prompt_save_on_close(event):
            return

        # Snapshot into a plain attribute *before* deleteLater(): once the
        # underlying Qt object is destroyed, reaching into child widgets is
        # no longer safe, but a plain Python attribute stays readable.
        self._merged_data_at_close = self.merged_data
        self.deleteLater()
        return QMainWindow.closeEvent(self, event)


def start_multi_setup_gui(geometry_data=None):
    global app
    # The Wayland->xcb guard the pyvista backend needs runs once on import of
    # pyOMA.GUI (see pyOMA/GUI/__init__.py), before any QApplication here.
    app = QApplication.instance() or QApplication(sys.argv)

    form = MultiSetupGUI(geometry_data)
    form.resize(760, 860)

    loop = QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec()
    return form._merged_data_at_close


def main():
    start_multi_setup_gui()


if __name__ == '__main__':
    main()
