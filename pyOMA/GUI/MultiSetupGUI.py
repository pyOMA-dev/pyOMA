# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 GUI orchestrating multi-setup OMA (PoSER and PoGER).

Each measurement setup gets its own tab (:class:`SetupTabWidget`), which
loads its files and pops open the existing single-setup dialogs
(:func:`~pyOMA.GUI.PreProcessSignalsGUI.start_preprocess_gui`,
:func:`~pyOMA.GUI.ModalAnalysisGUI.start_modal_analysis_gui`,
:func:`~pyOMA.GUI.StabilGUI.start_stabil_gui`) rather than re-implementing
those steps - this module only orchestrates, mirroring what
``scripts/multi_setup_analysis.py`` and ``scripts/multi_setup_analysis_poger.py``
already do as a script loop, made interactive.

PoSER (:class:`~pyOMA.core.PostProcessingTools.MergePoSER`) identifies and
pole-selects each setup independently, then merges; every setup tab needs
its own modal-ID + pole-selection pass, so :class:`SetupTabWidget` shows
those rows. PoGER (:class:`~pyOMA.core.SSICovRef.PogerSSICovRef`) pools the
pre-processed setups *before* identification and identifies/pole-selects the
joint data once - :class:`SetupTabWidget` hides its modal-ID/pole rows in
this mode, and :meth:`MultiSetupGUI._merge_poger` runs the single shared
identification + pole-selection pass instead.

Widget layout lives in ``ui/multi_setup.ui`` and ``ui/setup_tab.ui``
(compiled to ``generated/ui_multi_setup.py`` / ``generated/ui_setup_tab.py``
by ``scripts/build_ui.py``); this module only wires signals/slots.
"""
import sys
import logging

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QFileDialog
from PyQt6.QtCore import QEventLoop

from .generated.ui_multi_setup import Ui_MultiSetupGUI
from .generated.ui_setup_tab import Ui_SetupTabWidget
from .HelpersGUI import UnsavedChangesMixin
from .PreProcessSignalsGUI import start_preprocess_gui
from .ModalAnalysisGUI import start_modal_analysis_gui
from .StabilGUI import start_stabil_gui
from .PlotMSHGUI import start_msh_gui
from .GeometryProcessorGUI import start_geometry_processor_gui
from ..core.PreProcessingTools import PreProcessSignals, GeometryProcessor
from ..core.SSICovRef import PogerSSICovRef
from ..core.StabilDiagram import StabilCluster, StabilPlot
from ..core.PostProcessingTools import MergePoSER
from ..core.PlotMSH import ModeShapePlot

logger = logging.getLogger(__name__)

app = None

_CONFIG_FILE_FILTER = "Text files (*.txt);;All files (*)"
_MEAS_FILE_FILTER = "NumPy archive (*.npy *.npz);;All files (*)"
_STATE_FILE_FILTER = "NumPy archive (*.npz);;All files (*)"

_MODES = ['Single Setup', 'PoSER', 'PoGER']


class SetupTabWidget(QWidget, Ui_SetupTabWidget):
    """One tab of :class:`MultiSetupGUI`: loads and pre-processes a single
    measurement setup, and (in PoSER mode) runs its own modal identification
    and pole selection.

    Parameters
    ----------
    mode : str
        ``'PoSER'`` or ``'PoGER'`` - controls whether the modal-ID/pole-
        selection rows are shown.
    geometry_data : GeometryProcessor, optional
        Shared geometry, passed through to per-setup sub-dialogs.
    parent : QWidget, optional
    """

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
        self.btn_browse_config.clicked.connect(self._on_browse_config)
        self.btn_browse_meas.clicked.connect(self._on_browse_meas)
        self.btn_browse_chan_dofs.clicked.connect(self._on_browse_chan_dofs)
        self.btn_load_setup.clicked.connect(self._on_load_setup)
        self.btn_preprocess.clicked.connect(self._on_preprocess)
        self.btn_modal_id.clicked.connect(self._on_modal_id)
        self.btn_select_poles.clicked.connect(self._on_select_poles)

    def set_mode(self, mode):
        """Show/hide the per-setup modal-ID and pole-selection rows.

        PoSER and Single Setup both identify and pole-select this setup
        individually (Single Setup just skips the merge step afterwards);
        PoGER pools the pre-processed setups and identifies/pole-selects
        once on the joint data (handled by ``MultiSetupGUI``, not here), so
        those two rows are irrelevant in PoGER mode.
        """
        self.mode = mode
        is_poser = mode in ('PoSER', 'Single Setup')
        self.btn_modal_id.setVisible(is_poser)
        self.btn_select_poles.setVisible(is_poser)
        self._refresh_status()

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------
    def _on_browse_config(self):
        fname, _filter = QFileDialog.getOpenFileName(
            self, "Select Config File", "", _CONFIG_FILE_FILTER)
        if fname:
            self.edit_config_file.setText(fname)

    def _on_browse_meas(self):
        fname, _filter = QFileDialog.getOpenFileName(
            self, "Select Measurement File", "", _MEAS_FILE_FILTER)
        if fname:
            self.edit_meas_file.setText(fname)

    def _on_browse_chan_dofs(self):
        fname, _filter = QFileDialog.getOpenFileName(
            self, "Select Channel DOFs File", "", _CONFIG_FILE_FILTER)
        if fname:
            self.edit_chan_dofs_file.setText(fname)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _on_load_setup(self):
        conf_file = self.edit_config_file.text()
        meas_file = self.edit_meas_file.text()
        chan_dofs_file = self.edit_chan_dofs_file.text() or None
        if not conf_file or not meas_file:
            QMessageBox.warning(
                self, "Missing files",
                "Both a config file and a measurement file are required.")
            return
        try:
            self.prep_signals = PreProcessSignals.init_from_config(
                conf_file=conf_file, meas_file=meas_file,
                chan_dofs_file=chan_dofs_file)
        except Exception as exc:
            logger.exception("init_from_config failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.modal_data = None
        self.stabil_calc = None
        self._refresh_status()

    def _on_preprocess(self):
        if self.prep_signals is None:
            return
        start_preprocess_gui(self.prep_signals, self.geometry_data)
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
        self.btn_preprocess.setEnabled(self.prep_signals is not None)
        self.btn_modal_id.setEnabled(self.prep_signals is not None)
        self.btn_select_poles.setEnabled(self.modal_data is not None)

        if self.prep_signals is None:
            self.lbl_status.setText("Not loaded.")
            return
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

    # ------------------------------------------------------------------
    # Readiness checks used by MultiSetupGUI
    # ------------------------------------------------------------------
    @property
    def is_ready_poser(self):
        return self.stabil_calc is not None and bool(self.stabil_calc.select_modes)

    @property
    def is_ready_poger(self):
        return self.prep_signals is not None


class MultiSetupGUI(UnsavedChangesMixin, QMainWindow, Ui_MultiSetupGUI):
    """Interactive GUI orchestrating multi-setup OMA merging via PoSER or PoGER.

    Setups are added as tabs; "Merge Setups" then combines them via
    :class:`~pyOMA.core.PostProcessingTools.MergePoSER` (PoSER) or
    :class:`~pyOMA.core.SSICovRef.PogerSSICovRef` (PoGER), mirroring
    ``scripts/multi_setup_analysis.py`` / ``scripts/multi_setup_analysis_poger.py``
    interactively.

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
        self._poger_stabil_calc = None
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
        self.stack_settings.setCurrentIndex(1 if mode == 'PoGER' else 0)
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
        if self.mode == 'PoSER':
            ready = all(tab.is_ready_poser for tab in tabs)
            hint = "poles selected" if ready else "select poles on every setup first"
        else:
            ready = all(tab.is_ready_poger for tab in tabs)
            hint = "setups loaded" if ready else "load every setup first"
        self.btn_merge.setEnabled(ready and len(tabs) >= 2)
        if len(tabs) < 2:
            self.lbl_merge_status.setText("Add at least two setups to merge.")
        else:
            self.lbl_merge_status.setText(f"{len(tabs)} setup(s), {hint}.")

    def _on_merge(self):
        try:
            if self.mode == 'Single Setup':
                self._single_setup_tab = self._tabs[0]
                self.merged_data = None  # nothing to merge - the tab's own results are shown
            elif self.mode == 'PoSER':
                self._merge_poser()
            else:
                self._merge_poger()
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
        self._poger_stabil_calc = stabil_calc

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
            mode_shape_plot = ModeShapePlot(
                geometry_data=self.geometry_data,
                stabil_calc=tab.stabil_calc,
                modal_data=tab.modal_data,
                prep_signals=tab.prep_signals)
        elif self.merged_data is None:
            return
        elif isinstance(self.merged_data, MergePoSER):
            mode_shape_plot = ModeShapePlot(
                geometry_data=self.geometry_data, merged_data=self.merged_data)
        else:
            mode_shape_plot = ModeShapePlot(
                geometry_data=self.geometry_data,
                stabil_calc=self._poger_stabil_calc,
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
        cls = MergePoSER if self.mode == 'PoSER' else PogerSSICovRef
        try:
            loaded = cls.load_state(fname)
        except Exception as exc:
            logger.exception("load_state failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.merged_data = loaded
        self._poger_stabil_calc = None
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
    app = QApplication.instance() or QApplication(sys.argv)

    form = MultiSetupGUI(geometry_data)
    form.resize(760, 860)

    loop = QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec()
    return form._merged_data_at_close


def main():
    pass


if __name__ == '__main__':
    main()
