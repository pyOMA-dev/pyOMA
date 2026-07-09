# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 GUI hosting one widget per (single-setup) OMA method.

Wraps :class:`~pyOMA.GUI.SSIDataGUI.SSIDataWidget`,
:class:`~pyOMA.GUI.SSICovRefGUI.BRSSICovRefWidget`,
:class:`~pyOMA.GUI.VarSSIRefGUI.VarSSIRefWidget`,
:class:`~pyOMA.GUI.PLSCFGUI.PLSCFWidget` and
:class:`~pyOMA.GUI.PRCEGUI.PRCEWidget` behind a single method-selector combo
box and :class:`QStackedWidget`. ``PogerSSICovRef`` (multi-setup PoGER) and
``ERA`` have no widget here and are out of scope for this GUI.

Each page widget is constructed once (eagerly) against the same
``prep_signals``, and operates on its own ``instance`` in place - switching
pages never discards another page's progress. ``modal_data`` is a read-only
property returning the currently active page's ``instance``, matching how
``scripts/single_setup_analysis.py`` already produces ``modal_data`` via
``METHOD.init_from_config(...)``: this GUI is opened *after* that call, to
inspect/re-run steps on the same object (mutated in place, so the caller's
reference stays valid without needing to consume a return value) - it does
not replace ``init_from_config`` as the primary way ``modal_data`` is
produced.

Widget layout lives in ``ui/modal_analysis.ui`` (compiled to
``generated/ui_modal_analysis.py`` by ``scripts/build_ui.py``); this module
only wires signals/slots and hosts the five method widgets.
"""
import sys
import logging

from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt6.QtCore import QEventLoop

from .generated.ui_modal_analysis import Ui_ModalAnalysisGUI
from .SSIDataGUI import SSIDataWidget
from .SSICovRefGUI import BRSSICovRefWidget
from .VarSSIRefGUI import VarSSIRefWidget
from .PLSCFGUI import PLSCFWidget
from .PRCEGUI import PRCEWidget
from ..core.SSIData import SSIDataMC
from ..core.SSICovRef import BRSSICovRef
from ..core.VarSSIRef import VarSSIRef
from ..core.PLSCF import PLSCF
from ..core.PRCE import PRCE
from ..core.PreProcessingTools import PreProcessSignals

logger = logging.getLogger(__name__)

app = None

_FILE_FILTER = "NumPy archive (*.npz);;All files (*)"
_CONFIG_FILE_FILTER = "Text files (*.txt);;All files (*)"

# (combo label, page widget class, class used to recognise a matching
# modal_data instance passed in at construction time).
_METHODS = [
    ('SSI-Data', SSIDataWidget, SSIDataMC),
    ('SSI-Cov-Ref', BRSSICovRefWidget, BRSSICovRef),
    ('Var-SSI-Ref', VarSSIRefWidget, VarSSIRef),
    ('pLSCF', PLSCFWidget, PLSCF),
    ('PRCE', PRCEWidget, PRCE),
]


class ModalAnalysisGUI(QMainWindow, Ui_ModalAnalysisGUI):
    """Interactive GUI hosting a widget per single-setup OMA method.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object shared by all method widgets.
    modal_data : ModalBase subclass instance, optional
        An already-computed (or partially computed) instance to inspect,
        e.g. the result of ``METHOD.init_from_config(...)``. The matching
        method page is selected and adopts it; other pages start fresh.
    parent : QWidget, optional
    """

    def __init__(self, prep_signals, modal_data=None, parent=None):
        super().__init__(parent)
        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be a PreProcessSignals instance, "
                f"got {type(prep_signals).__name__}")
        self.prep_signals = prep_signals

        self.setupUi(self)
        self.lbl_setup_name.setText(prep_signals.setup_name or '-')

        self._pages = [cls(prep_signals) for _label, cls, _match_cls in _METHODS]
        for page in self._pages:
            self.stacked_widget.addWidget(page)
        for label, _cls, _match_cls in _METHODS:
            self.combo_method.addItem(label)

        self._wire_buttons()
        self._select_page_for(modal_data)
        self.show()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.combo_method.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)
        self.btn_save.clicked.connect(self.close)
        self.btn_load.clicked.connect(self._on_ok_close)
        self.actionSave_State.triggered.connect(self._on_save)
        self.actionLoad_State.triggered.connect(self._on_load)
        self.actionQuit.triggered.connect(self.close)
        self.actionLoad_Config.triggered.connect(self._on_load_config)
        self.actionSave_Config.triggered.connect(self._on_save_config)

    def _on_ok_close(self):
        """'OK and Close': refuse to close if the current page hasn't
        computed anything yet (nothing meaningful to accept)."""
        if self.modal_data.modal_frequencies is None:
            QMessageBox.warning(
                self, "Not computed",
                "The current method has not been run yet, so there is "
                "nothing to accept. Run it first, or use Cancel to close "
                "without a result.")
            return
        self.close()

    def _select_page_for(self, modal_data):
        if modal_data is None:
            self.combo_method.setCurrentIndex(0)
            return
        for index, (_label, _cls, match_cls) in enumerate(_METHODS):
            if isinstance(modal_data, match_cls):
                self._pages[index].set_instance(modal_data)
                self.combo_method.setCurrentIndex(index)
                return
        logger.warning(
            "No modal-analysis widget matches %s; showing the first page instead.",
            type(modal_data).__name__)
        self.combo_method.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # modal_data accessor
    # ------------------------------------------------------------------
    @property
    def modal_data(self):
        """The currently active page's ``instance`` (mutated in place by its widget)."""
        return self.stacked_widget.currentWidget().instance

    # ------------------------------------------------------------------
    # Save / Load (generic across all pages)
    # ------------------------------------------------------------------
    def _on_save(self):
        fname, _filter = QFileDialog.getSaveFileName(
            self, "Save Modal Analysis State", "", _FILE_FILTER)
        if not fname:
            return
        try:
            self.modal_data.save_state(fname)
        except Exception as exc:
            logger.exception("save_state failed")
            QMessageBox.warning(self, "Save failed", str(exc))

    def _on_load(self):
        fname, _filter = QFileDialog.getOpenFileName(
            self, "Load Modal Analysis State", "", _FILE_FILTER)
        if not fname:
            return
        page = self.stacked_widget.currentWidget()
        # Load via the page's *currently selected* concrete class (matters
        # for SSIDataWidget, whose combo picks SSIData/SSIDataMC/SSIDataCV) -
        # if the saved file was a different variant, pick the matching
        # variant in that page's own combo first, then load.
        cls = type(page.instance)
        try:
            loaded = cls.load_state(fname, self.prep_signals)
        except Exception as exc:
            logger.exception("load_state failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        page.set_instance(loaded)

    def _on_load_config(self):
        fname, _filter = QFileDialog.getOpenFileName(
            self, "Load Modal Analysis Config", "", _CONFIG_FILE_FILTER)
        if not fname:
            return
        page = self.stacked_widget.currentWidget()
        cls = type(page.instance)
        try:
            loaded = cls.init_from_config(fname, self.prep_signals)
        except Exception as exc:
            logger.exception("init_from_config failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        page.set_instance(loaded)

    def _on_save_config(self):
        fname, _filter = QFileDialog.getSaveFileName(
            self, "Save Modal Analysis Config", "", _CONFIG_FILE_FILTER)
        if not fname:
            return
        try:
            self.modal_data.write_config(fname)
        except Exception as exc:
            logger.exception("write_config failed")
            QMessageBox.warning(self, "Save failed", str(exc))

    def closeEvent(self, *args, **kwargs):
        # Snapshot into a plain attribute *before* deleteLater(): once the
        # underlying Qt object is destroyed, self.modal_data (which reaches
        # into self.stacked_widget) is no longer safe to evaluate, but a
        # plain Python attribute stays readable for as long as something
        # holds a reference to this (Python-side) object.
        self._modal_data_at_close = self.modal_data
        self.deleteLater()
        return QMainWindow.closeEvent(self, *args, **kwargs)


def start_modal_analysis_gui(prep_signals, modal_data=None):
    global app
    app = QApplication.instance() or QApplication(sys.argv)

    form = ModalAnalysisGUI(prep_signals, modal_data)
    form.resize(700, 800)

    loop = QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec()
    return form._modal_data_at_close


def main():
    pass


if __name__ == '__main__':
    main()
