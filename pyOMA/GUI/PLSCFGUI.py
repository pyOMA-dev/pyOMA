# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 widget for pyOMA.core.PLSCF.PLSCF.

Wraps a :class:`~pyOMA.core.PLSCF.PLSCF` instance: each button runs one step
of its ``build_half_spectra`` -> ``compute_modal_params`` sequence (see
``PLSCF.init_from_config`` for the canonical order) directly on the
instance, so the object handed in (or created on first use) is mutated in
place.

Widget layout lives in ``ui/plscf.ui`` (compiled to ``generated/ui_plscf.py``
by ``scripts/build_ui.py``); this module only wires signals/slots and the
build/compute steps.
"""
import logging

from PyQt6.QtWidgets import QWidget, QMessageBox

from .generated.ui_plscf import Ui_PLSCFWidget
from ..core.PLSCF import PLSCF
from ..core.PreProcessingTools import PreProcessSignals

logger = logging.getLogger(__name__)

_MODAL_CONTRIB_VALUES = {'Auto': None, 'On': True, 'Off': False}


class PLSCFWidget(QWidget, Ui_PLSCFWidget):
    """Interactive widget for the pLSCF (Poly-reference LSCF / PolyMAX) method.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object used to construct a new ``PLSCF``
        instance, if *instance* is not given.
    instance : PLSCF, optional
        An existing (possibly partially or fully computed) ``PLSCF`` object
        to inspect/continue. Defaults to a fresh, unbuilt instance.
    parent : QWidget, optional
    """

    def __init__(self, prep_signals, instance=None, parent=None):
        super().__init__(parent)
        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be a PreProcessSignals instance, "
                f"got {type(prep_signals).__name__}")
        self.prep_signals = prep_signals

        self.setupUi(self)
        self._wire_buttons()

        self.set_instance(instance if instance is not None else PLSCF(prep_signals))

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.btn_build_half_spectra.clicked.connect(self._on_build_half_spectra)
        self.btn_compute_modal_params.clicked.connect(self._on_compute_modal_params)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    def set_instance(self, instance):
        """Adopt *instance* as the object this widget operates on and refresh
        every field/button from its current state."""
        if not isinstance(instance, PLSCF):
            raise TypeError(
                f"instance must be a PLSCF instance, got {type(instance).__name__}")
        self.instance = instance

        if instance.nperseg is not None:
            self.spin_nperseg.setValue(instance.nperseg)
        if instance.begin_frequency is not None:
            self.spin_begin_frequency.setValue(instance.begin_frequency)
        if instance.end_frequency is not None:
            self.spin_end_frequency.setValue(instance.end_frequency)
        if instance.max_model_order is not None:
            self.spin_max_model_order.setValue(instance.max_model_order)

        built = instance.state[0]
        self.btn_compute_modal_params.setEnabled(built)
        self.lbl_num_omega.setText(str(instance.num_omega) if built else '-')
        self._refresh_status()

    def _refresh_status(self):
        built, computed = self.instance.state
        if computed:
            text = ("Modal parameters computed up to order "
                    f"{self.instance.max_model_order}.")
        elif built:
            text = "Half-spectra built. Ready to compute modal parameters."
        else:
            text = "Not built yet."
        self.lbl_status.setText(text)

    # ------------------------------------------------------------------
    # Step 1: build_half_spectra
    # ------------------------------------------------------------------
    def _on_build_half_spectra(self):
        nperseg = self.spin_nperseg.value() or None
        begin_frequency = self.spin_begin_frequency.value()
        begin_frequency = None if begin_frequency < 0 else begin_frequency
        end_frequency = self.spin_end_frequency.value()
        end_frequency = None if end_frequency < 0 else end_frequency
        window_decay = self.spin_window_decay.value()
        try:
            self.instance.build_half_spectra(
                nperseg, begin_frequency, end_frequency, window_decay=window_decay)
        except Exception as exc:
            logger.exception("build_half_spectra failed")
            QMessageBox.warning(self, "Build Half-Spectra failed", str(exc))
            return
        self.set_instance(self.instance)

    # ------------------------------------------------------------------
    # Step 2: compute_modal_params
    # ------------------------------------------------------------------
    def _on_compute_modal_params(self):
        max_model_order = self.spin_max_model_order.value()
        complex_coefficients = self.chk_complex_coefficients.isChecked()
        algo = self.combo_algo.currentText()
        modal_contrib = _MODAL_CONTRIB_VALUES[self.combo_modal_contrib.currentText()]
        try:
            self.instance.compute_modal_params(
                max_model_order, complex_coefficients=complex_coefficients,
                algo=algo, modal_contrib=modal_contrib)
        except Exception as exc:
            logger.exception("compute_modal_params failed")
            QMessageBox.warning(self, "Compute Modal Parameters failed", str(exc))
            return
        self.set_instance(self.instance)
