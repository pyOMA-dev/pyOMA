# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2026  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
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
from PyQt6.QtWidgets import QWidget

from .generated.ui_plscf import Ui_PLSCFWidget
from .HelpersGUI import EstimatorWidgetMixin, _parse_int_list
from ..core.PLSCF import PLSCF

_MODAL_CONTRIB_VALUES = {'Auto': None, 'On': True, 'Off': False}


class PLSCFWidget(EstimatorWidgetMixin, QWidget, Ui_PLSCFWidget):
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

    estimator_cls = PLSCF

    def __init__(self, prep_signals, instance=None, parent=None):
        super().__init__(parent)
        self._init_estimator(prep_signals, instance)

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
        self._adopt_instance(instance)

        if instance.nperseg is not None:
            self.spin_nperseg.setValue(instance.nperseg)
        if instance.begin_frequency is not None:
            self.spin_begin_frequency.setValue(instance.begin_frequency)
        if instance.end_frequency is not None:
            self.spin_end_frequency.setValue(instance.end_frequency)
        if instance.num_blocks is not None:
            self.spin_num_blocks.setValue(instance.num_blocks)
            self.edit_training_blocks.setText(
                ','.join(str(b) for b in instance.training_blocks))
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
        num_blocks = self.spin_num_blocks.value() or None
        training_blocks = _parse_int_list(self.edit_training_blocks.text()) if num_blocks else None
        self._run_step(
            "Build Half-Spectra", self.instance.build_half_spectra,
            nperseg, begin_frequency, end_frequency, window_decay=window_decay,
            num_blocks=num_blocks, training_blocks=training_blocks)

    # ------------------------------------------------------------------
    # Step 2: compute_modal_params
    # ------------------------------------------------------------------
    def _on_compute_modal_params(self):
        max_model_order = self.spin_max_model_order.value()
        complex_coefficients = self.chk_complex_coefficients.isChecked()
        algo = self.combo_algo.currentText()
        modal_contrib = _MODAL_CONTRIB_VALUES[self.combo_modal_contrib.currentText()]
        validation_blocks = _parse_int_list(self.edit_validation_blocks.text())
        self._run_step(
            "Compute Modal Parameters", self.instance.compute_modal_params,
            max_model_order, complex_coefficients=complex_coefficients,
            algo=algo, modal_contrib=modal_contrib, validation_blocks=validation_blocks)
