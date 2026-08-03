# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2026  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 widget for pyOMA.core.SSICovRef.BRSSICovRef.

Wraps a :class:`~pyOMA.core.SSICovRef.BRSSICovRef` instance: each button
runs one step of its ``build_toeplitz_cov`` -> ``compute_modal_params``
sequence (see ``BRSSICovRef.init_from_config`` for the canonical order)
directly on the instance, so the object handed in (or created on first use)
is mutated in place. ``PogerSSICovRef`` (the multi-setup PoGER variant) is
a genuinely different, multi-``PreProcessSignals`` workflow and is not
covered by this widget.

Widget layout lives in ``ui/ssi_cov_ref.ui`` (compiled to
``generated/ui_ssi_cov_ref.py`` by ``scripts/build_ui.py``); this module
only wires signals/slots and the build/compute steps.
"""
from PyQt6.QtWidgets import QWidget

from .generated.ui_ssi_cov_ref import Ui_BRSSICovRefWidget
from .HelpersGUI import EstimatorWidgetMixin, _parse_int_list
from ..core.SSICovRef import BRSSICovRef


class BRSSICovRefWidget(EstimatorWidgetMixin, QWidget, Ui_BRSSICovRefWidget):
    """Interactive widget for the BRSSICovRef (SSI-Cov-Ref) method.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object used to construct a new
        ``BRSSICovRef`` instance, if *instance* is not given.
    instance : BRSSICovRef, optional
        An existing (possibly partially or fully computed) ``BRSSICovRef``
        object to inspect/continue. Defaults to a fresh, unbuilt instance.
    parent : QWidget, optional
    """

    estimator_cls = BRSSICovRef

    def __init__(self, prep_signals, instance=None, parent=None):
        super().__init__(parent)
        self._init_estimator(prep_signals, instance)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.btn_build_toeplitz_cov.clicked.connect(self._on_build_toeplitz_cov)
        self.btn_compute_modal_params.clicked.connect(self._on_compute_modal_params)
        self.btn_estimate_state.clicked.connect(self._on_estimate_state)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    def set_instance(self, instance):
        """Adopt *instance* as the object this widget operates on and refresh
        every field/button from its current state."""
        self._adopt_instance(instance)

        if instance.num_block_columns is not None:
            self.spin_num_block_columns.setValue(instance.num_block_columns)
        if instance.num_block_rows is not None:
            self.spin_num_block_rows.setValue(instance.num_block_rows)
        if instance.num_blocks is not None:
            self.spin_num_blocks.setValue(instance.num_blocks)
            self.edit_training_blocks.setText(
                ','.join(str(b) for b in instance.training_blocks))
        if instance.max_model_order is not None:
            self.spin_max_model_order.setValue(instance.max_model_order)

        built = instance.state[0]
        self.btn_compute_modal_params.setEnabled(built)
        self.btn_estimate_state.setEnabled(built)
        self._refresh_status()

    def _refresh_status(self):
        built, _state_mat, computed = self.instance.state
        if computed:
            text = ("Modal parameters computed up to order "
                    f"{self.instance.max_model_order}.")
        elif built:
            text = "Toeplitz covariance matrix built. Ready to compute modal parameters."
        else:
            text = "Not built yet."
        self.lbl_status.setText(text)

    # ------------------------------------------------------------------
    # Step 1: build_toeplitz_cov
    # ------------------------------------------------------------------
    def _on_build_toeplitz_cov(self):
        num_block_columns = self.spin_num_block_columns.value() or None
        num_block_rows = self.spin_num_block_rows.value() or None
        shift = self.spin_shift.value()
        num_blocks = self.spin_num_blocks.value() or None
        training_blocks = _parse_int_list(self.edit_training_blocks.text()) if num_blocks else None
        self._run_step(
            "Build Toeplitz Covariance Matrix", self.instance.build_toeplitz_cov,
            num_block_columns, num_block_rows=num_block_rows, shift=shift,
            num_blocks=num_blocks, training_blocks=training_blocks)

    # ------------------------------------------------------------------
    # Step 2: compute_modal_params
    # ------------------------------------------------------------------
    def _on_compute_modal_params(self):
        max_model_order = self.spin_max_model_order.value() or None
        max_modes = self.spin_max_modes.value() or None
        algo = self.combo_algo.currentText()
        modal_contrib = self.chk_modal_contrib.isChecked()
        validation_blocks = _parse_int_list(self.edit_validation_blocks.text())
        self._run_step(
            "Compute Modal Parameters", self.instance.compute_modal_params,
            max_model_order, max_modes=max_modes, algo=algo,
            modal_contrib=modal_contrib, validation_blocks=validation_blocks)

    # ------------------------------------------------------------------
    # Advanced: estimate_state (single order)
    # ------------------------------------------------------------------
    def _on_estimate_state(self):
        order = self.spin_estimate_order.value()
        max_modes = self.spin_estimate_max_modes.value() or None
        algo = self.combo_estimate_algo.currentText()
        # estimate_state reports its own result rather than advancing the
        # instance's state, so it skips the post-step refresh.
        result = self._run_step(
            "Estimate State", self.instance.estimate_state,
            order, max_modes=max_modes, algo=algo, refresh=False)
        if result is self.STEP_FAILED:
            return
        A, C, _G = result
        self.lbl_status.setText(
            f"Estimated state at order {order}: A{A.shape}, C{C.shape}.")
