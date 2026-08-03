# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2026  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 widget for pyOMA.core.SSIData (SSIData/SSIDataMC/SSIDataCV).

Wraps one of the three closely-related data-driven SSI classes: a
``combo_variant`` box picks the concrete class, and the widget shows/hides
the variant-specific option groups accordingly. Every button runs one step
of the ``build_block_hankel`` -> ``compute_modal_params`` sequence (see each
class's ``init_from_config``, and ``SSIDataMC``'s docstring for
``SSIDataCV``) directly on the instance, mutating it in place.

Widget layout lives in ``ui/ssi_data.ui`` (compiled to
``generated/ui_ssi_data.py`` by ``scripts/build_ui.py``); this module only
wires signals/slots and the build/compute steps.
"""
from PyQt6.QtWidgets import QWidget, QMessageBox

from .generated.ui_ssi_data import Ui_SSIDataWidget
from .HelpersGUI import EstimatorWidgetMixin, _parse_int_list
from ..core.SSIData import SSIData, SSIDataMC, SSIDataCV

_VARIANTS = {
    'SSI-Data': SSIData,
    'SSI-Data (Modal Contributions)': SSIDataMC,
    'SSI-Data (Cross-Validation)': SSIDataCV,
}
_VARIANT_NAMES = {cls: name for name, cls in _VARIANTS.items()}


class SSIDataWidget(EstimatorWidgetMixin, QWidget, Ui_SSIDataWidget):
    """Interactive widget for the SSI-Data family (SSIData/SSIDataMC/SSIDataCV).

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object used to construct a new instance, if
        *instance* is not given.
    instance : SSIDataMC, optional
        An existing (possibly partially or fully computed) instance of one
        of the three variants to inspect/continue. Defaults to a fresh,
        unbuilt ``SSIData`` instance.
    parent : QWidget, optional
    """

    # Builds a plain SSIData, but adopts any member of the SSIData family.
    estimator_cls = SSIData
    instance_cls = SSIDataMC
    instance_type_name = 'SSIData/SSIDataMC/SSIDataCV'

    def __init__(self, prep_signals, instance=None, parent=None):
        super().__init__(parent)
        self._init_estimator(prep_signals, instance)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.combo_variant.currentTextChanged.connect(self._on_variant_changed)
        self.btn_build_block_hankel.clicked.connect(self._on_build_block_hankel)
        self.btn_compute_modal_params.clicked.connect(self._on_compute_modal_params)
        self.btn_estimate_state.clicked.connect(self._on_estimate_state)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    def set_instance(self, instance):
        """Adopt *instance* as the object this widget operates on and refresh
        every field/button from its current state."""
        self._adopt_instance(instance)

        self.combo_variant.blockSignals(True)
        self.combo_variant.setCurrentText(_VARIANT_NAMES[type(instance)])
        self.combo_variant.blockSignals(False)
        self._update_variant_visibility()

        if instance.num_block_rows is not None:
            self.spin_num_block_rows.setValue(instance.num_block_rows)
        if isinstance(instance, SSIDataCV):
            self.spin_num_blocks.setValue(instance.num_blocks)
        if instance.max_model_order is not None:
            self.spin_max_model_order.setValue(instance.max_model_order)

        built = instance.state[0]
        self.btn_compute_modal_params.setEnabled(built)
        self.btn_estimate_state.setEnabled(built)
        self._refresh_status()

    def _update_variant_visibility(self):
        cls = type(self.instance)
        self.cv_build_box.setVisible(cls is SSIDataCV)
        self.mc_compute_box.setVisible(cls is SSIDataMC)
        self.cv_compute_box.setVisible(cls is SSIDataCV)
        self.plain_advanced_box.setVisible(cls is SSIData)

    def _refresh_status(self):
        built, _unused1, computed, _unused2 = self.instance.state
        variant = _VARIANT_NAMES[type(self.instance)]
        if computed:
            text = (f"[{variant}] Modal parameters computed up to order "
                    f"{self.instance.max_model_order}.")
        elif built:
            text = f"[{variant}] Block-Hankel matrix built. Ready to compute modal parameters."
        else:
            text = f"[{variant}] Not built yet."
        self.lbl_status.setText(text)

    # ------------------------------------------------------------------
    # Variant switching
    # ------------------------------------------------------------------
    def _on_variant_changed(self, text):
        new_cls = _VARIANTS[text]
        if type(self.instance) is new_cls:
            self._update_variant_visibility()
            return
        if any(self.instance.state):
            reply = QMessageBox.question(
                self, "Switch OMA variant",
                "Switching variants discards the current build/compute progress. Continue?")
            if reply != QMessageBox.StandardButton.Yes:
                self.combo_variant.blockSignals(True)
                self.combo_variant.setCurrentText(_VARIANT_NAMES[type(self.instance)])
                self.combo_variant.blockSignals(False)
                return
        self.set_instance(new_cls(self.prep_signals))

    # ------------------------------------------------------------------
    # Step 1: build_block_hankel
    # ------------------------------------------------------------------
    def _on_build_block_hankel(self):
        num_block_rows = self.spin_num_block_rows.value()
        reduced_projection = self.chk_reduced_projection.isChecked()
        # Nested so that argument parsing stays inside the guarded region, and
        # named after the step so the log message is unchanged.
        def build_block_hankel():
            if isinstance(self.instance, SSIDataCV):
                training_blocks = _parse_int_list(self.edit_training_blocks.text())
                self.instance.build_block_hankel(
                    num_block_rows, num_blocks=self.spin_num_blocks.value(),
                    training_blocks=training_blocks, reduced_projection=reduced_projection)
            else:
                self.instance.build_block_hankel(
                    num_block_rows, reduced_projection=reduced_projection)

        self._run_step("Build Block-Hankel Matrix", build_block_hankel)

    # ------------------------------------------------------------------
    # Step 2: compute_modal_params
    # ------------------------------------------------------------------
    def _on_compute_modal_params(self):
        max_model_order = self.spin_max_model_order.value()
        cls = type(self.instance)

        def compute_modal_params():
            if cls is SSIData:
                self.instance.compute_modal_params(max_model_order)
            elif cls is SSIDataCV:
                validation_blocks = _parse_int_list(self.edit_validation_blocks.text())
                self.instance.compute_modal_params(
                    max_model_order, validation_blocks=validation_blocks)
            else:  # SSIDataMC
                self.instance.compute_modal_params(
                    max_model_order, j=(self.spin_j.value() or None),
                    synth_sig=self.chk_synth_sig.isChecked())

        self._run_step("Compute Modal Parameters", compute_modal_params)

    # ------------------------------------------------------------------
    # Advanced: estimate_state (single order)
    # ------------------------------------------------------------------
    def _on_estimate_state(self):
        order = self.spin_estimate_order.value()

        def estimate_state():
            if type(self.instance) is SSIData:
                max_modes = self.spin_estimate_max_modes.value() or None
                algo = self.combo_estimate_algo.currentText()
                return self.instance.estimate_state(
                    order, max_modes=max_modes, algo=algo)
            return self.instance.estimate_state(order)

        # Reports its own result rather than advancing the instance's state.
        result = self._run_step("Estimate State", estimate_state, refresh=False)
        if result is self.STEP_FAILED:
            return
        A, C, _Q, _R, _S = result
        self.lbl_status.setText(
            f"Estimated state at order {order}: A{A.shape}, C{C.shape}.")
