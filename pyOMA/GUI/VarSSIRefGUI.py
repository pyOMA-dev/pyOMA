# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 widget for pyOMA.core.VarSSIRef.VarSSIRef.

Wraps a :class:`~pyOMA.core.VarSSIRef.VarSSIRef` instance: each button runs
one step of its ``build_subspace_mat`` -> ``compute_state_matrices`` ->
``prepare_sensitivities`` -> ``compute_modal_params`` sequence (see
``VarSSIRef.init_from_config``) directly on the instance, mutating it in
place. ``VarSSIRef`` tracks "sensitivities prepared" as its own
``sensitivities_prepared`` attribute (separate from the 3-slot ``state``
list, which only covers build/compute-state-matrices/compute-modal-params)
and invalidates downstream flags itself whenever an earlier step is re-run,
so this widget can read button-enablement straight off the instance.

Widget layout lives in ``ui/var_ssi_ref.ui`` (compiled to
``generated/ui_var_ssi_ref.py`` by ``scripts/build_ui.py``); this module
only wires signals/slots and the build/compute steps.
"""
import logging

from PyQt6.QtWidgets import QWidget, QMessageBox

from .generated.ui_var_ssi_ref import Ui_VarSSIRefWidget
from ..core.VarSSIRef import VarSSIRef
from ..core.PreProcessingTools import PreProcessSignals

logger = logging.getLogger(__name__)


class VarSSIRefWidget(QWidget, Ui_VarSSIRefWidget):
    """Interactive widget for the VarSSIRef (variance-based SSI-Cov-Ref) method.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object used to construct a new ``VarSSIRef``
        instance, if *instance* is not given.
    instance : VarSSIRef, optional
        An existing (possibly partially or fully computed) ``VarSSIRef``
        object to inspect/continue. Defaults to a fresh, unbuilt instance.
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

        self.set_instance(
            instance if instance is not None else VarSSIRef(prep_signals))

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.btn_build_subspace_mat.clicked.connect(self._on_build_subspace_mat)
        self.btn_compute_state_matrices.clicked.connect(self._on_compute_state_matrices)
        self.btn_prepare_sensitivities.clicked.connect(self._on_prepare_sensitivities)
        self.btn_compute_modal_params.clicked.connect(self._on_compute_modal_params)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    def set_instance(self, instance):
        """Adopt *instance* as the object this widget operates on and refresh
        every field/button from its current state."""
        if not isinstance(instance, VarSSIRef):
            raise TypeError(
                f"instance must be a VarSSIRef instance, got {type(instance).__name__}")
        self.instance = instance
        self._refresh()

    def _refresh(self):
        """Repopulate fields/buttons from ``self.instance``'s current state."""
        instance = self.instance
        if instance.num_block_columns is not None:
            self.spin_num_block_columns.setValue(instance.num_block_columns)
        if instance.num_block_rows is not None:
            self.spin_num_block_rows.setValue(instance.num_block_rows)
        if instance.max_model_order is not None:
            self.spin_state_max_model_order.setValue(instance.max_model_order)
            self.spin_max_model_order.setValue(instance.max_model_order)
        self.combo_lsq_method.setCurrentText(instance.lsq_method)
        self.combo_variance_algo.setCurrentText(instance.variance_algo)

        self.btn_compute_state_matrices.setEnabled(instance.state[0])
        self.btn_prepare_sensitivities.setEnabled(instance.state[1])
        self.btn_compute_modal_params.setEnabled(instance.sensitivities_prepared)
        self._refresh_status()

    def _refresh_status(self):
        instance = self.instance
        if instance.state[2]:
            text = f"Modal parameters computed up to order {instance.max_model_order}."
        elif instance.sensitivities_prepared:
            text = "Sensitivities prepared. Ready to compute modal parameters."
        elif instance.state[1]:
            text = "State matrices computed. Ready to prepare sensitivities."
        elif instance.state[0]:
            text = "Subspace matrix built. Ready to compute state matrices."
        else:
            text = "Not built yet."
        self.lbl_status.setText(text)

    # ------------------------------------------------------------------
    # Step 1: build_subspace_mat
    # ------------------------------------------------------------------
    def _on_build_subspace_mat(self):
        num_block_columns = self.spin_num_block_columns.value()
        num_block_rows = self.spin_num_block_rows.value() or None
        num_blocks = self.spin_num_blocks.value() or None
        subspace_method = self.combo_subspace_method.currentText()
        try:
            self.instance.build_subspace_mat(
                num_block_columns, num_block_rows=num_block_rows,
                num_blocks=num_blocks, subspace_method=subspace_method)
        except Exception as exc:
            logger.exception("build_subspace_mat failed")
            QMessageBox.warning(self, "Build Subspace Matrix failed", str(exc))
            return
        self._refresh()

    # ------------------------------------------------------------------
    # Step 2: compute_state_matrices
    # ------------------------------------------------------------------
    def _on_compute_state_matrices(self):
        max_model_order = self.spin_state_max_model_order.value() or None
        lsq_method = self.combo_lsq_method.currentText()
        try:
            self.instance.compute_state_matrices(max_model_order, lsq_method=lsq_method)
        except Exception as exc:
            logger.exception("compute_state_matrices failed")
            QMessageBox.warning(self, "Compute State Matrices failed", str(exc))
            return
        self._refresh()

    # ------------------------------------------------------------------
    # Step 3: prepare_sensitivities
    # ------------------------------------------------------------------
    def _on_prepare_sensitivities(self):
        variance_algo = self.combo_variance_algo.currentText()
        debug = self.chk_sensitivities_debug.isChecked()
        try:
            self.instance.prepare_sensitivities(variance_algo=variance_algo, debug=debug)
        except Exception as exc:
            logger.exception("prepare_sensitivities failed")
            QMessageBox.warning(self, "Prepare Sensitivities failed", str(exc))
            return
        self._refresh()

    # ------------------------------------------------------------------
    # Step 4: compute_modal_params
    # ------------------------------------------------------------------
    def _on_compute_modal_params(self):
        max_model_order = self.spin_max_model_order.value() or None
        debug = self.chk_compute_debug.isChecked()
        try:
            self.instance.compute_modal_params(max_model_order, debug=debug)
        except Exception as exc:
            logger.exception("compute_modal_params failed")
            QMessageBox.warning(self, "Compute Modal Parameters failed", str(exc))
            return
        self._refresh()
