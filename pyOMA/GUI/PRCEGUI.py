# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2026  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 widget for pyOMA.core.PRCE.PRCE.

Wraps a :class:`~pyOMA.core.PRCE.PRCE` instance: each button runs one step of
its ``build_corr_tensor`` -> ``compute_modal_params`` sequence (see
``PRCE.init_from_config`` for the canonical order) directly on the instance,
so the object handed in (or created on first use) is mutated in place.

Widget layout lives in ``ui/prce.ui`` (compiled to ``generated/ui_prce.py``
by ``scripts/build_ui.py``); this module only wires signals/slots and the
build/compute steps.
"""
from PyQt6.QtWidgets import QWidget

from .generated.ui_prce import Ui_PRCEWidget
from .HelpersGUI import EstimatorWidgetMixin
from ..core.PRCE import PRCE


class PRCEWidget(EstimatorWidgetMixin, QWidget, Ui_PRCEWidget):
    """Interactive widget for the PRCE (Poly-reference Complex Exponential) method.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object used to construct a new ``PRCE``
        instance, if *instance* is not given.
    instance : PRCE, optional
        An existing (possibly partially or fully computed) ``PRCE`` object
        to inspect/continue. Defaults to a fresh, unbuilt instance.
    parent : QWidget, optional
    """

    estimator_cls = PRCE

    def __init__(self, prep_signals, instance=None, parent=None):
        super().__init__(parent)
        self._init_estimator(prep_signals, instance)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_buttons(self):
        self.btn_build_corr_tensor.clicked.connect(self._on_build_corr_tensor)
        self.btn_compute_modal_params.clicked.connect(self._on_compute_modal_params)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    def set_instance(self, instance):
        """Adopt *instance* as the object this widget operates on and refresh
        every field/button from its current state."""
        self._adopt_instance(instance)

        if instance.num_corr_samples is not None:
            self.spin_num_corr_samples.setValue(instance.num_corr_samples)
        if instance.max_model_order is not None:
            self.spin_max_model_order.setValue(instance.max_model_order)

        built = instance.state[0]
        self.btn_compute_modal_params.setEnabled(built)
        self._refresh_status()

    def _refresh_status(self):
        built, computed = self.instance.state
        if computed:
            text = ("Modal parameters computed up to order "
                    f"{self.instance.max_model_order}.")
        elif built:
            text = "Correlation tensor built. Ready to compute modal parameters."
        else:
            text = "Not built yet."
        self.lbl_status.setText(text)

    # ------------------------------------------------------------------
    # Step 1: build_corr_tensor
    # ------------------------------------------------------------------
    def _on_build_corr_tensor(self):
        self._run_step(
            "Build Correlation Tensor", self.instance.build_corr_tensor,
            self.spin_num_corr_samples.value())

    # ------------------------------------------------------------------
    # Step 2: compute_modal_params
    # ------------------------------------------------------------------
    def _on_compute_modal_params(self):
        self._run_step(
            "Compute Modal Parameters", self.instance.compute_modal_params,
            self.spin_max_model_order.value())
