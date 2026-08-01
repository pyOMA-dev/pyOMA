# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal PyQt6 dialog to define a channel-DOF assignment.

Wraps :meth:`~pyOMA.core.PreProcessingTools.PreProcessSignals.set_chan_dof` /
:meth:`~...remove_chan_dof`: a :class:`~pyOMA.core.PlotMSH.ModeShapePlot`
preview shows every existing channel-DOF assignment, with the one currently
being edited highlighted in a different color. OK commits the node/azimuth/
elevation currently shown, Cancel discards, Delete removes any existing
assignment for this channel.

Widget layout lives in ``ui/chan_dof_editor.ui`` (compiled to
``generated/ui_chan_dof_editor.py`` by ``scripts/build_ui.py``); this module
only wires signals/slots and holds the preview logic.
"""
import logging

from PyQt6.QtWidgets import QDialog, QMessageBox

from .generated.ui_chan_dof_editor import Ui_ChanDofEditorGUI
from ..core.PreProcessingTools import PreProcessSignals, GeometryProcessor
from ..core import resolve_mode_shape_backend

logger = logging.getLogger(__name__)


class ChanDofEditorGUI(QDialog, Ui_ChanDofEditorGUI):
    """Dialog to add/edit/delete the DOF assignment of a single channel.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Provides the channel to edit and the ``chan_dofs`` list mutated by
        this dialog (via :meth:`~PreProcessSignals.set_chan_dof` /
        :meth:`~PreProcessSignals.remove_chan_dof`).
    geometry_data : GeometryProcessor
        Supplies the nodes offered in the node dropdown and drawn in the
        preview.
    channel : int
        Index of the channel being assigned.
    parent : QWidget, optional
    """

    def __init__(self, prep_signals, geometry_data, channel, parent=None):
        super().__init__(parent)
        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be a PreProcessSignals instance, "
                f"got {type(prep_signals).__name__}")
        if not isinstance(geometry_data, GeometryProcessor):
            raise TypeError(
                f"geometry_data must be a GeometryProcessor instance, "
                f"got {type(geometry_data).__name__}")
        self.prep_signals = prep_signals
        self.geometry_data = geometry_data
        self.channel = channel

        # Backend chosen globally; both drive the editor through the neutral
        # redraw_geometry / draw_draft_chan_dof / attach_qt_canvas contract.
        self.mode_shape_plot = resolve_mode_shape_backend()(
            geometry_data, prep_signals=prep_signals)
        # This channel's own (possibly pre-existing) assignment is redrawn
        # separately below, highlighted, as the live-edited draft - drop it
        # from the "other assignments" background context so it isn't drawn
        # twice.
        self.mode_shape_plot.chan_dofs = [
            cd for cd in self.mode_shape_plot.chan_dofs if cd[0] != channel]
        # Parent-child assignments are unrelated to channel-DOF assignment
        # and would only clutter this preview.
        self.mode_shape_plot.show_parent_childs = False

        self.setupUi(self)
        channel_name = prep_signals.channel_headers[channel]
        self.setWindowTitle(f"DOF assignment: channel {channel}: {channel_name}")
        self._wire_canvas()
        self._wire_controls()
        self._populate_node_combo()
        self._load_existing_assignment()
        self._redraw_preview()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_canvas(self):
        # Backend-neutral (see GeometryProcessorGUI._wire_canvas): matplotlib
        # adopts the placeholder canvas, a pyvista QtInteractor replaces it.
        widget = self.mode_shape_plot.attach_qt_canvas(self.canvas)
        if widget is not self.canvas:
            self.root_layout.replaceWidget(self.canvas, widget)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = widget
        self.mode_shape_plot.canvas = self.canvas

    def _wire_controls(self):
        self.combo_node.currentIndexChanged.connect(self._redraw_preview)
        self.spin_az.valueChanged.connect(self._redraw_preview)
        self.spin_elev.valueChanged.connect(self._redraw_preview)
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_delete.clicked.connect(self._on_delete)

    def _populate_node_combo(self):
        self.combo_node.addItems(sorted(self.geometry_data.nodes.keys()))

    def _load_existing_assignment(self):
        existing = self.prep_signals.get_chan_dof(self.channel)
        self.btn_delete.setEnabled(existing is not None)
        if existing is None:
            return
        node, az, elev = existing
        self.combo_node.setCurrentText(node)
        self.spin_az.setValue(az)
        self.spin_elev.setValue(elev)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _redraw_preview(self):
        # Backend-neutral: rebuild the geometry (which draws the *other*
        # channels' assignments), show those arrows, then overlay the
        # highlighted draft for the channel currently being edited.
        msh = self.mode_shape_plot
        msh.redraw_geometry()
        msh.refresh_chan_dofs(True)

        node = self.combo_node.currentText()
        if node:
            self._draw_highlighted_draft(node)

        msh.redraw()

    def _draw_highlighted_draft(self, node):
        az = self.spin_az.value()
        elev = self.spin_elev.value()
        channel_name = self.prep_signals.channel_headers[self.channel]
        self.mode_shape_plot.draw_draft_chan_dof(
            self.channel, node, az, elev, channel_name)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_ok(self):
        node = self.combo_node.currentText()
        if not node:
            QMessageBox.warning(self, "DOF assignment", "Select a node first.")
            return
        self.prep_signals.set_chan_dof(
            self.channel, node, self.spin_az.value(), self.spin_elev.value())
        self.accept()

    def _on_delete(self):
        self.prep_signals.remove_chan_dof(self.channel)
        self.accept()
