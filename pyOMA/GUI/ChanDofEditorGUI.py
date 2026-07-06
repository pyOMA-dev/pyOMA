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
from ..core.PlotMSH import ModeShapePlot

logger = logging.getLogger(__name__)

_HIGHLIGHT_COLOR = 'red'


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

        self.mode_shape_plot = ModeShapePlot(geometry_data, prep_signals=prep_signals)
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
        fig = self.mode_shape_plot.fig
        self.canvas.set_figure(fig)
        self.mode_shape_plot.canvas = self.canvas
        subplot = self.mode_shape_plot.subplot
        self.canvas.mpl_connect('motion_notify_event', subplot._on_move)
        self.canvas.mpl_connect('button_press_event', subplot._button_press)
        self.canvas.mpl_connect('button_release_event', subplot._button_release)
        subplot.mouse_init()

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
        # Full clear + redraw (nodes/lines/parent-childs/other channel-DOF
        # assignments), mirroring GeometryProcessorGUI._redraw_geometry -
        # reset_view() only ever adds artists, it never removes ones left
        # over from a previous draw.
        msh = self.mode_shape_plot
        subplot = msh.subplot
        subplot.cla()
        subplot.set_aspect('equal', 'datalim')
        subplot.patch = msh.fig.patch
        subplot.grid(False)
        subplot.set_axis_off()
        subplot.mouse_init()
        msh.patches_objects = {}
        msh.lines_objects = []
        msh.nd_lines_objects = []
        msh.cn_lines_objects = {}
        msh.arrows_objects = []
        msh.channels_objects = []
        msh.axis_obj = {}
        msh.reset_view()

        node = self.combo_node.currentText()
        if node:
            self._draw_highlighted_draft(node)

        self.canvas.draw_idle()

    def _draw_highlighted_draft(self, node):
        msh = self.mode_shape_plot
        az = self.spin_az.value()
        elev = self.spin_elev.value()
        channel_name = self.prep_signals.channel_headers[self.channel]
        index = len(msh.channels_objects)
        msh.add_chan_dof(self.channel, node, az, elev, channel_name, index)
        arrow = msh.channels_objects[index]
        arrow.set_color(_HIGHLIGHT_COLOR)
        if arrow.text is not None:
            arrow.text.set_color(_HIGHLIGHT_COLOR)

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
