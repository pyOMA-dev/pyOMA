# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Interactive PyQt6 GUI for pyOMA.core.PreProcessingTools.GeometryProcessor.

Wraps :class:`~pyOMA.core.PreProcessingTools.GeometryProcessor`: every
node/line/parent-child action mutates the ``GeometryProcessor`` instance in
place, and a :class:`~pyOMA.core.PlotMSH.ModeShapePlot` (geometry-only, no
modal data) re-renders the 3-D preview against its current state.

Widget layout lives in ``ui/geometry_processor.ui`` (compiled to
``generated/ui_geometry_processor.py`` by ``scripts/build_ui.py``); this
module only wires signals/slots and holds the table/preview logic.
"""
import os
import sys
import logging

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QComboBox, QTableWidgetItem,
    QFileDialog, QInputDialog,
)
from PyQt6.QtCore import Qt, QEventLoop

from .generated.ui_geometry_processor import Ui_GeometryProcessorGUI
from .HelpersGUI import save_figure_dialog, UnsavedChangesMixin
from ..core.PreProcessingTools import GeometryProcessor
from ..core.PlotMSH import ModeShapePlot

logger = logging.getLogger(__name__)

app = None

_FILE_FILTER = "Text files (*.txt);;All files (*)"


class GeometryProcessorGUI(UnsavedChangesMixin, QMainWindow, Ui_GeometryProcessorGUI):
    """Interactive GUI for building/editing structural geometry.

    Parameters
    ----------
    geometry_data : GeometryProcessor, optional
        The geometry object to inspect and edit. Actions performed through
        this GUI mutate it in place. Defaults to a new, empty instance.
    parent : QWidget, optional
    """

    def __init__(self, geometry_data=None, parent=None):
        super().__init__(parent)
        if geometry_data is None:
            geometry_data = GeometryProcessor()
        if not isinstance(geometry_data, GeometryProcessor):
            raise TypeError(
                f"geometry_data must be a GeometryProcessor instance, "
                f"got {type(geometry_data).__name__}")
        self.geometry_data = geometry_data
        self.mode_shape_plot = ModeShapePlot(geometry_data)
        self._dirty = False

        self.setupUi(self)
        self._wire_canvas()
        self._wire_viewport()
        self._wire_node_table()
        self._wire_line_table()
        self._wire_pc_table()
        self._wire_menu()
        self._wire_buttons()

        self._refresh_node_table()
        self._refresh_line_table()
        self._refresh_pc_table()
        self._redraw_geometry()
        self.show()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_canvas(self):
        fig = self.mode_shape_plot.fig
        fig.set_size_inches((100, 100))
        self.canvas.set_figure(fig)
        self.mode_shape_plot.canvas = self.canvas
        subplot = self.mode_shape_plot.subplot
        self.canvas.mpl_connect('motion_notify_event', subplot._on_move)
        self.canvas.mpl_connect('button_press_event', subplot._button_press)
        self.canvas.mpl_connect('button_release_event', subplot._button_release)
        subplot.mouse_init()

    def _wire_viewport(self):
        for button in (self.viewport_button_x, self.viewport_button_y,
                       self.viewport_button_z, self.viewport_button_iso):
            button.released.connect(self._on_viewport_button)
        self.reset_button.clicked.connect(self._redraw_geometry)

    def _on_viewport_button(self):
        self.mode_shape_plot.change_viewport(self.sender().text())

    def _wire_node_table(self):
        self.btn_add_node.clicked.connect(self._on_add_node)
        self.btn_delete_node.clicked.connect(self._on_delete_node)
        self.node_table.itemChanged.connect(self._on_node_item_changed)

    def _wire_line_table(self):
        self.btn_add_line.clicked.connect(self._on_add_line)
        self.btn_delete_line.clicked.connect(self._on_delete_line)

    def _wire_pc_table(self):
        self.btn_add_pc.clicked.connect(self._on_add_pc)
        self.btn_delete_pc.clicked.connect(self._on_delete_pc)
        self.pc_table.itemChanged.connect(self._on_pc_item_changed)

    def _wire_menu(self):
        self.action_load_nodes.triggered.connect(self._on_load_nodes)
        self.action_load_lines.triggered.connect(self._on_load_lines)
        self.action_load_parent_childs.triggered.connect(self._on_load_parent_childs)
        self.action_save_nodes.triggered.connect(self._on_save_nodes)
        self.action_save_lines.triggered.connect(self._on_save_lines)
        self.action_save_parent_childs.triggered.connect(self._on_save_parent_childs)
        self.action_quit.triggered.connect(self.close)

    def _wire_buttons(self):
        self.load_state_button.clicked.connect(self.load_state)
        self.save_state_button.clicked.connect(self.save_state)
        self.save_figure_button.clicked.connect(self.save_figure)
        self.ok_close_button.clicked.connect(self.close)

    # ------------------------------------------------------------------
    # Node table
    # ------------------------------------------------------------------
    def _refresh_node_table(self):
        table = self.node_table
        table.blockSignals(True)
        table.setRowCount(0)
        for name in sorted(self.geometry_data.nodes.keys()):
            x, y, z = self.geometry_data.nodes[name]
            row = table.rowCount()
            table.insertRow(row)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, QTableWidgetItem(f'{x:g}'))
            table.setItem(row, 2, QTableWidgetItem(f'{y:g}'))
            table.setItem(row, 3, QTableWidgetItem(f'{z:g}'))
        table.blockSignals(False)

    def _suggest_node_name(self):
        i = 1
        while str(i) in self.geometry_data.nodes:
            i += 1
        return str(i)

    def _on_add_node(self):
        name, ok = QInputDialog.getText(
            self, "Add node", "Node name:", text=self._suggest_node_name())
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.geometry_data.nodes:
            QMessageBox.warning(self, "Add node", f"Node {name!r} already exists.")
            return
        self.geometry_data.add_node(name, [0.0, 0.0, 0.0])
        self._dirty = True
        self._refresh_node_table()
        self._refresh_line_table()
        self._refresh_pc_table()
        self._redraw_geometry()

    def _on_delete_node(self):
        rows = sorted({index.row() for index in self.node_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.warning(self, "Delete node", "Select at least one node first.")
            return
        names = [self.node_table.item(row, 0).text() for row in rows]
        for name in names:
            self.geometry_data.take_node(name)
        self._dirty = True
        self._refresh_node_table()
        self._refresh_line_table()
        self._refresh_pc_table()
        self._redraw_geometry()

    def _on_node_item_changed(self, item):
        if item.column() == 0:
            return  # name is read-only; re-add under a new name instead of renaming
        row = item.row()
        name = self.node_table.item(row, 0).text()
        try:
            x = float(self.node_table.item(row, 1).text())
            y = float(self.node_table.item(row, 2).text())
            z = float(self.node_table.item(row, 3).text())
        except (ValueError, AttributeError):
            QMessageBox.warning(self, "Edit node", "Coordinates must be numeric.")
            self._refresh_node_table()
            return
        # add_node() overwrites coordinates in place for an existing name
        # without touching lines/parent-childs, so no take_node() needed here.
        self.geometry_data.add_node(name, [x, y, z])
        self._dirty = True
        self._refresh_node_table()
        self._redraw_geometry()

    # ------------------------------------------------------------------
    # Line table
    # ------------------------------------------------------------------
    def _refresh_line_table(self):
        table = self.line_table
        table.blockSignals(True)
        table.setRowCount(0)
        node_names = sorted(self.geometry_data.nodes.keys())
        for row, (start, end) in enumerate(self.geometry_data.lines):
            table.insertRow(row)
            table.setCellWidget(row, 0, self._make_line_combo(node_names, start, row))
            table.setCellWidget(row, 1, self._make_line_combo(node_names, end, row))
        table.blockSignals(False)

    def _make_line_combo(self, node_names, current, row):
        combo = QComboBox()
        combo.addItems(node_names)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(lambda _text, r=row: self._on_line_row_changed(r))
        return combo

    def _on_line_row_changed(self, row):
        old_line = tuple(self.geometry_data.lines[row])
        start = self.line_table.cellWidget(row, 0).currentText()
        end = self.line_table.cellWidget(row, 1).currentText()
        self.geometry_data.take_line(line=old_line)
        self.geometry_data.add_line((start, end))
        self._dirty = True
        self._refresh_line_table()
        self._redraw_geometry()

    def _on_add_line(self):
        node_names = sorted(self.geometry_data.nodes.keys())
        if len(node_names) < 2:
            QMessageBox.warning(self, "Add line", "Add at least two nodes first.")
            return
        self.geometry_data.add_line((node_names[0], node_names[1]))
        self._dirty = True
        self._refresh_line_table()
        self._redraw_geometry()

    def _on_delete_line(self):
        rows = sorted({index.row() for index in self.line_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.warning(self, "Delete line", "Select at least one line first.")
            return
        lines_to_remove = [tuple(self.geometry_data.lines[row]) for row in rows]
        for line in lines_to_remove:
            self.geometry_data.take_line(line=line)
        self._dirty = True
        self._refresh_line_table()
        self._redraw_geometry()

    # ------------------------------------------------------------------
    # Parent-child table
    # ------------------------------------------------------------------
    def _refresh_pc_table(self):
        table = self.pc_table
        table.blockSignals(True)
        table.setRowCount(0)
        node_names = sorted(self.geometry_data.nodes.keys())
        for row, (parent, x_p, y_p, z_p, child, x_c, y_c, z_c) in enumerate(
                self.geometry_data.parent_childs):
            table.insertRow(row)
            table.setCellWidget(row, 0, self._make_pc_combo(node_names, parent, row))
            table.setItem(row, 1, QTableWidgetItem(f'{x_p:g}'))
            table.setItem(row, 2, QTableWidgetItem(f'{y_p:g}'))
            table.setItem(row, 3, QTableWidgetItem(f'{z_p:g}'))
            table.setCellWidget(row, 4, self._make_pc_combo(node_names, child, row))
            table.setItem(row, 5, QTableWidgetItem(f'{x_c:g}'))
            table.setItem(row, 6, QTableWidgetItem(f'{y_c:g}'))
            table.setItem(row, 7, QTableWidgetItem(f'{z_c:g}'))
        table.blockSignals(False)

    def _make_pc_combo(self, node_names, current, row):
        combo = QComboBox()
        combo.addItems(node_names)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(lambda _text, r=row: self._on_pc_row_changed(r))
        return combo

    def _on_pc_item_changed(self, item):
        if item.column() in (0, 4):
            return  # combos, not plain items
        self._on_pc_row_changed(item.row())

    def _on_pc_row_changed(self, row):
        old_pc = self.geometry_data.parent_childs[row]
        try:
            parent = self.pc_table.cellWidget(row, 0).currentText()
            x_p = float(self.pc_table.item(row, 1).text())
            y_p = float(self.pc_table.item(row, 2).text())
            z_p = float(self.pc_table.item(row, 3).text())
            child = self.pc_table.cellWidget(row, 4).currentText()
            x_c = float(self.pc_table.item(row, 5).text())
            y_c = float(self.pc_table.item(row, 6).text())
            z_c = float(self.pc_table.item(row, 7).text())
        except (ValueError, AttributeError):
            QMessageBox.warning(self, "Edit parent-child", "Amplitudes must be numeric.")
            self._refresh_pc_table()
            return
        self.geometry_data.take_parent_child(ms=old_pc)
        self.geometry_data.add_parent_child((parent, x_p, y_p, z_p, child, x_c, y_c, z_c))
        self._dirty = True
        self._refresh_pc_table()
        self._redraw_geometry()

    def _on_add_pc(self):
        node_names = sorted(self.geometry_data.nodes.keys())
        if not node_names:
            QMessageBox.warning(self, "Add assignment", "Add at least one node first.")
            return
        parent = node_names[0]
        child = node_names[1] if len(node_names) > 1 else node_names[0]
        self.geometry_data.add_parent_child((parent, 1.0, 0.0, 0.0, child, 1.0, 0.0, 0.0))
        self._dirty = True
        self._refresh_pc_table()
        self._redraw_geometry()

    def _on_delete_pc(self):
        rows = sorted({index.row() for index in self.pc_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.warning(self, "Delete assignment", "Select at least one assignment first.")
            return
        pcs_to_remove = [self.geometry_data.parent_childs[row] for row in rows]
        for parent_child in pcs_to_remove:
            self.geometry_data.take_parent_child(ms=parent_child)
        self._dirty = True
        self._refresh_pc_table()
        self._redraw_geometry()

    # ------------------------------------------------------------------
    # File menu: load/save
    # ------------------------------------------------------------------
    def _on_load_nodes(self):
        fname, _sel = QFileDialog.getOpenFileName(self, "Load nodes", filter=_FILE_FILTER)
        if not fname:
            return
        self.geometry_data.add_nodes(GeometryProcessor.nodes_loader(fname))
        self._dirty = True
        self._refresh_node_table()
        self._refresh_line_table()
        self._refresh_pc_table()
        self._redraw_geometry()

    def _on_load_lines(self):
        fname, _sel = QFileDialog.getOpenFileName(self, "Load lines", filter=_FILE_FILTER)
        if not fname:
            return
        self.geometry_data.add_lines(GeometryProcessor.lines_loader(fname))
        self._dirty = True
        self._refresh_line_table()
        self._redraw_geometry()

    def _on_load_parent_childs(self):
        fname, _sel = QFileDialog.getOpenFileName(
            self, "Load parent-child assignments", filter=_FILE_FILTER)
        if not fname:
            return
        self.geometry_data.add_parent_childs(GeometryProcessor.parent_childs_loader(fname))
        self._dirty = True
        self._refresh_pc_table()
        self._redraw_geometry()

    def _on_save_nodes(self):
        fname, _sel = QFileDialog.getSaveFileName(self, "Save nodes", filter=_FILE_FILTER)
        if not fname:
            return
        self.geometry_data.nodes_saver(fname, self.geometry_data.nodes)

    def _on_save_lines(self):
        fname, _sel = QFileDialog.getSaveFileName(self, "Save lines", filter=_FILE_FILTER)
        if not fname:
            return
        self.geometry_data.lines_saver(fname, self.geometry_data.lines)

    def _on_save_parent_childs(self):
        fname, _sel = QFileDialog.getSaveFileName(
            self, "Save parent-child assignments", filter=_FILE_FILTER)
        if not fname:
            return
        self.geometry_data.parent_childs_saver(fname, self.geometry_data.parent_childs)

    def save_state(self):
        """Save nodes/lines/parent-childs together as nodes.txt/lines.txt/
        parent_childs.txt in one chosen directory (bundles the three
        individual save actions above)."""
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a directory to save the geometry state to")
        if not directory:
            return
        self.geometry_data.save_geometry(
            os.path.join(directory, 'nodes.txt'),
            os.path.join(directory, 'lines.txt'),
            os.path.join(directory, 'parent_childs.txt'))

    def load_state(self):
        """Load nodes/lines/parent-childs from a directory written by
        :meth:`save_state`, replacing the current geometry."""
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a directory to load the geometry state from")
        if not directory:
            return
        nodes_file = os.path.join(directory, 'nodes.txt')
        if not os.path.exists(nodes_file):
            QMessageBox.warning(self, "Load failed", f"No nodes.txt found in {directory!r}.")
            return
        try:
            loaded = GeometryProcessor.load_geometry(
                nodes_file,
                os.path.join(directory, 'lines.txt'),
                os.path.join(directory, 'parent_childs.txt'))
        except Exception as exc:
            logger.exception("load_geometry failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.geometry_data.nodes = loaded.nodes
        self.geometry_data.lines = loaded.lines
        self.geometry_data.parent_childs = loaded.parent_childs
        self._dirty = False  # freshly loaded from disk - nothing unsaved yet
        self._refresh_node_table()
        self._refresh_line_table()
        self._refresh_pc_table()
        self._redraw_geometry()

    def save_figure(self):
        save_figure_dialog(self, self.canvas)

    def _do_save(self):
        self.save_state()
        self._dirty = False

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _redraw_geometry(self):
        # ModeShapePlot.reset_view() only re-adds nodes/lines/parent-childs
        # still present in geometry_data; it never removes artists for ones
        # that were just deleted (its add_*() calls only overwrite the slot
        # for an index/key that is redrawn). So the axes are cleared and the
        # per-artist bookkeeping reset to empty before every redraw, mirroring
        # the fresh-axes state ModeShapePlot._setup_figure() constructs.
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
        msh.axis_obj = {}
        msh.reset_view()
        self.canvas.draw_idle()

    def closeEvent(self, event):
        if not self._prompt_save_on_close(event):
            return
        self.deleteLater()
        return QMainWindow.closeEvent(self, event)


def start_geometry_processor_gui(geometry_data=None):
    global app
    app = QApplication.instance() or QApplication(sys.argv)

    form = GeometryProcessorGUI(geometry_data)
    form.resize(1200, 800)

    loop = QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec()
    return


def main():
    start_geometry_processor_gui()


if __name__ == '__main__':
    main()
