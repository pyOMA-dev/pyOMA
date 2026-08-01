# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""PyQt6 GUI for interactive 3-D mode-shape animation (PlotMSHGUI)."""

# system i/o
from pyOMA.core.ModeShapeBase import ModeShapeBase
from .HelpersGUI import my_excepthook
from .generated.ui_plot_msh import Ui_PlotMSH
from matplotlib import rcParams
from PyQt6.QtCore import pyqtSignal, Qt, QEventLoop
from PyQt6.QtWidgets import QMainWindow, QButtonGroup, QStyle, \
    QFileDialog, QInputDialog, QApplication
import sys
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

app = None

sys.excepthook = my_excepthook


def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10 ** sig_fig) == int(b * 10 ** sig_fig)
            )


class ModeShapeGUI(QMainWindow, Ui_PlotMSH):
    """PyQt6 main window for interactive 3-D mode-shape animation.

    Wraps any :class:`~pyOMA.core.ModeShapeBase.ModeShapeBase` backend in
    a full Qt main-window with controls for mode selection, animation
    speed, amplitude scaling, and display options (nodes, beams, axis
    arrows, parent-child connections).

    The backend supplies its own rendering widget and interaction
    hookup, so both the matplotlib
    :class:`~pyOMA.core.PlotMSH.ModeShapePlot` and the pyvista
    :class:`~pyOMA.core.PlotMSHpv.ModeShapePlotPVQt` can be displayed.
    Controls a backend does not support are disabled rather than hidden.

    Parameters
    ----------
    msh_plot : ModeShapeBase
        Populated mode-shape plot object to display.

    .. TODO::
        * Button for Axes3d.set_axis_off/on
        * Use the logging module to replace print commands
    """

    # define this class's signals and the types of data they emit
    grid_requested = pyqtSignal(str, bool)
    beams_requested = pyqtSignal(str, bool)
    childs_requested = pyqtSignal(str, bool)
    chan_dofs_requested = pyqtSignal(str, bool)

    def __init__(self,
                 mode_shape_plot,
                 reduced_gui=False):
        """
        Parameters
        ----------
        mode_shape_plot : ModeShapeBase
            Populated mode-shape plot object to display.
        reduced_gui : bool, optional
            When ``True``, show a simplified control panel without advanced
            options.  Default is ``False``.
        """

        QMainWindow.__init__(self)
        if not isinstance(mode_shape_plot, ModeShapeBase):
            raise TypeError(
                f"mode_shape_plot must be a ModeShapeBase subclass, "
                f"got {type(mode_shape_plot).__name__!r}")
        self.mode_shape_plot = mode_shape_plot
        self.animated = False
        self.setupUi(self)

        self._wire_menu()
        self._wire_canvas(mode_shape_plot)
        self._wire_view_checkboxes(mode_shape_plot)
        self._wire_mode_controls(mode_shape_plot, reduced_gui)
        self._wire_animation_widgets(mode_shape_plot)
        self._wire_viewport_and_limits()
        self._apply_reduced_gui(reduced_gui)

        self.reset_view()
        self.mode_combo.setCurrentIndex(1)
        self.imag_checkbox.setChecked(True)
        self.mode_combo.setCurrentIndex(0)
        self.show()

    def _wire_menu(self):
        """Connect the .ui-declared menu actions to their slots."""
        self.action_save_plot.triggered.connect(self.save_plot)
        self.action_save_animation.triggered.connect(self.save_animation)
        self.action_quit.triggered.connect(self.close)

    def _wire_canvas(self, mode_shape_plot):
        """Install the backend's rendering widget and connect camera interaction.

        The plot object is asked for the widget to display and for its own
        interaction hookup, so this works for any backend.  Matplotlib
        adopts the ``MyMplCanvas`` placeholder from the .ui file and wires
        its 3-D mouse handlers; a VTK-based backend brings its own widget,
        which then replaces the placeholder in the layout.
        """
        widget = mode_shape_plot.attach_qt_canvas(self.canvas)

        if widget is not self.canvas:
            self.vbox.replaceWidget(self.canvas, widget)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = widget

        mode_shape_plot.canvas = self.canvas
        mode_shape_plot.connect_view_change(self.update_lims)

    def _wire_view_checkboxes(self, mode_shape_plot):
        """Set initial check states and connect the view-options checkboxes."""
        self.axis_checkbox.setTristate(False)
        self.axis_checkbox.setCheckState(
            Qt.CheckState.Checked if mode_shape_plot.show_axis else Qt.CheckState.Unchecked)
        self.axis_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_axis)

        self.nodes_checkbox.setTristate(False)
        self.nodes_checkbox.setCheckState(
            Qt.CheckState.Checked if mode_shape_plot.show_nodes else Qt.CheckState.Unchecked)
        self.nodes_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_nodes)

        self.line_checkbox.setTristate(False)
        self.line_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_lines)
        self.ms_checkbox.setTristate(False)
        self.chandof_checkbox.setTristate(False)

        self.conn_lines_checkbox.setTristate(False)
        self.conn_lines_checkbox.setCheckState(
            Qt.CheckState.Checked if mode_shape_plot.show_cn_lines else Qt.CheckState.Unchecked)
        self.conn_lines_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_cn_lines)

        self.nd_lines_checkbox.setTristate(False)
        self.nd_lines_checkbox.setCheckState(
            Qt.CheckState.Checked if mode_shape_plot.show_nd_lines else Qt.CheckState.Unchecked)
        self.nd_lines_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_nd_lines)

        self.traces_checkbox.setTristate(False)
        self.traces_checkbox.setCheckState(
            Qt.CheckState.Checked if mode_shape_plot.show_traces else Qt.CheckState.Unchecked)
        self.traces_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_traces)

        # Show parent-childs Assignm. and Show Channel-DOF Assignm. are
        # mutually exclusive with each other; Show Lines is independent of
        # both and is wired directly above, not through this group.
        self.draw_button_group = QButtonGroup()
        self.draw_button_group.setExclusive(False)
        self.draw_button_group.addButton(self.ms_checkbox, 0)
        self.draw_button_group.addButton(self.chandof_checkbox, 1)
        self.draw_button_group.idClicked.connect(self.toggle_draw)

        if mode_shape_plot.show_lines:
            self.line_checkbox.setCheckState(Qt.CheckState.Checked)
        if mode_shape_plot.show_parent_childs:
            self.ms_checkbox.setCheckState(Qt.CheckState.Checked)
        elif mode_shape_plot.show_chan_dofs:
            self.chandof_checkbox.setCheckState(Qt.CheckState.Checked)

    def _wire_mode_controls(self, mode_shape_plot, reduced_gui):
        """Populate the mode combo box; wire amplitude and real/imag controls."""
        frequencies = [
            '{}: {}'.format(i + 1, f)
            for i, f in enumerate(self.mode_shape_plot.get_frequencies())]
        if frequencies and not reduced_gui:
            self.mode_combo.addItems(frequencies)
            self.mode_combo.currentTextChanged.connect(self.change_mode)
        else:
            self.mode_combo.setEnabled(False)

        self.amplitude_box.setRange(0, 1000000000)
        self.amplitude_box.setValue(mode_shape_plot.amplitude)
        self.amplitude_box.valueChangedDelayed.connect(mode_shape_plot.change_amplitude)

        # Keep a reference: an exclusive QButtonGroup with no QObject parent
        # is only kept alive by this attribute.
        self.real_imag_group = QButtonGroup()
        self.real_imag_group.addButton(self.real_checkbox, 0)
        self.real_imag_group.addButton(self.imag_checkbox, 1)
        self.real_imag_group.setExclusive(True)
        self.imag_checkbox.setCheckState(
            Qt.CheckState.Unchecked if mode_shape_plot.real else Qt.CheckState.Checked)
        self.real_checkbox.setCheckState(
            Qt.CheckState.Checked if mode_shape_plot.real else Qt.CheckState.Unchecked)
        self.real_checkbox.stateChanged[int].connect(self.mode_shape_plot.change_part)

        self.ani_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.ani_button.released.connect(self.animate)

    def _wire_animation_widgets(self, mode_shape_plot):
        """Wire the time-history animation widgets; ani_data_button always wired."""
        if (mode_shape_plot.prep_signals is not None
                and mode_shape_plot.supports_data_animation):
            self.ani_lowpass_box.setRange(0, 1000000000)
            self.ani_lowpass_box.valueChangedDelayed.connect(self.prepare_filter)
            self.ani_highpass_box.setRange(0, 1000000000)
            self.ani_highpass_box.valueChangedDelayed.connect(self.prepare_filter)
            self.ani_speed_box.setRange(0, 1000000000)
            self.ani_speed_box.valueChanged[float].connect(self.change_animation_speed)
            self.ani_position_slider.setRange(
                0, mode_shape_plot.prep_signals.signals.shape[0])
            self.ani_position_slider.valueChanged.connect(self.set_ani_time)
        else:
            # The "Time Histories" tab has nothing to control without
            # time-domain signal data, or on a backend that cannot animate
            # it; disable it rather than leave live but effectively
            # unreachable controls.
            self.tab_2.setEnabled(False)
        self.ani_data_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.ani_data_button.setEnabled(mode_shape_plot.supports_data_animation)
        if mode_shape_plot.supports_data_animation:
            self.ani_data_button.released.connect(self.filter_and_animate_data)

    def _wire_viewport_and_limits(self):
        """Connect viewport/angle/axis-limit/zoom controls; populate self.val_widgets."""
        self.val_widgets = {}

        for button in (self.viewport_button_x, self.viewport_button_y,
                        self.viewport_button_z, self.viewport_button_iso):
            button.released.connect(self.change_viewport)

        angles = self.mode_shape_plot.get_view_angles() or (0, 0, 0)
        for angle, value, edit in zip(
                ['elev', 'az', 'roll'],
                angles,
                [self.angle_edit_elev, self.angle_edit_az, self.angle_edit_roll]):
            edit.setText(f'{value:2.0f}')
            edit.editingFinished.connect(self.change_viewport)
            self.val_widgets[angle] = edit

        if not self.mode_shape_plot.supports_axis_limits:
            # A VTK camera has no per-axis limits to edit; the backend
            # resets its own view instead.
            self._disable_axis_limit_widgets()
            return

        lims = self.mode_shape_plot.get_view_limits()
        axis_widgets = {
            'X': (self.x_limits_dec_button, self.x_limits_min_edit,
                  self.x_limits_max_edit, self.x_limits_inc_button),
            'Y': (self.y_limits_dec_button, self.y_limits_min_edit,
                  self.y_limits_max_edit, self.y_limits_inc_button),
            'Z': (self.z_limits_dec_button, self.z_limits_min_edit,
                  self.z_limits_max_edit, self.z_limits_inc_button),
        }
        for row, dir_ in enumerate(['X', 'Y', 'Z']):
            r_but, r_val, l_val, l_but = axis_widgets[dir_]
            r_val.setText(str(lims[row * 2]))
            l_val.setText(str(lims[row * 2 + 1]))
            r_but.released.connect(self.change_view)
            r_val.editingFinished.connect(self.change_view)
            l_val.editingFinished.connect(self.change_view)
            l_but.released.connect(self.change_view)
            self.val_widgets[dir_] = [r_but, r_val, l_val, l_but]

        self.zoom_plus_button.released.connect(self.change_view)
        self.zoom_minus_button.released.connect(self.change_view)
        self.reset_button.released.connect(self.reset_view)

    def _disable_axis_limit_widgets(self):
        """Grey out the axis-limit and zoom controls for limit-less backends."""
        axis_widgets = {
            'X': (self.x_limits_dec_button, self.x_limits_min_edit,
                  self.x_limits_max_edit, self.x_limits_inc_button),
            'Y': (self.y_limits_dec_button, self.y_limits_min_edit,
                  self.y_limits_max_edit, self.y_limits_inc_button),
            'Z': (self.z_limits_dec_button, self.z_limits_min_edit,
                  self.z_limits_max_edit, self.z_limits_inc_button),
        }
        for dir_, widgets in axis_widgets.items():
            for widget in widgets:
                widget.setEnabled(False)
            self.val_widgets[dir_] = list(widgets)

        for button in (self.zoom_plus_button, self.zoom_minus_button):
            button.setEnabled(False)
        self.reset_button.released.connect(self.reset_view)

    def _apply_reduced_gui(self, reduced_gui):
        """Hide the info box and its separator when reduced_gui is requested."""
        self.sep_v2.setVisible(not reduced_gui)
        self.info_box.setVisible(not reduced_gui)

    def _apply_sender_to_w_lims(self, sender, w_lims, val_widgets, hrange):
        """Mutate w_lims in-place based on the triggering widget; return updated hrange."""
        for min_max, widgets in zip(w_lims, val_widgets):
            if sender == widgets[0]:
                min_max[0] -= hrange / 3
                min_max[1] -= hrange / 3
                return hrange
            if sender == widgets[3]:
                min_max[0] += hrange / 3
                min_max[1] += hrange / 3
                return hrange
            for i, widget in enumerate(widgets[1:3]):
                if sender == widget:
                    min_max[i] = float(sender.text())
                    return min_max[1] - min_max[0]
        if sender.text() == '+':
            return hrange / 1.2
        if sender.text() == '-':
            return hrange * 1.2
        return hrange

    def reset_view(self):
        self.stop_ani()
        self.axis_checkbox.setChecked(True)
        self.nodes_checkbox.setChecked(True)
        self.line_checkbox.setChecked(True)
        self.ms_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.chandof_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.mode_shape_plot.refresh_parent_childs(False)
        self.mode_shape_plot.refresh_chan_dofs(False)
        self.mode_shape_plot.reset_view()
        self._refresh_limit_widgets()

    def _refresh_limit_widgets(self):
        """Write the backend's current axis limits into the edit fields."""
        lims = self.mode_shape_plot.get_view_limits()
        if lims is None:
            return
        for row, dir_ in enumerate(['X', 'Y', 'Z']):
            self.val_widgets[dir_][1].setText(f'{lims[row * 2]:.3f}')
            self.val_widgets[dir_][2].setText(f'{lims[row * 2 + 1]:.3f}')

    # @pyqtSlot()
    def change_view(self):
        '''
        shift the view along specified axis by +-20 % (hardcoded)
        works in combination with the appropriate buttons as senders
        or by passing one of  ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

        '''

        lims = self.mode_shape_plot.get_view_limits()
        if lims is None:
            return
        minx, maxx, miny, maxy, minz, maxz = lims
        w_lims = [[minx, maxx], [miny, maxy], [minz, maxz]]
        dx, dy, dz = (maxx - minx), (maxy - miny), (maxz - minz)
        hrange = max(dx, dy, dz)
        val_widgets = [self.val_widgets[dir_] for dir_ in ['X', 'Y', 'Z']]
        hrange = self._apply_sender_to_w_lims(self.sender(), w_lims, val_widgets, hrange)

        [[minx, maxx], [miny, maxy], [minz, maxz]] = w_lims
        xmed = maxx - (maxx - minx) / 2
        ymed = maxy - (maxy - miny) / 2
        zmed = maxz - (maxz - minz) / 2

        minx, maxx = xmed - hrange / 2, xmed + hrange / 2
        miny, maxy = ymed - hrange / 2, ymed + hrange / 2
        minz, maxz = zmed - hrange / 2, zmed + hrange / 2

        self.mode_shape_plot.set_view_limits(minx, maxx, miny, maxy, minz, maxz)

        for min_max, widgets in zip([(minx, maxx), (miny, maxy), (minz, maxz)], val_widgets):
            for val, widget in zip(min_max, widgets[1:3]):
                widget.setText(f'{val:.3f}')
        self.mode_shape_plot.redraw()

    def update_lims(self, event):
        if event.button == 3:
            self._refresh_limit_widgets()

    # @pyqtSlot()
    def change_viewport(self, viewport=None):
        '''
        change the viewport
        for non-ISO viewports the projection methods of matplotlib
        will be monkeypatched, because otherwise it would not be an
        axonometric view (functions are defined at the top of document)
        works in combination with the appropriate buttons as senders or
        by passing one of ['X', 'Y', 'Z', 'ISO']

        '''
        if viewport is None:
            viewport = self.sender().text()
        if self.sender() in self.val_widgets.values():
            az = float(self.val_widgets['az'].text())
            elev = float(self.val_widgets['elev'].text())
            roll = float(self.val_widgets['roll'].text())
            viewport = (elev, az, roll)

        self.mode_shape_plot.change_viewport(viewport)

    # @pyqtSlot()
    def save_plot(self, path=None):
        '''
        save the curently displayed frame as a graphics file
        '''
        canvas = self.canvas

        if hasattr(canvas, 'get_supported_filetypes_grouped'):
            # matplotlib FigureCanvas: offer its full list of vector/raster
            # formats. Copied and modified from NavigationToolbar2QT.
            sorted_filetypes = sorted(
                canvas.get_supported_filetypes_grouped().items())
            startpath = os.path.expanduser(rcParams.get('savefig.directory', ''))
            start = os.path.join(startpath, canvas.get_default_filename())
            filters = ';;'.join(
                '%s (%s)' % (name, " ".join('*.%s' % ext for ext in exts))
                for name, exts in sorted_filetypes)
        else:
            # A VTK/pyvista backend has no matplotlib canvas; it renders to a
            # raster, so offer a PNG screenshot instead of the mpl format list.
            start = '%s.png' % (self.mode_shape_plot.setup_name or 'mode_shape')
            filters = 'PNG image (*.png);;All Files (*)'

        fname, ext = QFileDialog.getSaveFileName(
            self, caption="Choose a filename to save to", directory=start, filter=filters)

        if fname:
            self.mode_shape_plot.save_plot(fname)
            self.statusBar().showMessage('Saved to %s' % fname, 2000)

    def save_animation(self):
        '''Export one image per animation frame of the current mode to a folder.

        Prompts for a target directory and a raster/vector format, then defers
        to the backend's ``export_animation_frames`` (works for every backend).
        '''
        if getattr(self.mode_shape_plot, 'mode_index', None) is None:
            self.statusBar().showMessage(
                'Select a mode before saving its animation.', 3000)
            return
        directory = QFileDialog.getExistingDirectory(
            self, caption='Choose a folder for the animation frames')
        if not directory:
            return
        fmt, ok = QInputDialog.getItem(
            self, 'Animation format', 'Save each frame as:',
            ['png', 'pdf'], 0, False)
        if not ok:
            return
        try:
            paths = self.mode_shape_plot.export_animation_frames(directory, fmt=fmt)
        except Exception as exc:  # keep the window alive on a backend error
            logger.warning('Animation export failed: %r', exc)
            self.statusBar().showMessage('Animation export failed: %s' % exc, 4000)
            return
        self.statusBar().showMessage(
            'Saved %d frames to %s' % (len(paths), directory), 3000)

    def plot_this(self, index):
        # self.mode_shape_plot.stop_ani()
        self.mode_shape_plot.change_mode(mode_index=index)
        # self.animate()

    # @pyqtSlot(str)
    def change_mode(self, mode):
        '''
        if user selects a new mode,
        extract the mode number from the passed string (contains frequency...)
        write modal values to the infobox
        and plot the mode shape
        '''

        # print('in change_mode: mode = ', mode)

        # mode numbering starts at 1 python lists start at 0
        mode_num = mode.split(':')  # if mode is empty
        if not mode_num[0]:
            return

        mode_num = int(float(mode_num[0])) - 1
        frequency = float(mode.split(':')[1])
        mode, order, frequency, damping, MPC, MP, MPD = self.mode_shape_plot.change_mode(
            frequency)

        text = 'Selected Mode\n'\
            +'=======================\n'\
            +'Frequency [Hz]:\t' + str(frequency) + '\n'\
            +'Damping [%]:\t' + str(damping) + '\n'\
            +'Model order:\t' + str(order) + '\n'\
            +'Mode number: \t' + str(mode) + '\n'\
            +'MPC [-]:\t' + str(MPC) + '\n'\
            +'MP  [°]:\t' + str(MP) + '\n'\
            +'MPD [-]:\t' + str(MPD) + '\n\n'
        # print(text)
        self.info_box.setText(text)

    # @pyqtSlot(int)

    def toggle_draw(self, i):
        '''
        helper function to receive the signal from the draw_button_group
        i is the number of the button that had it's state changed.
        Show parent-childs Assignm. (0) and Show Channel-DOF Assignm. (1)
        are mutually exclusive with each other; Show Lines is independent
        of this group and is wired directly to refresh_lines instead.
        '''
        self.draw_button_group.idClicked.disconnect(self.toggle_draw)
        self.mode_shape_plot.refresh_parent_childs(False)
        self.mode_shape_plot.refresh_chan_dofs(False)
        if self.draw_button_group.button(i).checkState() == Qt.CheckState.Checked:
            for j in range(2):
                if j == i:
                    continue
                self.draw_button_group.button(j).setCheckState(Qt.CheckState.Unchecked)
            if i == 0:
                self.mode_shape_plot.refresh_parent_childs(True)
            elif i == 1:
                self.mode_shape_plot.refresh_chan_dofs(True)
        self.draw_button_group.idClicked.connect(self.toggle_draw)

    # @pyqtSlot()

    def stop_ani(self):
        '''
        convenience method to stop the animation and restore the still plot
        '''
        if self.animated:
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.animated = False

    # @pyqtSlot()
    def animate(self):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        if self.mode_shape_plot.animated:
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.mode_shape_plot.stop_ani()
        else:
            if self.mode_shape_plot.data_animated:
                self.ani_data_button.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                self.mode_shape_plot.stop_ani()
            self.nodes_checkbox.setCheckState(Qt.CheckState.Unchecked)
            # self.axis_checkbox.setCheckState(Qt.CheckState.Unchecked)
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.mode_shape_plot.animate()

    def prepare_filter(self):
        lowpass = self.ani_lowpass_box.value()
        highpass = self.ani_highpass_box.value()
        # print(lowpass, highpass)
        try:
            lowpass = float(lowpass)
        except ValueError:
            lowpass = None
        try:
            highpass = float(highpass)
        except ValueError:
            highpass = None

        if lowpass == 0.0:
            lowpass = None
        if highpass == 0.0:
            highpass = None
        if lowpass and highpass:
            if lowpass <= highpass:
                raise ValueError(
                    f"lowpass ({lowpass}) must be greater than highpass ({highpass})")
        # print(highpass, lowpass)
        self.mode_shape_plot.prep_signals.filter_signals(lowpass, highpass)

    def set_ani_time(self, pos):
        # print(pos)
        tot_len = self.mode_shape_plot.prep_signals.signals.shape[0]
        # pos = int(pos*tot_len)
        self.mode_shape_plot.line_ani.frame_seq = iter(range(pos, tot_len))

    def change_animation_speed(self, speed):
        try:
            speed = float(speed)
        except ValueError:
            return

        # print(speed)
        self.mode_shape_plot.line_ani.event_source.interval = int(speed)
        self.mode_shape_plot.line_ani.event_source._timer_set_interval()

    # @pyqtSlot()
    def filter_and_animate_data(self):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        if self.mode_shape_plot.data_animated:
            self.ani_data_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.mode_shape_plot.stop_ani()
        else:
            if self.mode_shape_plot.animated:
                self.ani_button.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                self.mode_shape_plot.stop_ani()
            self.nodes_checkbox.setCheckState(Qt.CheckState.Unchecked)
            self.axis_checkbox.setCheckState(Qt.CheckState.Unchecked)
            self.ani_data_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.mode_shape_plot.filter_and_animate_data(
                callback=self.ani_position_data.setText)

    def closeEvent(self, *args, **kwargs):
        self.mode_shape_plot.stop_ani()
        self.deleteLater()
        return QMainWindow.closeEvent(self, *args, **kwargs)


def start_msh_gui(mode_shape_plot):

    def _handler(msg_type, msg_string):
        pass

    # qInstallMessageHandler(_handler)  # suppress unimportant error msg
    global app
    app = QApplication.instance() or QApplication(sys.argv)

    form = ModeShapeGUI(mode_shape_plot)

    loop = QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec()
    return


if __name__ == "__main__":
    pass
