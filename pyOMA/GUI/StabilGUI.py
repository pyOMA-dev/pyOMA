# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""PyQt6 interactive stabilization diagram and mode-selection GUI."""
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

app = None

from .HelpersGUI import my_excepthook, UnsavedChangesMixin
from .PlotMSHGUI import ModeShapeGUI
from .generated.ui_stabil_gui import Ui_StabilGUI
from .generated.ui_complex_plot import Ui_ComplexPlot
from .generated.ui_histo_plot import Ui_HistoPlot
from pyOMA.core.StabilDiagram import StabilPlot, StabilCluster, StabilCalc
from pyOMA.core import resolve_mode_shape_backend
from PyQt6.QtCore import Qt, pyqtSlot, QEventLoop
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication, QMessageBox
import numpy as np
import sys
import os

from matplotlib import rcParams
from matplotlib import ticker

from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
import matplotlib.pyplot as plot

plot.rc('figure', figsize=[8.5039399474194, 5.255723925793184], dpi=100,)
plot.rc('font', size=10)
plot.rc('legend', fontsize=10, labelspacing=0.1)
plot.rc('axes', linewidth=0.2)
plot.rc('xtick.major', width=0.2)
plot.rc('ytick.major', width=0.2)
plot.ioff()

sys.excepthook = my_excepthook


class StabilGUI(UnsavedChangesMixin, QMainWindow, Ui_StabilGUI):
    """PyQt6 main window for interactive stabilisation diagram and mode selection.

    Displays a :class:`~pyOMA.core.StabilDiagram.StabilPlot` with interactive
    pole-picking, PSD overlay, adjustable stabilisation criteria, and optionally
    a linked :class:`~pyOMA.GUI.PlotMSHGUI.ModeShapeGUI` for immediate mode-
    shape feedback.

    Parameters
    ----------
    stabil_plot : StabilPlot
        Populated stabilisation-diagram plot object.
    cmpl_plot : StabilPlot or similar
        Complementary plot shown beside the stabilisation diagram (e.g. a
        PSD or correlation plot).
    msh_plot : ModeShapeGUI, optional
        When provided, mode-shape is updated whenever the user selects a pole.

    .. TODO::
        * scale markers right on every platform
        * frequency range as argument or from ssi params, sampling freq
        * add switch to choose between "unstable only in ..." or "stable in ..."
        * distinguish between stabilization criteria and filtering criteria
        * add zoom and sliders (horizontal/vertical) for the main figure
    """

    def __init__(self, stabil_plot, cmpl_plot, msh_plot=None):
        """
        Parameters
        ----------
        stabil_plot : StabilPlot
            Populated stabilisation-diagram plot object.
        cmpl_plot : StabilPlot or similar
            Complementary plot shown beside the stabilisation diagram.
        msh_plot : ModeShapeGUI, optional
            When provided, mode-shape is updated on pole selection.
        """

        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle(
            'Stabilization Diagram: {} - {}'.format(
                stabil_plot.stabil_calc.setup_name,
                stabil_plot.stabil_calc.start_time))

        self.stabil_plot = stabil_plot
        self.stabil_calc = stabil_plot.stabil_calc
        if self.stabil_calc.state < 2:
            self.stabil_calc.calculate_stabilization_masks()

        self.cmpl_plot = cmpl_plot
        self.msh_plot = msh_plot
        self.current_mode = (0, 0)

        self.histo_plot_f = None
        self.histo_plot_sf = None
        self.histo_plot_d = None
        self.histo_plot_sd = None
        self.histo_plot_dr = None
        self.histo_plot_mac = None
        self.histo_plot_mpc = None
        self.histo_plot_mpd = None

        self._wire_menu()
        self._wire_canvas()
        self._wire_mode_display_widgets(cmpl_plot)
        self._wire_stab_val_widget()
        self._wire_diag_val_widget()
        self._wire_buttons()
        self._wire_panel_toggles()

        self.stabil_calc.add_callback('add_mode', self.mode_selector_add)
        self.stabil_calc.add_callback('remove_mode', self.mode_selector_take)

        for index, mode in enumerate(self.stabil_calc.select_modes):
            self.mode_selector_add(mode, index)
        self._dirty = False  # reflects modes already selected before this GUI opened

        self.show()

        for widg in [self.mode_val_view_text, self.current_value_view_text]:
            widg.setText('\n \n \n \n \n \n \n')
            height = widg.document().size().toSize().height() + 3
            widg.setFixedHeight(height)
        self.update_stabil_view()

    def _wire_menu(self):
        """Connect the .ui-declared menu actions to their slots.

        action_save_plot has never been wired to anything (pre-existing,
        left as-is - not something this migration should fix).
        """
        self.action_quit.triggered.connect(self.close)
        self.actionSave_State.triggered.connect(self.save_state)
        self.actionLoad_State.triggered.connect(self.load_state)

    def _wire_canvas(self):
        """Attach stabil_plot's figure to the canvas and set up the pole cursor."""
        self.fig = self.stabil_plot.fig
        self.canvas.set_figure(self.fig)
        self.stabil_plot.fig.set_facecolor('none')
        self.init_cursor()

    def _wire_mode_display_widgets(self, cmpl_plot):
        """Wire the mode selector, mode-shape-viewer toggle, and mode display views."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.transparent)
        self.mode_val_view_text.setPalette(palette)
        self.current_value_view_text.setPalette(palette)

        self.mode_selector.currentIndexChanged[int].connect(self.update_mode_val_view)
        self.msh_toggle_button.toggled.connect(self.toggle_msh_plot)
        self.msh_toggle_button.setEnabled(self.msh_plot is not None)

        cmpl_plot.plot_diagram()
        self._embed_cmpl_plot(cmpl_plot)

    def _embed_cmpl_plot(self, cmpl_plot):
        """Embed the ComplexPlot window as a permanent widget in the right pane."""
        cmpl_plot.setParent(self.right_pane_widget)
        cmpl_plot.setWindowFlags(Qt.WindowType.Widget)
        self.right_pane_layout.insertWidget(1, cmpl_plot, 1)
        cmpl_plot.show()

    def _wire_stab_val_widget(self):
        """Populate and wire the 'Stabilization Criteria' frame's widgets."""
        caps = self.stabil_calc.capabilities

        self.df_edit.setText(str(self.stabil_calc.df_max * 100))
        self.df_histo_button.released.connect(self.create_histo_plot_f)
        self.stdf_histo_button.released.connect(self.create_histo_plot_sf)

        self.dd_edit.setText(str(self.stabil_calc.dd_max * 100))
        self.dd_histo_button.released.connect(self.create_histo_plot_d)
        self.stdd_histo_button.released.connect(self.create_histo_plot_sd)

        d_range = self.stabil_calc.d_range
        self.d_min_edit.setText(str(d_range[0]))
        self.d_max_edit.setText(str(d_range[1]))
        self.dr_histo_button.released.connect(self.create_histo_plot_dr)

        self.mac_edit.setText(str(self.stabil_calc.dmac_max * 100))
        self.mac_histo_button.released.connect(self.create_histo_plot_mac)

        self.mpc_edit.setText(str(self.stabil_calc.mpc_min))
        self.mpc_histo_button.released.connect(self.create_histo_plot_mpc)
        self.mpd_edit.setText(str(self.stabil_calc.mpd_max))
        self.mpd_histo_button.released.connect(self.create_histo_plot_mpd)

        self.show_mc_button.released.connect(self.show_MC_plot)

        if caps['auto']:
            self.num_iter_edit.setText(str(self.stabil_calc.num_iter))
            self.threshold_edit.setPlaceholderText(str(self.stabil_calc.threshold))
        self.clear_auto_button.released.connect(self.prepare_auto_clearing)
        self.classify_auto_button.released.connect(self.prepare_auto_classification)
        self.select_auto_button.released.connect(self.prepare_auto_selection)

        self._apply_stab_val_visibility(caps)

    def _apply_stab_val_visibility(self, caps):
        """Show/hide capability-gated rows of the 'Stabilization Criteria' frame.

        A static form can't express the original code's conditional row
        construction, so every row is always built and toggled instead.
        """
        self._set_visible(caps['f'], self.freq_label, self.df_edit, self.df_histo_button)
        self._set_visible(
            caps['std'],
            self.cov_freq_label, self.stdf_edit, self.stdf_histo_button,
            self.cov_damping_label, self.stdd_edit, self.stdd_histo_button)
        self._set_visible(
            caps['d'],
            self.damping_label, self.dd_edit, self.dd_histo_button,
            self.damping_range_label, self.d_min_edit, self.damping_range_to_label,
            self.d_max_edit, self.dr_histo_button)
        self._set_visible(
            caps['msh'],
            self.mac_label, self.mac_edit, self.mac_histo_button,
            self.mpc_label, self.mpc_edit, self.mpc_histo_button,
            self.mpd_label, self.mpd_edit, self.mpd_histo_button)
        # MTN was never finished in pyOMA/core (capabilities['mtn'] is
        # hardcoded off there) and this button has no method to call -
        # keep it permanently hidden rather than wiring a dangling signal.
        self._set_visible(False, self.mtn_label, self.mtn_edit, self.mtn_histo_button)
        self._set_visible(caps['MC'], self.mc_label, self.MC_edit, self.show_mc_button)
        self._set_visible(
            caps['auto'],
            self.clear_auto_button, self.num_iter_edit,
            self.classify_auto_button, self.use_stabil_checkbox, self.threshold_edit,
            self.select_auto_button, self.num_modes_edit)

    def _wire_diag_val_widget(self):
        """Populate and wire the 'View Settings' frame's widgets."""
        caps = self.stabil_calc.capabilities

        self.stabil_plot.toggle_stable(True)
        self.stable_pole_checkbox.stateChanged.connect(self.stabil_plot.toggle_stable)
        self.snap_stable_radio.toggled.connect(self.snap_stable)

        self.stabil_plot.toggle_all(True)
        self.all_poles_checkbox.stateChanged.connect(self.stabil_plot.toggle_all)
        self.snap_all_radio.toggled.connect(self.snap_all)

        if caps['auto']:
            show_clear = self.stabil_calc.state >= 3
            self.autoclear_checkbox.setChecked(show_clear)
            self.stabil_plot.toggle_clear(show_clear)
            show_select = self.stabil_calc.state >= 4
            self.autoselect_checkbox.setChecked(show_select)
            self.stabil_plot.toggle_select(show_select)
        self.autoclear_checkbox.stateChanged.connect(self.stabil_plot.toggle_clear)
        self.snap_clear_radio.toggled.connect(self.snap_clear)
        self.autoselect_checkbox.stateChanged.connect(self.stabil_plot.toggle_select)
        self.snap_select_radio.toggled.connect(self.snap_select)

        if caps['data']:
            self.stabil_plot.plot_sv_psd(False)
            self.show_psd_checkbox.stateChanged.connect(self.stabil_plot.plot_sv_psd)

        if caps['std']:
            self.show_stdf_checkbox.stateChanged.connect(self.stabil_plot.toggle_stdf)

        f_range = (0, self.stabil_calc.get_max_f())
        self.freq_low_edit.setText('{:2.3f}'.format(f_range[0]))
        self.freq_high_edit.setText('{:2.3f}'.format(f_range[1] * 1.05))

        n_range = (0, 1, self.stabil_calc.modal_data.max_model_order)
        self.n_low_edit.setText('{:2d}'.format(n_range[0]))
        self.n_step_edit.setText('{:2d}'.format(n_range[1]))
        self.n_high_edit.setText('{:2d}'.format(n_range[2]))

        self._apply_diag_val_visibility(caps)

    def _apply_diag_val_visibility(self, caps):
        """Show/hide capability-gated rows of the 'View Settings' frame."""
        self._set_visible(
            caps['auto'],
            self.autoclear_checkbox, self.snap_clear_radio,
            self.autoselect_checkbox, self.snap_select_radio)
        self._set_visible(caps['data'], self.show_psd_checkbox)
        self._set_visible(caps['std'], self.show_stdf_checkbox)

    @staticmethod
    def _set_visible(visible, *widgets):
        for widget in widgets:
            widget.setVisible(visible)

    def _wire_buttons(self):
        """Connect the bottom button row to their slots."""
        self.save_figure_button.released.connect(self.save_figure)
        self.export_results_button.released.connect(self.save_results)
        self.ok_close_button.released.connect(self.close)

    def _wire_panel_toggles(self):
        """Wire the left/right pane collapse toggle buttons."""
        self.left_toggle_btn.toggled.connect(
            self._make_toggle_handler(self.left_toggle_btn, self.left_pane_widget))
        self.right_toggle_btn.toggled.connect(
            self._make_toggle_handler(self.right_toggle_btn, self.right_pane_widget))

    def _make_toggle_handler(self, button, panel):
        def handler(checked):
            panel.setVisible(checked)
            button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        return handler

    def create_histo_plot_f(self):
        '''
        create
        show/hide
        update stable
        '''
        # print('here')
        array = self.stabil_calc.freq_diffs
        self.histo_plot_f = self.create_histo_plot(
            array,
            self.histo_plot_f,
            title='Frequency Differences (percent)',
            ranges=(
                0,
                1),
            select_ranges=[
                float(
                    self.df_edit.text()) /
                100],
            select_callback=[
                lambda x: [
                    self.df_edit.setText(
                        str(
                            x *
                            100)),
                    self.update_stabil_view()]])

    def create_histo_plot_sf(self):
        '''
        create
        show/hide
        update stable
        '''
        array = np.ma.array(
            self.stabil_calc.modal_data.std_frequencies /
            self.stabil_calc.modal_data.modal_frequencies)
        self.histo_plot_sf = self.create_histo_plot(
            array,
            self.histo_plot_sf,
            title='CoV Frequencies (percent)',
            ranges=(
                0,
                1),
            select_ranges=[
                float(
                    self.stdf_edit.text()) /
                100],
            select_callback=[
                lambda x: [
                    self.stdf_edit.setText(
                        str(
                            x *
                            100)),
                    self.update_stabil_view()]])

    def create_histo_plot_d(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.damp_diffs
        self.histo_plot_d = self.create_histo_plot(
            array,
            self.histo_plot_d,
            title='Damping Differences (percent)',
            ranges=(
                0,
                1),
            select_ranges=[
                float(
                    self.dd_edit.text()) /
                100],
            select_callback=[
                lambda x: [
                    self.dd_edit.setText(
                        str(
                            x *
                            100)),
                    self.update_stabil_view()]])

    def create_histo_plot_sd(self):
        '''
        create
        show/hide
        update stable
        '''
        array = np.ma.array(
            self.stabil_calc.modal_data.std_damping /
            self.stabil_calc.modal_data.modal_damping)
        self.histo_plot_sd = self.create_histo_plot(
            array,
            self.histo_plot_sd,
            title='CoV Damping (percent)',
            ranges=(
                0,
                1),
            select_ranges=[
                float(
                    self.stdd_edit.text()) /
                100],
            select_callback=[
                lambda x: [
                    self.stdd_edit.setText(
                        str(
                            x *
                            100)),
                    self.update_stabil_view()]])

    def create_histo_plot_dr(self):
        '''
        create
        show/hide
        update stable
        '''
        array = np.ma.array(self.stabil_calc.modal_data.modal_damping)
        self.histo_plot_dr = self.create_histo_plot(
            array, self.histo_plot_dr, title='Damping range ', ranges=(
                0, 10), select_ranges=[
                float(
                    self.d_min_edit.text()), float(
                    self.d_max_edit.text())], select_callback=[
                        lambda x: [
                            self.d_min_edit.setText(
                                str(x)), self.update_stabil_view()], lambda x: [
                                    self.d_max_edit.setText(
                                        str(x)), self.update_stabil_view()]])

    def create_histo_plot_mac(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.MAC_diffs
        self.histo_plot_mac = self.create_histo_plot(
            array,
            self.histo_plot_mac,
            title='MAC Diffs (percent)',
            ranges=(
                0,
                1),
            select_ranges=[
                float(
                    self.mac_edit.text()) /
                100],
            select_callback=[
                lambda x: [
                    self.mac_edit.setText(
                        str(
                            x *
                            100)),
                    self.update_stabil_view()]])

    def create_histo_plot_mpc(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.MPC_matrix
        self.histo_plot_mpc = self.create_histo_plot(
            array, self.histo_plot_mpc, title='MPC', ranges=(
                0, 1), select_ranges=[
                float(
                    self.mpc_edit.text()), None], select_callback=[
                    lambda x: [
                        self.mpc_edit.setText(
                            str(x)), self.update_stabil_view()], str])

    def create_histo_plot_mpd(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.MPD_matrix
        self.histo_plot_mpd = self.create_histo_plot(
            array, self.histo_plot_mpd, title='MPD', ranges=(
                0, 90), select_ranges=[
                float(
                    self.mpd_edit.text())], select_callback=[
                    lambda x: [
                        self.mpd_edit.setText(
                            str(x)), self.update_stabil_view()]])

    def show_MC_plot(self):

        b = self.sender().isChecked()
        self.stabil_plot.show_MC(b)

    def create_histo_plot(
            self,
            array,
            plot_obj,
            title='',
            ranges=None,
            select_ranges=None,
            select_callback=None):
        '''
        should work like following::

            button press    if None        → visible = True, create
                            if visible     → visible = False, hide
                            if not visible → visible = True, show
            update stabil   if None        → skip
                            if visible     → visible = visible, update
                            if not visible → visible = visible, update
            close button                   → visible = False, hide


        but doesn't, since the function can not distinguish between
        "button press" and "update stabil"

        '''
        if select_ranges is None:
            select_ranges = [None]
        if select_callback is None:
            select_callback = [None]
        old_mask = np.copy(array.mask)
        array.mask = np.ma.nomask

        mask_stable = self.stabil_calc.get_stabilization_mask('mask_stable')
        if len(array.shape) == 3:
            mask = array == 0
            array.mask = mask
            a = np.min(array, axis=2)
            a.mask = mask_stable
            stable_data = a.compressed()
        else:
            array.mask = mask_stable
            stable_data = array.compressed()

        if plot_obj is None:  # create
            # print('here again')
            mask_pre = self.stabil_calc.get_stabilization_mask('mask_pre')

            if len(array.shape) == 3:
                mask = array == 0
                array.mask = mask
                a = np.min(array, axis=2)
                a.mask = mask_pre
                all_data = a.compressed()
            else:
                array.mask = mask_pre
                all_data = array.compressed()

            plot_obj = HistoPlot(
                all_data,
                stable_data,
                title,
                ranges,
                select_ranges=select_ranges,
                select_callback=select_callback)
        else:  # update
            plot_obj.update_histo(stable_data, select_ranges)

        array.mask = old_mask

        # show (hide is accomplished by just closing the window closeEvent was overridden to self.hide() )
        # print('show')
        if plot_obj.visible:
            plot_obj.show()
        return plot_obj

    def _format_modal_values(self, i):
        '''Render the modal values of pole *i* as text for one of the value views.

        Parameters
        ----------
            i: int
                Index of the pole in the stabilization diagram.

        Returns
        -------
            s: str
                One line per modal value that is not NaN.
            mp: float
                Mean phase, which :meth:`update_mode_val_view` passes on to
                :meth:`update_mode_plot`.
        '''
        n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
            i)
        if self.stabil_calc.capabilities['std']:
            import scipy.stats
            num_blocks = self.stabil_calc.modal_data.num_blocks
            stdf = scipy.stats.t.ppf(
                0.975, num_blocks) * stdf / np.sqrt(num_blocks)
            stdd = scipy.stats.t.ppf(
                0.975, num_blocks) * stdd / np.sqrt(num_blocks)

        s = ''
        for text, val in [('Frequency=%1.3fHz, \n' % (f), f),
                          ('CI Frequency ± %1.3e, \n' % (stdf), stdf),
                          ('Order=%1.0f, \n' % (n), n),
                          ('Damping=%1.3f%%,  \n' % (d), d),
                          ('CI Damping ± %1.3e,  \n' % (stdd), stdd),
                          ('MPC=%1.5f, \n' % (mpc), mpc),
                          ('MP=%1.3f°, \n' % (mp), mp),
                          ('MPD=%1.5f°, \n' % (mpd), mpd),
                          ('dMP=%1.3f°, \n' % (dmp), dmp),
                          # ('dMPD=%1.5f°, \n' % (dmpd),     dmpd),
                          ('MTN=%1.5f, \n' % (mtn), mtn),
                          ('MC=%1.5f, \n' % (MC), MC),
                          ('Ext=%1.5f°, \n' % (ex_1), ex_1),
                          ('Ext=%1.3f°, \n' % (ex_2), ex_2)
                          ]:
            if val is not np.nan:
                s += text
        return s, mp

    @staticmethod
    def _set_view_text(view, s):
        '''Show *s* in *view* and shrink the widget to its content height.'''
        view.setText(s)
        view.setFixedHeight(view.document().size().toSize().height() + 3)

    def update_mode_val_view(self, index):
        # display information about currently selected mode
        i = self.stabil_calc.select_modes[index]
        self.current_mode = i
        s, mp = self._format_modal_values(i)
        self._set_view_text(self.mode_val_view_text, s)
        self.update_mode_plot(i, mp)

    @pyqtSlot(tuple)
    def mode_selector_add(self, i, index):
        # add mode tomode_selector and select it
        self._dirty = True
        _, f, *_ = self.stabil_calc.get_modal_values(i)
        # index = self.stabil_calc.select_modes.index(i)
        # print(n,f,d,mpc, mp, mpd)
        # print(index)
        text = '{} - {:2.3f}'.format(index, f)
        self.mode_selector.currentIndexChanged[int].disconnect(self.update_mode_val_view)
        self.mode_selector.addItem(text)

        self.mode_selector.setCurrentIndex(index)
        self.update_mode_val_view(index)
        self.mode_selector.currentIndexChanged[
            int].connect(self.update_mode_val_view)

    @pyqtSlot(tuple)
    def mode_selector_take(self, i_, index):
        self._dirty = True
        if self.current_mode == i_:
            if self.stabil_calc.select_modes:
                self.current_mode = self.stabil_calc.select_modes[0]
            else:
                self.current_mode = (0, 0)
        self.mode_selector.currentIndexChanged[
            int].disconnect(self.update_mode_val_view)
        self.mode_selector.clear()

        for index, i in enumerate(self.stabil_calc.select_modes):
            _, f, *_ = self.stabil_calc.get_modal_values(i)
            text = '{} - {:2.3f}'.format(index, f)
            self.mode_selector.addItem(text)
            if self.current_mode == i:
                for ind in range(self.mode_selector.count()):
                    if text == self.mode_selector.itemText(ind):
                        break
            else:
                ind = 0

        if self.mode_selector.count():
            self.mode_selector.setCurrentIndex(ind)
            self.update_mode_val_view(ind)
        self.mode_selector.currentIndexChanged[
            int].connect(self.update_mode_val_view)

    def update_mode_plot(self, i, mpd=None):
        # update the plot of the currently selected mode
        msh = self.stabil_calc.get_mode_shape(i)
        self.cmpl_plot.scatter_this(msh, mpd)
        # time.sleep(1)
        if self.msh_plot is not None:
            self.msh_plot.plot_this(i)

    def toggle_msh_plot(self, b):
        # change the type of mode plot
        # print('msh',b)
        if b:
            self.msh_plot.show()
        else:
            self.msh_plot.hide()

    def init_cursor(self):
        self.cursor = self.stabil_plot.init_cursor()
        self.cursor.add_callback('show_current_info', self.update_value_view)

    # @pyqtSlot(bool)
    def snap_frequency(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_df')
            self.cursor.set_mask(mask, 'mask_df')

    # @pyqtSlot(bool)
    def snap_damping(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_dd')
            self.cursor.set_mask(mask, 'mask_dd')

    # @pyqtSlot(bool)
    def snap_vector(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_dmac')
            self.cursor.set_mask(mask, 'mask_dmac')

    # @pyqtSlot(bool)
    def snap_stable(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_stable')
            self.cursor.set_mask(mask, 'mask_stable')

    # @pyqtSlot(bool)
    def snap_all(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_pre')
            self.cursor.set_mask(mask, 'mask_pre')

    # @pyqtSlot(bool)
    def snap_clear(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_autoclear')
            self.cursor.set_mask(mask, 'mask_autoclear')

    # @pyqtSlot(bool)
    def snap_select(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_autoselect')
            self.cursor.set_mask(mask, 'mask_autoselect')

    def update_value_view(self, i):
        self.current_mode = i
        s, _mp = self._format_modal_values(i)
        self._set_view_text(self.current_value_view_text, s)

    def _collect_stabil_params(self):
        """Read stabilization-criterion values from the UI edit boxes."""
        caps = self.stabil_calc.capabilities
        stdf_max = float(self.stdf_edit.text()) / 100 if caps['std'] else None
        stdd_max = float(self.stdd_edit.text()) / 100 if caps['std'] else None
        dmac_max = float(self.mac_edit.text()) / 100 if caps['msh'] else None
        mpc_min = float(self.mpc_edit.text()) if caps['msh'] else None
        mpd_max = float(self.mpd_edit.text()) if caps['msh'] else None
        MC_min = float(self.MC_edit.text()) if caps['MC'] else None
        return dict(
            df_max=float(self.df_edit.text()) / 100,
            stdf_max=stdf_max,
            dd_max=float(self.dd_edit.text()) / 100,
            stdd_max=stdd_max,
            dmac_max=dmac_max,
            d_range=(float(self.d_min_edit.text()), float(self.d_max_edit.text())),
            mpc_min=mpc_min,
            mpd_max=mpd_max,
            MC_min=MC_min,
            order_range=(
                int(self.n_low_edit.text()), int(self.n_step_edit.text()), int(self.n_high_edit.text())),
        )

    def _refresh_histos(self):
        """Redraw any open histogram windows."""
        for attr, creator in [
            ('histo_plot_f', self.create_histo_plot_f),
            ('histo_plot_sf', self.create_histo_plot_sf),
            ('histo_plot_d', self.create_histo_plot_d),
            ('histo_plot_sd', self.create_histo_plot_sd),
            ('histo_plot_dr', self.create_histo_plot_dr),
            ('histo_plot_mac', self.create_histo_plot_mac),
            ('histo_plot_mpc', self.create_histo_plot_mpc),
            ('histo_plot_mpd', self.create_histo_plot_mpd),
        ]:
            if getattr(self, attr) is not None:
                creator()

    def update_stabil_view(self):
        params = self._collect_stabil_params()
        f_range = (float(self.freq_low_edit.text()), float(self.freq_high_edit.text()))
        order_range = params['order_range']
        self.stabil_plot.update_stabilization(**params)
        self.stabil_plot.update_xlim(f_range)
        self.stabil_plot.update_ylim((order_range[0], order_range[2]))
        self._refresh_histos()

    def prepare_auto_clearing(self):
        if not self.stabil_calc.capabilities['auto']:
            raise RuntimeError("Automatic clearing requires a StabilCalc with 'auto' capabilities.")
        num_iter = int(self.num_iter_edit.text())

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_clearing(num_iter)
            self.threshold_edit.setPlaceholderText(
                str(self.stabil_calc.threshold))
            self.stabil_plot.update_stabilization()
            self.autoclear_checkbox.setChecked(True)

    def prepare_auto_classification(self):
        if not self.stabil_calc.capabilities['auto']:
            raise RuntimeError("Automatic classification requires a StabilCalc with 'auto' capabilities.")
        use_stabil = self.use_stabil_checkbox.isChecked()

        threshold = self.threshold_edit.text()
        # print(threshold, type(threshold), threshold.isnumeric())
        if threshold.isnumeric():
            threshold = float(threshold)
        else:
            threshold = None

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_classification(threshold, use_stabil)
            self.autoselect_checkbox.setChecked(True)

    def prepare_auto_selection(self):
        if not self.stabil_calc.capabilities['auto']:
            raise RuntimeError("Automatic selection requires a StabilCalc with 'auto' capabilities.")
        num_modes = self.num_modes_edit.text()
        if num_modes.isnumeric():
            num_modes = int(num_modes)
        else:
            num_modes = 0

        for datapoint in reversed(self.stabil_calc.select_modes):
            self.stabil_calc.remove_mode(datapoint)
        print(self.stabil_calc.select_modes)

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_selection(num_modes)
            # self.stabil_plot.update_stabilization()
            # self.stabil_calc.plot_selection()

    def save_figure(self, fname=None):

        # copied and modified from
        # matplotlib.backends.backend_qt4.NavigationToolbar2QT
        canvas = self.stabil_plot.ax.figure.canvas

        filetypes = canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        # default_filetype = canvas.get_default_filetype()

        startpath = rcParams.get('savefig.directory', '')
        startpath = os.path.expanduser(startpath)
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter_ = '%s (%s)' % (name, exts_list)
            filters.append(filter_)
        filters = ';;'.join(filters)

        if fname is None:
            fname, ext = QFileDialog.getSaveFileName(
                self, caption="Choose a filename to save to", directory=start, filter=filters)
            # print(fname)
        self.stabil_plot.save_figure(fname)

    def save_results(self):

        fname, fext = QFileDialog.getSaveFileName(self, caption="Choose a directory to save to",
                                                  directory=os.getcwd(), filter='Text File (*.txt)')
        # fname, fext = os.path.splitext(fname)

        if fext != 'txt':
            fname += '.txt'

        self.stabil_calc.export_results(fname)

    def _save_to_file(self):
        """Prompt for a filename and save stabil_calc state; return True if saved."""
        fname, _ = QFileDialog.getSaveFileName(self, caption="Choose a directory to save to",
                                                  directory=os.getcwd(), filter='Numpy Archive File (*.npz)')

        if fname == '':
            return False
        fname, fext = os.path.splitext(fname)

        if fext != 'npz':
            fname += '.npz'

        self.stabil_calc.save_state(fname)
        self._dirty = False
        return True

    def save_state(self):
        if self._save_to_file():
            self.close()

    def _do_save(self):
        self._save_to_file()

    def load_state(self):

        fname, _ = QFileDialog.getOpenFileName(self, caption="Choose a state file to load",
                                                 directory=os.getcwd(), filter='Numpy Archive File (*.npz)')
        if fname == '':
            return

        try:
            stabil_calc = StabilCalc.load_state(fname, self.stabil_calc.modal_data)
        except Exception as exc:
            logger.exception("load_state failed")
            QMessageBox.warning(self, "Load failed", str(exc))
            return

        if stabil_calc is None:
            QMessageBox.warning(self, "Load failed", f"No stabilization state found in {fname!r}.")
            return

        self.stabil_calc = stabil_calc
        self.stabil_plot.stabil_calc = stabil_calc
        self.mode_selector.clear()
        self.stabil_calc.add_callback('add_mode', self.mode_selector_add)
        self.stabil_calc.add_callback('remove_mode', self.mode_selector_take)
        for index, mode in enumerate(self.stabil_calc.select_modes):
            self.mode_selector_add(mode, index)
        self._dirty = False  # freshly loaded from disk - nothing unsaved yet
        self.update_stabil_view()

    def closeEvent(self, event):
        if not self._prompt_save_on_close(event):
            return

        if self.msh_plot is not None:
            self.msh_plot.mode_shape_plot.stop_ani()

        # self.stabil_calc.select_modes = self.stabil_calc.select_modes
        self.deleteLater()

        return QMainWindow.closeEvent(self, event)

    def keyPressEvent(self, e):
        # print(e.key())
        if e.key() == Qt.Key.Key_Return or e.key() == Qt.Key.Key_Enter:
            self.update_stabil_view()
            # print('2')
        super().keyPressEvent(e)


class ComplexPlot(QMainWindow, Ui_ComplexPlot):

    def __init__(self):

        QMainWindow.__init__(self)
        self.setupUi(self)
        self.fig = Figure(facecolor='white', dpi=100, figsize=(4, 4))
        self.canvas.set_figure(self.fig)

    @staticmethod
    def _normalize_mp_angle(mp):
        """Normalize mp from degrees to radians in [0, 2π]."""
        while mp < 0:
            mp += 180
        while mp > 360:
            mp -= 360
        return mp * np.pi / 180

    @staticmethod
    def _compute_mp_endpoints(mp):
        """Return (x1, x2, y1, y2) for a line at angle mp (radians) inside [-1, 1]²."""
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
        t = np.tan(mp)
        if mp <= np.pi / 2:
            x1 = max(xmin, ymin / t)
            x2 = min(xmax, ymax / t)
            y1 = max(ymin, xmin * t)
            y2 = min(ymax, xmax * t)
        elif mp <= np.pi:
            x1 = max(xmin, ymax / t)
            x2 = min(xmax, ymin / t)
            y2 = max(ymin, xmax * t)
            y1 = min(ymax, xmin * t)
        elif mp <= 3 * np.pi / 2:
            x1 = max(xmin, ymin / t)
            x2 = min(xmax, ymax / t)
            y1 = max(ymin, xmin * t)
            y2 = min(ymax, xmax * t)
        else:
            x1 = max(xmin, ymax / t)
            x2 = min(xmax, ymin / t)
            y2 = max(ymin, xmax * t)
            y1 = min(ymax, xmin * t)
        return x1, x2, y1, y2

    def scatter_this(self, msh, mp=None):
        self.ax.cla()
        self.ax.scatter(msh.real, msh.imag)
        if mp is not None:
            mp = self._normalize_mp_angle(mp)
            x1, x2, y1, y2 = self._compute_mp_endpoints(mp)
            self.ax.plot([x1, x2], [y1, y2])
        lim = max(max(abs(msh.real)) * 1.1, max(abs(msh.imag)) * 1.1)
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_position(('data', 0 - 1))
        self.ax.spines['top'].set_position(('data', 0 - 1))
        for side in ['right', 'top']:
            self.ax.spines[side].set_color('none')
        for axis, _ in zip([self.ax.xaxis, self.ax.yaxis], [0, 0]):
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_formatter(ticker.NullFormatter())
        self.fig.canvas.draw_idle()

    def plot_diagram(self):

        self.fig.set_tight_layout(True)
        self.ax = self.fig.add_subplot(111)
        self.ax.autoscale_view(tight=True)
        # Set the axis's spines to be centered at the given point
        # (Setting all 4 spines so that the tick marks go in both directions)
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_position(('data', 0 - 1))
        self.ax.spines['top'].set_position(('data', 0 - 1))

        self.ax.xaxis.set_label_text('Re')
        self.ax.yaxis.set_label_text('Im')

        # Hide the line (but not ticks) for "extra" spines
        for side in ['right', 'top']:
            self.ax.spines[side].set_color('none')

        # On both the x and y axes...
        for axis, _ in zip([self.ax.xaxis, self.ax.yaxis], [0, 0]):
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_formatter(ticker.NullFormatter())
        self.fig.canvas.draw_idle()


class HistoPlot(QMainWindow, Ui_HistoPlot):

    def __init__(
            self,
            all_data,
            stabil_data,
            title='',
            ranges=None,
            select_ranges=None,
            select_callback=None):
        if select_ranges is None:
            select_ranges = [None]
        if select_callback is None:
            select_callback = [None]
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)

        if ranges is None:
            ranges = (all_data.min(), all_data.max())
        step = (ranges[1] - ranges[0]) / 50
        self.lrange_box.setSingleStep(step)
        self.lrange_box.setValue(ranges[0])
        self.lrange_box.valueChangedDelayed.connect(self.update_range)

        self.urange_box.setSingleStep(step)
        self.urange_box.setValue(ranges[1])
        self.urange_box.valueChangedDelayed.connect(self.update_range)

        self.axes = self.canvas.axes
        self.main_widget.setFocus()
        self.all_data = np.copy(all_data)
        self.stabil_data = np.copy(stabil_data)

        self.all_patches = None
        self.stabil_patches = None
        self.ranges = None
        self.visible = True
        self.select_ranges = select_ranges
        self.select_callback = select_callback
        self.selector_lines = []
        # print('here in hist')
        self.update_range()

    def _should_setup_selector_lines(self):
        """Return True when draggable selector lines should be created."""
        has_callback = self.select_callback[0] is not None
        has_range = (self.select_ranges[0] is not None
                     or self.select_ranges[1] is not None)
        return has_callback and has_range and not self.selector_lines

    def _setup_selector_lines(self):
        """Create draggable vertical lines and wire mouse-event handlers."""
        for val in self.select_ranges:
            if val is None:
                self.selector_lines.append(None)
            else:
                line = self.axes.axvline(val, picker=5, color='red')
                self.selector_lines.append(line)
        self.axes.figure.canvas.mpl_connect('pick_event', self.on_pick_event)
        self.axes.figure.canvas.mpl_connect(
            "button_release_event", self.on_release_event)
        self.axes.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_move_event)
        self.dragged = None

    def update_range(self, *args):
        self.ranges = (self.lrange_box.value(), self.urange_box.value())
        if self.ranges[0] >= self.ranges[1]:
            return
        if self.all_patches:
            for patch in self.all_patches:
                patch.remove()
        n, self.bins, self.all_patches = self.axes.hist(
            self.all_data, bins=50, color='blue', range=self.ranges)
        self.axes.set_xlim(self.ranges)
        self.axes.set_ylim((0, max(n) * 1.1))
        self.axes.set_yticks([])
        if self._should_setup_selector_lines():
            self._setup_selector_lines()
        self.update_histo(self.stabil_data)

    def update_histo(self, stabil_data, select_ranges=None):
        self.stabil_data = np.copy(stabil_data)

        if self.stabil_patches:
            for patch in self.stabil_patches:
                patch.remove()

        _, _, self.stabil_patches = self.axes.hist(
            stabil_data, bins=self.bins, color='orange')
        if self.selector_lines and select_ranges is not None:

            # self.axes.figure.canvas.mpl_disconnect(self.connect_cid)
            for val, line in zip(select_ranges, self.selector_lines):
                if line is None:
                    continue
                line.set_xdata(val)
        self.axes.figure.canvas.draw_idle()

    def closeEvent(self, e):
        self.visible = False
        e.ignore()
        self.hide()

    def on_pick_event(self, event):

        self.dragged = event.artist
        self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

        return True

    def on_release_event(self, event):
        " Update text position and redraw"

        if self.dragged is not None:
            xdata = event.xdata
            if not xdata:
                return False
            # old_pos = self.dragged.get_xdata()
            # new_pos = old_pos[0] + event.xdata - self.pick_pos[0]
            self.dragged.set_xdata(xdata)
            # print(self.dragged.get_xdata(), event.xdata)
            ind = self.selector_lines.index(self.dragged)
            self.select_callback[ind](xdata)

            if len(self.selector_lines) == 1:
                self.urange_box.setValue(xdata * 2)
                self.urange_box.delayed_emit()
            elif len(self.selector_lines) == 2:
                delta_x = (self.ranges[0 if ind == 1 else 1] - xdata) / 2
                [self.lrange_box, self.urange_box][ind].setValue(xdata - delta_x)
                [self.lrange_box, self.urange_box][ind].delayed_emit()

            self.dragged = None
            self.axes.figure.canvas.draw_idle()
        return True

    def on_move_event(self, event):
        " Update text position and redraw"

        if self.dragged is not None:
            # old_pos = self.dragged.get_xdata()
            # new_pos = old_pos[0] + event.xdata - self.pick_pos[0]
            self.dragged.set_xdata(event.xdata)
            # print(self.dragged.get_xdata(), event.xdata)
            # self.dragged = None
            self.axes.figure.canvas.draw_idle()
        return True


def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10 ** sig_fig) == int(b * 10 ** sig_fig)
            )


def start_stabil_gui(
        stabil_plot,
        modal_data,
        geometry_data=None,
        prep_signals=None,
        select_modes=None,
        mode_shape_plot_cls=None,
        **kwargs):
    '''Open the stabilisation-diagram GUI.

    Parameters
    ----------
    stabil_plot : StabilPlot
        The stabilisation-diagram plot to interact with.
    modal_data : ModalBase
        The identified modal data behind the diagram.
    geometry_data : PreProcessingTools.GeometryProcessor, optional
        When given, an embedded mode-shape viewer is created so a selected
        pole's shape can be inspected from within the diagram.
    prep_signals : PreProcessingTools.PreProcessSignals, optional
        Signals backing the mode-shape viewer.
    select_modes : list, optional
        Pre-selected mode indices.
    mode_shape_plot_cls : type, optional
        Backend class for the *embedded* mode-shape viewer.  Defaults to the
        globally resolved backend (:func:`pyOMA.core.resolve_mode_shape_backend`,
        i.e. pyvista when installed, else matplotlib).  Pass an explicit class
        to override just this viewer.
    **kwargs
        Forwarded to the mode-shape backend constructor.
    '''
    # print(kwargs)
    if select_modes is None:
        select_modes = []
    if mode_shape_plot_cls is None:
        mode_shape_plot_cls = resolve_mode_shape_backend()

    def _handler(msg_type, msg_string):
        pass

    global app
    app = QApplication.instance() or QApplication(sys.argv)

    if not isinstance(stabil_plot, StabilPlot):
        raise TypeError(
            f"stabil_plot must be StabilPlot, got {type(stabil_plot).__name__!r}")
    cmpl_plot = ComplexPlot()
    if geometry_data is not None:  # and prep_signals is not None:

        mode_shape_plot = mode_shape_plot_cls(stabil_calc=stabil_plot.stabil_calc,
                                              modal_data=modal_data,
                                              geometry_data=geometry_data,
                                              prep_signals=prep_signals,
                                              **kwargs)

        msh_plot = ModeShapeGUI(mode_shape_plot, reduced_gui=True)
        msh_plot.setGeometry(1000, 0, 800, 600)
        msh_plot.reset_view()
        msh_plot.hide()

    else:
        msh_plot = None

    # qInstallMessageHandler(handler) #suppress unimportant error msg

    stabil_gui = StabilGUI(stabil_plot, cmpl_plot, msh_plot)
    # stabil_gui.cursor.add_datapoints(select_modes)
    loop = QEventLoop()
    stabil_gui.destroyed.connect(loop.quit)
    loop.exec()
    FigureCanvasBase(stabil_plot.fig)
    return


if __name__ == '__main__':
    pass
