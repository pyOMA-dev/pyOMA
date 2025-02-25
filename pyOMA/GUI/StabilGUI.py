# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

from .HelpersGUI import DelayedDoubleSpinBox, MyMplCanvas, my_excepthook
from .PlotMSHGUI import ModeShapeGUI
from pyOMA.core.StabilDiagram import StabilPlot, StabilCluster
from pyOMA.core.PlotMSH import ModeShapePlot
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot, QObject, qInstallMessageHandler, QTimer, QEventLoop
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QPushButton, \
    QCheckBox, QButtonGroup, QLabel, QComboBox, \
    QTextEdit, QGridLayout, QFrame, QVBoxLayout, QAction, \
    QFileDialog, QMessageBox, QApplication, QRadioButton, \
    QLineEdit, QSizePolicy, QDoubleSpinBox
import numpy as np
import sys
import os

# check if python is running in headless mode i.e. as a server script
# if 'DISPLAY' in os.environ:
    # matplotlib.use("Qt5Agg", force=True)
from matplotlib import rcParams
from matplotlib import ticker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
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

NoneType = type(None)

sys.excepthook = my_excepthook


def resizeEvent_(self, event):
    w = event.size().width()
    h = event.size().height()
    dpival = self.figure.dpi
    winch, hinch = self.figure.get_size_inches()
    aspect = winch / hinch
    if w / h <= aspect:
        h = w / aspect
    else:
        w = h * aspect
    winch = w / dpival
    hinch = h / dpival
    self.figure.set_size_inches(winch, hinch)
    FigureCanvasBase.resize_event(self)
    self.draw()
    self.update()
    QWidget.resizeEvent(self, event)

'''
..TODO::
 * scale markers right on every platform
 * frequency range as argument or from ssi params, sampling freq
 * add switch to choose between "unstable only in ..." or "stable in ..."
 * (select and merge several poles with a rectangular mouse selection)
 * distinguish beetween stabilization criteria and filtering criteria
 * add zoom and sliders (horizontal/vertical) for the main figure
 * distinguish between  "export results" and "save state"

'''


class StabilGUI(QMainWindow):

    def __init__(self, stabil_plot, cmpl_plot, msh_plot=None):

        QMainWindow.__init__(self)
        self.setWindowTitle(
            'Stabilization Diagram: {} - {}'.format(
                stabil_plot.stabil_calc.setup_name,
                stabil_plot.stabil_calc.start_time))

        self.stabil_plot = stabil_plot
        self.stabil_calc = stabil_plot.stabil_calc
        if self.stabil_calc.state < 2:
            self.stabil_calc.calculate_stabilization_masks()

        self.create_menu()
        self.create_main_frame(cmpl_plot, msh_plot)

        self.histo_plot_f = None
        self.histo_plot_sf = None
        self.histo_plot_d = None
        self.histo_plot_sd = None
        self.histo_plot_dr = None
        self.histo_plot_mac = None
        self.histo_plot_mpc = None
        self.histo_plot_mpd = None

        for index, mode in enumerate(self.stabil_calc.select_modes):
            self.mode_selector_add(mode, index)

        # self.setGeometry(0, 0, 1800, 1000)

        self.show()

        for widg in [self.mode_val_view, self.current_value_view]:
            widg.setText('\n \n \n \n \n \n \n')
            height = widg.document().size().toSize().height() + 3
            widg.setFixedHeight(height)
        # self.plot_selector_msh.setChecked(True)
        self.update_stabil_view()

    def create_main_frame(self, cmpl_plot, msh_plot):
        '''
        set up all the widgets and other elements to draw the GUI
        '''
        main_frame = QWidget()

        df_max = self.stabil_calc.df_max * 100
        dd_max = self.stabil_calc.dd_max * 100
        dmac_max = self.stabil_calc.dmac_max * 100
        d_range = self.stabil_calc.d_range
        mpc_min = self.stabil_calc.mpc_min
        mpd_max = self.stabil_calc.mpd_max

        self.fig = self.stabil_plot.fig

        self.canvas = FigureCanvasQTAgg(self.fig)

        self.canvas.setParent(main_frame)

        self.init_cursor()
        main_layout = QHBoxLayout()

        left_pane_layout = QVBoxLayout()
        left_pane_layout.addStretch(1)
        palette = QPalette()
        palette.setColor(QPalette.Base, Qt.transparent)

        self.current_value_view = QTextEdit()
        self.current_value_view.setFrameShape(QFrame.Box)
        self.current_value_view.setPalette(palette)

        self.diag_val_widget = QWidget()

        fra_1 = QFrame()
        fra_1.setFrameShape(QFrame.Panel)
        fra_1.setLayout(
            self.create_stab_val_widget(
                df_max=df_max,
                dd_max=dd_max,
                d_mac=dmac_max,
                d_range=d_range,
                mpc_min=mpc_min,
                mpd_max=mpd_max))
        left_pane_layout.addWidget(fra_1)

        left_pane_layout.addStretch(2)

        fra_2 = QFrame()
        fra_2.setFrameShape(QFrame.Panel)
        fra_2.setLayout(self.create_diag_val_widget())
        left_pane_layout.addWidget(fra_2)

        left_pane_layout.addStretch(2)
        # left_pane_layout.addWidget(self.current_value_view)
        # left_pane_layout.addStretch(1)

        # right_pane_layout = QVBoxLayout()

        self.plot_selector_c = QCheckBox('Mode Shape in Complex Plane')
        self.plot_selector_c.toggled.connect(self.toggle_cpl_plot)
        self.plot_selector_msh = QCheckBox('Mode Shape in Spatial Model')
        self.plot_selector_msh.toggled.connect(self.toggle_msh_plot)

#         self.group = QButtonGroup()
#         self.group.addButton(self.plot_selector_c)
#         self.group.addButton(self.plot_selector_msh)

        self.mode_selector = QComboBox()
        self.mode_selector.currentIndexChanged[
            int].connect(self.update_mode_val_view)

        self.mode_plot_widget = QWidget()
        self.cmplx_plot_widget = QWidget()

        self.cmpl_plot = cmpl_plot
        # fig = self.cmpl_plot.fig
        # canvas1 = FigureCanvasQTAgg(fig)
        # canvas1.setParent(self.cmplx_plot_widget)

        self.msh_plot = msh_plot

        lay = QHBoxLayout()
        # lay.addWidget(canvas1)
        self.cmplx_plot_widget.setLayout(lay)
        self.cmpl_plot.plot_diagram()

        lay = QHBoxLayout()
        # lay.addWidget(canvas2)
        self.mode_plot_widget.setLayout(lay)

        self.mode_val_view = QTextEdit()
        self.mode_val_view.setFrameShape(QFrame.Box)

        self.mode_val_view.setPalette(palette)
        left_pane_layout.addStretch(1)
        left_pane_layout.addWidget(self.mode_selector)

        left_pane_layout.addWidget(self.plot_selector_c)
        left_pane_layout.addWidget(self.plot_selector_msh)
        left_pane_layout.addStretch(2)
        self.mode_plot_layout = QVBoxLayout()
        self.mode_plot_layout.addWidget(self.cmplx_plot_widget)
        left_pane_layout.addLayout(self.mode_plot_layout)
        left_pane_layout.addStretch(2)
        left_pane_layout.addWidget(self.mode_val_view)
        left_pane_layout.addStretch(1)

        main_layout.addLayout(left_pane_layout)
        main_layout.addWidget(self.canvas)
        main_layout.setStretchFactor(self.canvas, 1)
        # main_layout.addLayout(right_pane_layout)
        vbox = QVBoxLayout()
        vbox.addLayout(main_layout)
        vbox.addLayout(self.create_buttons())
        main_frame.setLayout(vbox)
        self.stabil_plot.fig.set_facecolor('none')
        self.setCentralWidget(main_frame)
        self.current_mode = (0, 0)

        self.stabil_calc.add_callback('add_mode', self.mode_selector_add)
        self.stabil_calc.add_callback('remove_mode', self.mode_selector_take)
        return

    def create_buttons(self):
        b0 = QPushButton('Apply')
        b0.released.connect(self.update_stabil_view)
        b1 = QPushButton('Save Figure')
        b1.released.connect(self.save_figure)

        b2 = QPushButton('Export Results')
        b2.released.connect(self.save_results)
        b3 = QPushButton('Save State')
        b3.released.connect(self.save_state)
        b4 = QPushButton('OK and Close')
        b4.released.connect(self.close)

        lay = QHBoxLayout()

        lay.addWidget(b0)
        lay.addWidget(b1)
        lay.addWidget(b2)
        lay.addWidget(b3)
        lay.addWidget(b4)
        lay.addStretch()

        return lay

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
            select_ranges=[None],
            select_callback=[None]):
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

    def update_mode_val_view(self, index):
        # display information about currently selected mode
        i = self.stabil_calc.select_modes[index]

        n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
            i)
        if self.stabil_calc.capabilities['std']:
            import scipy.stats
            num_blocks = self.stabil_calc.modal_data.num_blocks
            stdf = scipy.stats.t.ppf(
                0.975, num_blocks) * stdf / np.sqrt(num_blocks)
            stdd = scipy.stats.t.ppf(
                0.975, num_blocks) * stdd / np.sqrt(num_blocks)

        self.current_mode = i
        s = ''
        for text, val in [('Frequency=%1.3fHz, \n' % (f), f),
                          ('CI Frequency ± %1.3e, \n' % (stdf), stdf),
                          ('Order=%1.0f, \n' % (n), n),
                          ('Damping=%1.3f%%,  \n' % (d), d),
                          ('CI Damping ± %1.3e,  \n' % (stdd), stdd),
                          ('MPC=%1.5f, \n' % (mpc), mpc),
                          ('MP=%1.3f\u00b0, \n' % (mp), mp),
                          ('MPD=%1.5f\u00b0, \n' % (mpd), mpd),
                          ('dMP=%1.3f\u00b0, \n' % (dmp), dmp),
                          # ('dMPD=%1.5f\u00b0, \n' % (dmpd),     dmpd),
                          ('MTN=%1.5f, \n' % (mtn), mtn),
                          ('MC=%1.5f, \n' % (MC), MC),
                          ('Ext=%1.5f\u00b0, \n' % (ex_1), ex_1),
                          ('Ext=%1.3f\u00b0, \n' % (ex_2), ex_2)
                          ]:
            if val is not np.nan:
                s += text
        self.mode_val_view.setText(s)
        height = self.mode_val_view.document().size().toSize().height() + 3
        self.mode_val_view.setFixedHeight(height)
        self.update_mode_plot(i, mp)

    @pyqtSlot(tuple)
    def mode_selector_add(self, i, index):
        # add mode tomode_selector and select it
        n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
            i)
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
        if self.current_mode == i_:
            if self.stabil_calc.select_modes:
                self.current_mode = self.stabil_calc.select_modes[0]
            else:
                self.current_mode = (0, 0)
        self.mode_selector.currentIndexChanged[
            int].disconnect(self.update_mode_val_view)
        self.mode_selector.clear()

        for index, i in enumerate(self.stabil_calc.select_modes):
            n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
                i)
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

    def toggle_cpl_plot(self, b):
        # change the type of mode plot

        # print('cpl',b)
        if b:
            self.cmpl_plot.show()
        else:
            self.cmpl_plot.hide()

    def init_cursor(self):
        self.cursor = self.stabil_plot.init_cursor()
        # print(self.stabil_calc.select_modes, type(self.stabil_calc.select_modes))
        # self.cursor = DataCursor(
        #     ax=stabil_plot.ax,
        #     order_data=stabil_plot.stabil_calc.order_dummy,
        #     f_data=stabil_plot.stabil_calc.masked_frequencies,
        #     datalist=stabil_plot.stabil_calc.select_modes,
        #     color='black', useblit=False)
        #
        #
        # stabil_plot.fig.canvas.mpl_connect(
        #     'button_press_event', self.cursor.onmove)
        # stabil_plot.fig.canvas.mpl_connect(
        #     'resize_event', self.cursor.fig_resized)
        # self.cursor.add_datapoints(self.stabil_calc.select_modes)
        # self.stabil_calc.select_callback = self.cursor.add_datapoint

        self.cursor.add_callback('show_current_info', self.update_value_view)

        # self.cursor.show_current_info.connect(
        #     self.update_value_view)
        # self.cursor.add_callback('mode_selected', self.mode_selector_add)
        # self.cursor.mode_selected.connect(self.mode_selector_add)
        # self.cursor.add_callback('mode_deselected', self.mode_selector_take)
        # self.cursor.mode_deselected.connect(self.mode_selector_take)
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

        n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
            i)

        if self.stabil_calc.capabilities['std']:
            import scipy.stats
            num_blocks = self.stabil_calc.modal_data.num_blocks
            stdf = scipy.stats.t.ppf(
                0.975, num_blocks) * stdf / np.sqrt(num_blocks)
            stdd = scipy.stats.t.ppf(
                0.975, num_blocks) * stdd / np.sqrt(num_blocks)

        self.current_mode = i
        s = ''
        for text, val in [('Frequency=%1.3fHz, \n' % (f), f),
                          ('CI Frequency ± %1.3e, \n' % (stdf), stdf),
                          ('Order=%1.0f, \n' % (n), n),
                          ('Damping=%1.3f%%,  \n' % (d), d),
                          ('CI Damping ± %1.3e,  \n' % (stdd), stdd),
                          ('MPC=%1.5f, \n' % (mpc), mpc),
                          ('MP=%1.3f\u00b0, \n' % (mp), mp),
                          ('MPD=%1.5f\u00b0, \n' % (mpd), mpd),
                          ('dMP=%1.3f\u00b0, \n' % (dmp), dmp),
                          # ('dMPD=%1.5f\u00b0, \n' % (dmpd),     dmpd),
                          ('MTN=%1.5f, \n' % (mtn), mtn),
                          ('MC=%1.5f, \n' % (MC), MC),
                          ('Ext=%1.5f\u00b0, \n' % (ex_1), ex_1),
                          ('Ext=%1.3f\u00b0, \n' % (ex_2), ex_2)
                          ]:
            if val is not np.nan:
                s += text

        self.current_value_view.setText(s)
        height = self.current_value_view.document(
        ).size().toSize().height() + 3
        self.current_value_view.setFixedHeight(height)

    def update_stabil_view(self):
        df_max = float(self.df_edit.text()) / 100
        dd_max = float(self.dd_edit.text()) / 100

        if self.stabil_calc.capabilities['std']:
            stdf_max = float(self.stdf_edit.text()) / 100
        else:
            stdf_max = None

        if self.stabil_calc.capabilities['std']:
            stdd_max = float(self.stdd_edit.text()) / 100
        else:
            stdd_max = None

        if self.stabil_calc.capabilities['msh']:
            dmac_max = float(self.mac_edit.text()) / 100
        else:
            dmac_max = None

        d_range = (
            float(self.d_min_edit.text()), float(self.d_max_edit.text()))

        if self.stabil_calc.capabilities['msh']:
            mpc_min = float(self.mpc_edit.text())
            mpd_max = float(self.mpd_edit.text())
        else:
            mpc_min = None
            mpd_max = None

        if self.stabil_calc.capabilities['MC']:
            MC_min = float(self.MC_edit.text())
        else:
            MC_min = None

        f_range = (float(self.freq_low.text()), float(self.freq_high.text()))
        order_range = (int(self.n_low.text()), int(
            self.n_step.text()), int(self.n_high.text()))

        self.stabil_plot.update_stabilization(
            df_max=df_max,
            stdf_max=stdf_max,
            dd_max=dd_max,
            stdd_max=stdd_max,
            dmac_max=dmac_max,
            d_range=d_range,
            mpc_min=mpc_min,
            mpd_max=mpd_max,
            MC_min=MC_min,
            order_range=order_range
            )
        self.stabil_plot.update_xlim(f_range)
        self.stabil_plot.update_ylim((order_range[0], order_range[2]))

        if self.histo_plot_f is not None:
            self.create_histo_plot_f()
        if self.histo_plot_sf is not None:
            self.create_histo_plot_sf()
        if self.histo_plot_d is not None:
            self.create_histo_plot_d()
        if self.histo_plot_sd is not None:
            self.create_histo_plot_sd()
        if self.histo_plot_dr is not None:
            self.create_histo_plot_dr()
        if self.histo_plot_mac is not None:
            self.create_histo_plot_mac()
        if self.histo_plot_mpc is not None:
            self.create_histo_plot_mpc()
        if self.histo_plot_mpd is not None:
            self.create_histo_plot_mpd()

    def create_menu(self):
        '''
        create the menubar and add actions to it
        '''

        def add_actions(target, actions):
            for action in actions:
                if action is None:
                    target.addSeparator()
                else:
                    target.addAction(action)

        def create_action(text, slot=None, shortcut=None,
                          icon=None, tip=None, checkable=False,
                          signal="triggered()"):
            action = QAction(text, self)
            if icon is not None:
                action.setIcon(QIcon(":/%s.png" % icon))
            if shortcut is not None:
                action.setShortcut(shortcut)
            if tip is not None:
                action.setToolTip(tip)
                action.setStatusTip(tip)
            if slot is not None:
                getattr(action, signal.strip('()')).connect(slot)
            if checkable:
                action.setCheckable(True)
            return action

        file_menu = self.menuBar().addMenu("&File")

        load_file_action = create_action("&Save plot",
                                         shortcut="Ctrl+S",
                                         slot=None,
                                         tip="Save the plot")
        quit_action = create_action("&Quit",
                                    slot=self.close,
                                    shortcut="Ctrl+Q",
                                    tip="Close the application")

        add_actions(file_menu,
                    (load_file_action, None, quit_action))

        self.menuBar().addMenu("&Help")

    def create_stab_val_widget(self, df_max=1, dd_max=5, d_mac=1,
                               d_range=(0, 5), mpc_min=0.9, mpd_max=15):
        layout = QGridLayout()
        layout.addWidget(QLabel('Stabilization Criteria'), 1, 1, 1, 3)

        layout.setColumnStretch(2, 1)
        i = 2

        if self.stabil_calc.capabilities['f']:
            layout.addWidget(QLabel('Frequency [%]'), i, 1)
            self.df_edit = QLineEdit(str(df_max))
            self.df_edit.setMaxLength(8)
            self.df_edit.setFixedWidth(60)
            layout.addWidget(self.df_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_f)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['std']:
            layout.addWidget(QLabel('CoV F. [% of F]'), i, 1)
            self.stdf_edit = QLineEdit('100')
            self.stdf_edit.setMaxLength(8)
            self.stdf_edit.setFixedWidth(60)
            layout.addWidget(self.stdf_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_sf)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['d']:
            layout.addWidget(QLabel('Damping[%]'), i, 1)
            self.dd_edit = QLineEdit(str(dd_max))
            self.dd_edit.setMaxLength(8)
            self.dd_edit.setFixedWidth(60)
            layout.addWidget(self.dd_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_d)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['std']:
            layout.addWidget(QLabel('CoV D. [% of D]'), i, 1)
            self.stdd_edit = QLineEdit('100')
            self.stdd_edit.setMaxLength(8)
            self.stdd_edit.setFixedWidth(60)
            layout.addWidget(self.stdd_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_sd)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['msh']:
            layout.addWidget(QLabel('MAC [%]'), i, 1)
            self.mac_edit = QLineEdit(str(d_mac))
            self.mac_edit.setMaxLength(8)
            self.mac_edit.setFixedWidth(60)
            layout.addWidget(self.mac_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mac)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['d']:
            layout.addWidget(QLabel('Damping range [%]'), i, 1)
            self.d_min_edit = QLineEdit(str(d_range[0]))
            self.d_min_edit.setMaxLength(8)
            self.d_min_edit.setFixedWidth(60)
            self.d_max_edit = QLineEdit(str(d_range[1]))
            self.d_max_edit.setMaxLength(8)
            self.d_max_edit.setFixedWidth(60)
            lay = QHBoxLayout()
            lay.addStretch()
            lay.addWidget(self.d_min_edit)
            lay.addWidget(QLabel('to'))
            lay.addWidget(self.d_max_edit)
            layout.addLayout(lay, i, 2, 1, 2)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_dr)
            layout.addWidget(button, i, 4)
            i += 1

        layout.setRowStretch(i, 2)
        i += 1

        if self.stabil_calc.capabilities['msh']:
            layout.addWidget(QLabel('MPC_min '), i, 1)
            self.mpc_edit = QLineEdit(str(mpc_min))
            self.mpc_edit.setMaxLength(8)
            self.mpc_edit.setFixedWidth(60)
            layout.addWidget(self.mpc_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mpc)
            layout.addWidget(button, i, 4)
            i += 1

            layout.addWidget(QLabel('MPD_max [°]'), i, 1)
            self.mpd_edit = QLineEdit(str(mpd_max))
            self.mpd_edit.setMaxLength(8)
            self.mpd_edit.setFixedWidth(60)
            layout.addWidget(self.mpd_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mpd)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['mtn']:
            layout.addWidget(QLabel('MTN_max []'), i, 1)
            self.mtn_edit = QLineEdit('0')
            self.mtn_edit.setMaxLength(8)
            self.mtn_edit.setFixedWidth(60)
            layout.addWidget(self.mtn_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mtn)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['MC']:
            layout.addWidget(QLabel('MC_min []'), i, 1)
            self.MC_edit = QLineEdit('0')
            self.MC_edit.setMaxLength(8)
            self.MC_edit.setFixedWidth(60)
            layout.addWidget(self.MC_edit, i, 3)
            button = QPushButton('Show MC')
            button.setCheckable(True)
            button.released.connect(self.show_MC_plot)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['auto']:
            b0 = QPushButton('Clear automatically')
            b0.released.connect(self.prepare_auto_clearing)

            self.num_iter_edit = QLineEdit(str(self.stabil_calc.num_iter))

            layout.addWidget(b0, i, 1)
            layout.addWidget(self.num_iter_edit, i, 2)
            i += 1

            b1 = QPushButton('Classify automatically')
            b1.released.connect(self.prepare_auto_classification)

            self.use_stabil_box = QCheckBox('Use Stabilization')
            self.use_stabil_box.setTristate(False)
            self.use_stabil_box.setChecked(False)
            self.threshold_box = QLineEdit()
            self.threshold_box.setPlaceholderText(
                str(self.stabil_calc.threshold))

            layout.addWidget(b1, i, 1)
            layout.addWidget(self.use_stabil_box, i, 2)
            layout.addWidget(self.threshold_box, i, 3)
            i += 1

            b2 = QPushButton('Select automatically')
            b2.released.connect(self.prepare_auto_selection)

            self.num_modes_box = QLineEdit(str(0))

            layout.addWidget(b2, i, 1)
            layout.addWidget(self.num_modes_box, i, 2)

        return layout

    def prepare_auto_clearing(self):
        assert self.stabil_calc.capabilities['auto']
        num_iter = int(self.num_iter_edit.text())

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_clearing(num_iter)
            self.threshold_box.setPlaceholderText(
                str(self.stabil_calc.threshold))
            self.stabil_plot.update_stabilization()
            self.check_clear.setChecked(True)

    def prepare_auto_classification(self):
        assert self.stabil_calc.capabilities['auto']
        use_stabil = self.use_stabil_box.isChecked()

        threshold = self.threshold_box.text()
        # print(threshold, type(threshold), threshold.isnumeric())
        if threshold.isnumeric():
            threshold = float(threshold)
        else:
            threshold = None

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_classification(threshold, use_stabil)
            self.check_select.setChecked(True)

    def prepare_auto_selection(self):
        assert self.stabil_calc.capabilities['auto']
        num_modes = self.num_modes_box.text()
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

    def create_diag_val_widget(
        self,
        show_sf=True,
        show_sd=True,
        show_sv=True,
        show_sa=True,
        show_all=True,
        show_psd=False,
        snap_to='sa',
        f_range=(
            0,
            0),
        n_range=(
            0,
            1,
            0)):
        layout = QGridLayout()

        layout.addWidget(QLabel('View Settings'), 1, 1, 1, 2)

        i = 2

        check_sa = QCheckBox('Stable Pole')
        check_sa.setChecked(show_sa)
        self.stabil_plot.toggle_stable(show_sa)
        check_sa.stateChanged.connect(self.stabil_plot.toggle_stable)
        snap_sa = QRadioButton()
        snap_sa.toggled.connect(self.snap_stable)
        snap_sa.setChecked(snap_to == 'sa')
        layout.addWidget(check_sa, i, 1)
        layout.addWidget(snap_sa, i, 2)
        i += 1

        check_all = QCheckBox('All Poles')
        check_all.setChecked(show_all)
        self.stabil_plot.toggle_all(show_all)
        check_all.stateChanged.connect(self.stabil_plot.toggle_all)
        snap_all = QRadioButton()
        snap_all.toggled.connect(self.snap_all)
        snap_all.setChecked(snap_to == 'all')
        layout.addWidget(check_all, i, 1)
        layout.addWidget(snap_all, i, 2)
        i += 1

        if self.stabil_calc.capabilities['auto']:
            show_clear = self.stabil_calc.state >= 3
            check_clear = QCheckBox('AutoClear')
            check_clear.setChecked(show_clear)
            self.check_clear = check_clear
            self.stabil_plot.toggle_clear(show_clear)
            check_clear.stateChanged.connect(self.stabil_plot.toggle_clear)
            snap_clear = QRadioButton()
            snap_clear.toggled.connect(self.snap_clear)
            snap_clear.setChecked(snap_to == 'clear')
            layout.addWidget(check_clear, i, 1)
            layout.addWidget(snap_clear, i, 2)
            i += 1

            show_select = self.stabil_calc.state >= 4
            check_select = QCheckBox('AutoSelect')
            check_select.setChecked(show_select)
            self.check_select = check_select
            self.stabil_plot.toggle_select(show_select)
            check_select.stateChanged.connect(self.stabil_plot.toggle_select)
            snap_select = QRadioButton()
            snap_select.toggled.connect(self.snap_select)
            snap_select.setChecked(snap_to == 'select')
            layout.addWidget(check_select, i, 1)
            layout.addWidget(snap_select, i, 2)
            i += 1

        if self.stabil_calc.capabilities['data']:
            psd_check = QCheckBox('Show PSD')
            psd_check.setChecked(show_psd)
            self.stabil_plot.plot_sv_psd(show_psd)
            psd_check.stateChanged.connect(self.stabil_plot.plot_sv_psd)
            layout.addWidget(psd_check, i, 1, 1, 2)
            i += 1

        lay = QHBoxLayout()
        lay.addWidget(QLabel('Freq. range:'))
        if f_range[1] == 0:
            f_range = (f_range[0], self.stabil_calc.get_max_f())
        self.freq_low = QLineEdit('{:2.3f}'.format(f_range[0]))
        self.freq_low.setFixedWidth(60)
        self.freq_high = QLineEdit('{:2.3f}'.format(f_range[1] * 1.05))
        self.freq_high.setFixedWidth(60)
        lay.addWidget(self.freq_low)
        lay.addWidget(QLabel('to'))
        lay.addWidget(self.freq_high)
        lay.addWidget(QLabel('[Hz]'))
        layout.addLayout(lay, i, 1, 1, 2)
        i += 1

        lay = QHBoxLayout()
        lay.addWidget(QLabel('Order. range (low:step:high):'))
        if n_range[2] == 0:
            n_range = (
                n_range[0],
                n_range[1],
                self.stabil_calc.modal_data.max_model_order)
        self.n_low = QLineEdit('{:2d}'.format(n_range[0]))
        self.n_low.setFixedWidth(60)
        self.n_step = QLineEdit('{:2d}'.format(n_range[1]))
        self.n_step.setFixedWidth(60)
        self.n_high = QLineEdit('{:2d}'.format(n_range[2]))
        self.n_high.setFixedWidth(60)
        lay.addWidget(self.n_low)
        lay.addWidget(self.n_step)
        lay.addWidget(self.n_high)
        layout.addLayout(lay, i, 1, 1, 2)
        i += 1

        return layout

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

    def save_state(self):

        fname, _ = QFileDialog.getSaveFileName(self, caption="Choose a directory to save to",
                                                  directory=os.getcwd(), filter='Numpy Archive File (*.npz)')

        if fname == '':
            return
        fname, fext = os.path.splitext(fname)

        if fext != 'npz':
            fname += '.npz'

        self.stabil_calc.save_state(fname)
        self.close()

    def closeEvent(self, *args, **kwargs):
        if self.msh_plot is not None:
            self.msh_plot.mode_shape_plot.stop_ani()

        # self.stabil_calc.select_modes = self.stabil_calc.select_modes
        self.deleteLater()

        return QMainWindow.closeEvent(self, *args, **kwargs)

    def keyPressEvent(self, e):
        # print(e.key())
        if e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter:
            self.update_stabil_view()
            # print('2')
        super().keyPressEvent(e)


class ComplexPlot(QMainWindow):

    def __init__(self):

        QMainWindow.__init__(self)
        self.setWindowTitle('Modeshapeplot in complex plane')
        self.setGeometry(300, 300, 1000, 600)
        main_frame = QWidget()
        vbox = QVBoxLayout()

        self.fig = Figure(facecolor='white', dpi=100, figsize=(4, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        # self.canvas.setParent(main_frame)

        vbox.addWidget(self.canvas, 10, Qt.AlignCenter)
        main_frame.setLayout(vbox)

        self.setCentralWidget(main_frame)
        # self.show()

    def scatter_this(self, msh, mp=None):

        self.ax.cla()
        self.ax.scatter(msh.real, msh.imag)

        if mp is not None:
            while mp < 0:
                mp += 180
            while mp > 360:
                mp -= 360
            mp = mp * np.pi / 180
            xmin, xmax = -1, 1
            ymin, ymax = -1, 1
            if mp <= np.pi / 2:
                x1 = max(xmin, ymin / np.tan(mp))
                x2 = min(xmax, ymax / np.tan(mp))
                y1 = max(ymin, xmin * np.tan(mp))
                y2 = min(ymax, xmax * np.tan(mp))
            elif mp <= np.pi:
                x1 = max(xmin, ymax / np.tan(mp))
                x2 = min(xmax, ymin / np.tan(mp))
                y2 = max(ymin, xmax * np.tan(mp))
                y1 = min(ymax, xmin * np.tan(mp))
            elif mp <= 3 * np.pi / 2:
                x1 = max(xmin, ymin / np.tan(mp))
                x2 = min(xmax, ymax / np.tan(mp))
                y1 = max(ymin, xmin * np.tan(mp))
                y2 = min(ymax, xmax * np.tan(mp))
            else:
                x1 = max(xmin, ymax / np.tan(mp))
                x2 = min(xmax, ymin / np.tan(mp))
                y2 = max(ymin, xmax * np.tan(mp))
                y1 = min(ymax, xmin * np.tan(mp))
            self.ax.plot([x1, x2], [y1, y2])
        lim = max(max(abs(msh.real)) * 1.1, max(abs(msh.imag)) * 1.1)
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_position(('data', 0 - 1))
        self.ax.spines['top'].set_position(('data', 0 - 1))

        # Hide the line (but not ticks) for "extra" spines
        for side in ['right', 'top']:
            self.ax.spines[side].set_color('none')

        # On both the x and y axes...
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

        self.ax.xaxis.set_label('Re')
        self.ax.yaxis.set_label('Im')

        # Hide the line (but not ticks) for "extra" spines
        for side in ['right', 'top']:
            self.ax.spines[side].set_color('none')

        # On both the x and y axes...
        for axis, _ in zip([self.ax.xaxis, self.ax.yaxis], [0, 0]):
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_formatter(ticker.NullFormatter())
        self.fig.canvas.draw_idle()

# class ModeShapeWidget(object):
#
#     def __init__(self, stabil_calc, modal_data, geometry_data, prep_data,**kwargs):
#
#         #print(kwargs)
#         super().__init__()
#         #sys.path.append("/vegas/users/staff/womo1998/Projects/2016_Burscheid")
#         #from main_Schwabach_2019 import print_mode_info
#
#         self.mode_shape_plot = ModeShapePlot(
#             stabil_calc=stabil_calc,
#             modal_data=modal_data,
#             geometry_data=geometry_data,
#             prep_data=prep_data,
#             amplitude=20,
#             linewidth=0.5,
#             #callback_fun=print_mode_info
#             **kwargs)
#         self.mode_shape_plot.show_axis = False
#         # self.mode_shape_plot.draw_nodes()
#         self.mode_shape_plot.draw_lines()
#         # self.mode_shape_plot.draw_parent_childs()
#         # self.mode_shape_plot.draw_chan_dofs()
#
#         self.fig = self.mode_shape_plot.fig
#         self.fig.set_size_inches((2, 2))
#         self.canvas = self.fig.canvas.switch_backends(FigureCanvasQTAgg)
#         self.mode_shape_plot.canvas = self.canvas
#         self.fig.get_axes()[0].mouse_init()
#         #self.canvas = self.mode_shape_plot.canvas
#
#         for axis in [self.mode_shape_plot.subplot.xaxis, self.mode_shape_plot.subplot.yaxis, self.mode_shape_plot.subplot.zaxis]:
#             axis.set_minor_locator(ticker.NullLocator())
#             axis.set_major_formatter(ticker.NullFormatter())
#         self.mode_shape_plot.animate()
#
#
#
#     def plot_this(self, index):
#         #self.mode_shape_plot.stop_ani()
#         self.mode_shape_plot.change_mode(mode_index=index)
#         #self.mode_shape_plot.animate()


class HistoPlot(QMainWindow):

    def __init__(
            self,
            all_data,
            stabil_data,
            title='',
            ranges=None,
            select_ranges=[None],
            select_callback=[None]):
        QMainWindow.__init__(self)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)

        self.main_widget = QWidget(self)

        l = QVBoxLayout(self.main_widget)
        sc = MyMplCanvas(self.main_widget, width=5, height=2.5, dpi=100)
        l.addWidget(sc)
        m = QHBoxLayout()
        m.addWidget(QLabel('min'))

        if ranges is None:
            ranges = (all_data.min(), all_data.max())

        step = (ranges[1] - ranges[0]) / 50
        self.lrange = DelayedDoubleSpinBox(decimals=8, singleStep=step)
        self.lrange.setValue(ranges[0])
        self.lrange.valueChangedDelayed.connect(self.update_range)
        m.addWidget(self.lrange)

        m.addWidget(QLabel('max'))
        self.urange = DelayedDoubleSpinBox(decimals=8, singleStep=step)
        self.urange.setValue(ranges[1])
        self.urange.valueChangedDelayed.connect(self.update_range)
        m.addWidget(self.urange)
        l.addLayout(m)

        self.axes = sc.axes
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
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

    def update_range(self, *args):
        self.ranges = (self.lrange.value(), self.urange.value())
        # print(self.ranges)
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
        if self.select_callback[0] is not None and (
                self.select_ranges[0] is not None or self.select_ranges[1] is not None) and not self.selector_lines:
            for val in self.select_ranges:
                if val is None:
                    self.selector_lines.append(None)
                else:
                    line = self.axes.axvline(val, picker=5, color='red')
                    # print(line)
                    self.selector_lines.append(line)
            self.axes.figure.canvas.mpl_connect(
                'pick_event', self.on_pick_event)
            self.axes.figure.canvas.mpl_connect(
                "button_release_event", self.on_release_event)
            self.axes.figure.canvas.mpl_connect(
                "motion_notify_event", self.on_move_event)
            self.dragged = None

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
                self.urange.setValue(xdata * 2)
                self.urange.delayed_emit()
            elif len(self.selector_lines) == 2:
                delta_x = (self.ranges[0 if ind == 1 else 1] - xdata) / 2
                [self.lrange, self.urange][ind].setValue(xdata - delta_x)
                [self.lrange, self.urange][ind].delayed_emit()

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
        select_modes=[],
        **kwargs):

    # print(kwargs)
    def handler(msg_type, msg_string):
        pass

    if 'app' not in globals().keys():
        global app
        app = QApplication(sys.argv)
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)

    assert isinstance(stabil_plot, StabilPlot)
    cmpl_plot = ComplexPlot()
    if geometry_data is not None:  # and prep_signals is not None:

        mode_shape_plot = ModeShapePlot(stabil_calc=stabil_plot.stabil_calc,
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
    loop.exec_()
    FigureCanvasBase(stabil_plot.fig)
    return


if __name__ == '__main__':
    pass
