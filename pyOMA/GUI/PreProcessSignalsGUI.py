# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive PyQt6 GUI for pyOMA.core.PreProcessingTools.

Wraps :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals` and
:class:`~pyOMA.core.PreProcessingTools.SignalPlot`: every pre-processing
action mutates the ``PreProcessSignals`` instance in place, and the plot
panel re-renders one of the ``SignalPlot`` diagrams against its current
state.

Hand-built widget code for now (matches the pre-QtDesigner-migration state
of PlotMSHGUI/StabilGUI). If/when this module gets the same QtDesigner
pipeline as PlotMSHGUI, restructure it the same way: ``ui/preprocess_signals.ui``
+ ``generated/ui_preprocess_signals.py`` + this file trimmed to a thin
``Ui_PreProcessSignals`` subclass.
"""
import sys
import logging

import numpy as np

from PyQt6.QtCore import Qt, QEventLoop
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QListWidget, QListWidgetItem, QAbstractItemView, QSplitter,
    QScrollArea, QStackedWidget, QMessageBox,
)

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from .HelpersGUI import MyMplCanvas
from ..core.PreProcessingTools import PreProcessSignals, SignalPlot, SDOF_ambient

logger = logging.getLogger(__name__)

app = None


class PreProcessSignalsGUI(QMainWindow):
    """Interactive GUI for signal pre-processing and diagnostic plotting.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        The signal object to inspect and process. Pre-processing actions
        performed through this GUI mutate it in place.
    parent : QWidget, optional
    """

    def __init__(self, prep_signals, parent=None):
        super().__init__(parent)
        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be a PreProcessSignals instance, "
                f"got {type(prep_signals).__name__}")
        self.prep_signals = prep_signals
        self.signal_plot = SignalPlot(prep_signals)

        self.setWindowTitle("pyOMA - Interactive Signal Pre-Processing")
        self._build_ui()
        self._refresh_channel_list()
        self._refresh_status()
        self._update_plot()
        self.show()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_plot_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_control_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_status_box())
        layout.addWidget(self._build_channel_box())
        layout.addWidget(self._build_preprocessing_box())
        layout.addWidget(self._build_plot_controls_box())
        layout.addStretch(1)

        scroll.setWidget(panel)
        scroll.setMinimumWidth(380)
        return scroll

    def _build_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.canvas = MyMplCanvas(panel, width=6.4, height=4.8)
        self.toolbar = NavigationToolbar2QT(self.canvas, panel)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        return panel

    # -- status ------------------------------------------------------
    def _build_status_box(self):
        box = QGroupBox("Signal status")
        form = QFormLayout(box)
        self.lbl_num_channels = QLabel()
        self.lbl_sampling_rate = QLabel()
        self.lbl_duration = QLabel()
        form.addRow("Channels:", self.lbl_num_channels)
        form.addRow("Sampling rate [Hz]:", self.lbl_sampling_rate)
        form.addRow("Duration [s]:", self.lbl_duration)
        return box

    def _refresh_status(self):
        self.lbl_num_channels.setText(str(self.prep_signals.num_analised_channels))
        self.lbl_sampling_rate.setText(f"{self.prep_signals.sampling_rate:.4g}")
        self.lbl_duration.setText(f"{self.prep_signals.duration:.4g}")

    # -- channel selection ---------------------------------------------
    def _build_channel_box(self):
        box = QGroupBox("Channels")
        layout = QVBoxLayout(box)
        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_channels.itemSelectionChanged.connect(self._update_plot)
        layout.addWidget(self.list_channels)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(self.list_channels.selectAll)
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(self.list_channels.clearSelection)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        self.chk_auto_ref = QCheckBox("Auto-reference (each channel vs. itself)")
        self.chk_auto_ref.stateChanged.connect(self._update_plot)
        layout.addWidget(self.chk_auto_ref)

        return box

    def _refresh_channel_list(self):
        self.list_channels.blockSignals(True)
        self.list_channels.clear()
        for idx, name in enumerate(self.prep_signals.channel_headers):
            self.list_channels.addItem(QListWidgetItem(f"{idx}: {name}"))
        self.list_channels.selectAll()
        self.list_channels.blockSignals(False)

    def _selected_channels(self):
        rows = [self.list_channels.row(item) for item in self.list_channels.selectedItems()]
        if not rows:
            return None  # PreProcessSignals/SignalPlot interpret None as "all channels"
        return sorted(rows)

    def _selected_refs(self):
        return 'auto' if self.chk_auto_ref.isChecked() else None

    # -- pre-processing actions -----------------------------------------
    def _build_preprocessing_box(self):
        box = QGroupBox("Pre-processing")
        layout = QVBoxLayout(box)
        layout.addWidget(self._build_offset_box())
        layout.addWidget(self._build_noise_box())
        layout.addWidget(self._build_filter_box())
        layout.addWidget(self._build_decimate_box())
        return box

    def _build_offset_box(self):
        box = QGroupBox("Offset / scaling")
        layout = QVBoxLayout(box)

        btn_offset = QPushButton("Correct offset")
        btn_offset.clicked.connect(self._on_correct_offset)
        layout.addWidget(btn_offset)

        row = QHBoxLayout()
        self.combo_precondition_method = QComboBox()
        self.combo_precondition_method.addItems(['iqr', 'range'])
        btn_precondition = QPushButton("Precondition")
        btn_precondition.clicked.connect(self._on_precondition)
        row.addWidget(self.combo_precondition_method)
        row.addWidget(btn_precondition)
        layout.addLayout(row)

        return box

    def _on_correct_offset(self):
        self.prep_signals.correct_offset()
        self._after_signal_mutation()

    def _on_precondition(self):
        self.prep_signals.precondition_signals(
            method=self.combo_precondition_method.currentText())
        self._after_signal_mutation()

    def _build_noise_box(self):
        box = QGroupBox("Add noise")
        form = QFormLayout(box)

        self.spin_noise_amplitude = QDoubleSpinBox()
        self.spin_noise_amplitude.setRange(0, 1e6)
        self.spin_noise_amplitude.setDecimals(6)

        self.spin_noise_snr = QDoubleSpinBox()
        self.spin_noise_snr.setRange(0, 1)
        self.spin_noise_snr.setDecimals(4)
        self.spin_noise_snr.setSingleStep(0.01)

        btn_noise = QPushButton("Add noise")
        btn_noise.clicked.connect(self._on_add_noise)

        form.addRow("Amplitude:", self.spin_noise_amplitude)
        form.addRow("SNR (fraction of RMS):", self.spin_noise_snr)
        form.addRow(btn_noise)
        return box

    def _on_add_noise(self):
        amplitude = self.spin_noise_amplitude.value()
        snr = self.spin_noise_snr.value()
        if amplitude == 0 and snr == 0:
            QMessageBox.warning(self, "Add noise", "Set a non-zero amplitude or SNR first.")
            return
        self.prep_signals.add_noise(amplitude=amplitude, snr=snr)
        self._after_signal_mutation()

    def _build_filter_box(self):
        box = QGroupBox("Filter")
        form = QFormLayout(box)

        self.chk_lowpass = QCheckBox("Lowpass [Hz]")
        self.spin_lowpass = QDoubleSpinBox()
        self.spin_lowpass.setRange(0.0001, 1e6)
        self.spin_lowpass.setEnabled(False)
        self.chk_lowpass.toggled.connect(self.spin_lowpass.setEnabled)
        row_lp = QHBoxLayout()
        row_lp.addWidget(self.chk_lowpass)
        row_lp.addWidget(self.spin_lowpass)
        form.addRow(row_lp)

        self.chk_highpass = QCheckBox("Highpass [Hz]")
        self.spin_highpass = QDoubleSpinBox()
        self.spin_highpass.setRange(0.0001, 1e6)
        self.spin_highpass.setEnabled(False)
        self.chk_highpass.toggled.connect(self.spin_highpass.setEnabled)
        row_hp = QHBoxLayout()
        row_hp.addWidget(self.chk_highpass)
        row_hp.addWidget(self.spin_highpass)
        form.addRow(row_hp)

        self.combo_ftype = QComboBox()
        self.combo_ftype.addItems(
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'moving_average', 'brickwall'])
        self.combo_ftype.currentTextChanged.connect(self._on_ftype_changed)
        form.addRow("Filter type:", self.combo_ftype)

        self.chk_auto_order = QCheckBox("Auto order")
        self.chk_auto_order.setChecked(True)
        self.spin_order = QSpinBox()
        self.spin_order.setRange(1, 1000)
        self.spin_order.setEnabled(False)
        self.chk_auto_order.toggled.connect(lambda checked: self.spin_order.setEnabled(not checked))
        row_order = QHBoxLayout()
        row_order.addWidget(self.chk_auto_order)
        row_order.addWidget(self.spin_order)
        form.addRow(row_order)

        self.spin_rp = QDoubleSpinBox()
        self.spin_rp.setRange(0.001, 10)
        self.spin_rp.setValue(0.05)
        self.spin_rs = QDoubleSpinBox()
        self.spin_rs.setRange(0.001, 100)
        self.spin_rs.setValue(0.05)
        self.lbl_rprs = QLabel("Rp / Rs (cheby/ellip):")
        row_rprs = QHBoxLayout()
        row_rprs.addWidget(self.spin_rp)
        row_rprs.addWidget(self.spin_rs)
        row_rprs_widget = QWidget()
        row_rprs_widget.setLayout(row_rprs)
        form.addRow(self.lbl_rprs, row_rprs_widget)

        self.chk_overwrite = QCheckBox("Overwrite signals")
        self.chk_overwrite.setChecked(True)
        form.addRow(self.chk_overwrite)

        self.chk_show_response = QCheckBox("Show filter response instead of signal plot")
        form.addRow(self.chk_show_response)

        btn_filter = QPushButton("Apply filter")
        btn_filter.clicked.connect(self._on_filter)
        form.addRow(btn_filter)

        self._on_ftype_changed(self.combo_ftype.currentText())
        return box

    def _on_ftype_changed(self, ftype):
        needs_rprs = ftype in ('cheby1', 'cheby2', 'ellip')
        self.spin_rp.setEnabled(needs_rprs)
        self.spin_rs.setEnabled(needs_rprs)
        self.lbl_rprs.setEnabled(needs_rprs)

    def _on_filter(self):
        lowpass = self.spin_lowpass.value() if self.chk_lowpass.isChecked() else None
        highpass = self.spin_highpass.value() if self.chk_highpass.isChecked() else None
        if lowpass is None and highpass is None:
            QMessageBox.warning(self, "Filter", "Enable at least one of lowpass/highpass.")
            return

        order = None if self.chk_auto_order.isChecked() else self.spin_order.value()
        ftype = self.combo_ftype.currentText()
        RpRs = [self.spin_rp.value(), self.spin_rs.value()] if ftype in ('cheby1', 'cheby2', 'ellip') else None
        overwrite = self.chk_overwrite.isChecked()

        plot_ax = None
        show_response = self.chk_show_response.isChecked()
        if show_response:
            self.canvas.figure.clear()
            plot_ax = self.canvas.figure.subplots(nrows=2, ncols=1)  # [time_ax, freq_ax]

        self.prep_signals.filter_signals(
            lowpass=lowpass, highpass=highpass, overwrite=overwrite,
            order=order, ftype=ftype, RpRs=RpRs, plot_ax=plot_ax)

        self._refresh_status()
        if show_response:
            self.canvas.draw_idle()  # filter response is shown instead of a signal diagram this time
        else:
            self._update_plot()

    def _build_decimate_box(self):
        box = QGroupBox("Decimate")
        form = QFormLayout(box)

        self.spin_decimate_factor = QSpinBox()
        self.spin_decimate_factor.setRange(1, 1000)
        self.spin_decimate_factor.setValue(2)
        form.addRow("Factor:", self.spin_decimate_factor)

        self.spin_nyq_rat = QDoubleSpinBox()
        self.spin_nyq_rat.setRange(2.0, 100.0)
        self.spin_nyq_rat.setValue(2.5)
        form.addRow("Nyquist ratio:", self.spin_nyq_rat)

        self.chk_decimate_highpass = QCheckBox("Highpass [Hz]")
        self.spin_decimate_highpass = QDoubleSpinBox()
        self.spin_decimate_highpass.setRange(0.0001, 1e6)
        self.spin_decimate_highpass.setEnabled(False)
        self.chk_decimate_highpass.toggled.connect(self.spin_decimate_highpass.setEnabled)
        row_hp = QHBoxLayout()
        row_hp.addWidget(self.chk_decimate_highpass)
        row_hp.addWidget(self.spin_decimate_highpass)
        form.addRow(row_hp)

        self.combo_decimate_ftype = QComboBox()
        self.combo_decimate_ftype.addItems(['cheby1', 'butter', 'cheby2', 'ellip', 'bessel'])
        form.addRow("Filter type:", self.combo_decimate_ftype)

        btn_decimate = QPushButton("Decimate")
        btn_decimate.clicked.connect(self._on_decimate)
        form.addRow(btn_decimate)

        return box

    def _on_decimate(self):
        decimate_factor = self.spin_decimate_factor.value()
        nyq_rat = self.spin_nyq_rat.value()
        highpass = self.spin_decimate_highpass.value() if self.chk_decimate_highpass.isChecked() else None
        filter_type = self.combo_decimate_ftype.currentText()

        self.prep_signals.decimate_signals(
            decimate_factor=decimate_factor, nyq_rat=nyq_rat,
            highpass=highpass, filter_type=filter_type)

        self._after_signal_mutation()

    def _after_signal_mutation(self):
        # sampling_rate/duration change (decimation) and cached spectra are
        # cleared by every mutating call above -> status and plot both need
        # a refresh.
        self._refresh_status()
        self._update_plot()

    # -- diagram controls -------------------------------------------------
    def _build_plot_controls_box(self):
        box = QGroupBox("Diagram")
        layout = QVBoxLayout(box)

        self.combo_diagram = QComboBox()
        self.combo_diagram.addItems(['Time series', 'Correlation', 'PSD', 'Signals overview'])
        self.combo_diagram.currentIndexChanged.connect(self._on_diagram_type_changed)
        layout.addWidget(self.combo_diagram)

        self.stack_params = QStackedWidget()
        self.stack_params.addWidget(self._build_timeseries_params())
        self.stack_params.addWidget(self._build_correlation_params())
        self.stack_params.addWidget(self._build_psd_params())
        self.stack_params.addWidget(self._build_overview_params())
        layout.addWidget(self.stack_params)

        btn_refresh = QPushButton("Refresh plot")
        btn_refresh.clicked.connect(self._update_plot)
        layout.addWidget(btn_refresh)

        return box

    def _on_diagram_type_changed(self, index):
        self.stack_params.setCurrentIndex(index)
        self._update_plot()

    def _build_timeseries_params(self):
        widget = QWidget()
        form = QFormLayout(widget)
        self.combo_ts_scale = QComboBox()
        self.combo_ts_scale.addItems(['time', 'samples'])
        self.combo_ts_scale.currentIndexChanged.connect(self._update_plot)
        form.addRow("Scale:", self.combo_ts_scale)
        return widget

    def _build_correlation_params(self):
        widget = QWidget()
        form = QFormLayout(widget)

        self.chk_corr_auto_mlags = QCheckBox("Auto")
        self.chk_corr_auto_mlags.setChecked(True)
        self.spin_corr_mlags = QSpinBox()
        self.spin_corr_mlags.setRange(1, 10 ** 8)
        self.spin_corr_mlags.setEnabled(False)
        self.chk_corr_auto_mlags.toggled.connect(lambda c: self.spin_corr_mlags.setEnabled(not c))
        row = QHBoxLayout()
        row.addWidget(self.chk_corr_auto_mlags)
        row.addWidget(self.spin_corr_mlags)
        form.addRow("m_lags:", row)

        self.combo_corr_scale = QComboBox()
        self.combo_corr_scale.addItems(['lags', 'samples'])
        form.addRow("Scale:", self.combo_corr_scale)

        self.combo_corr_method = QComboBox()
        self.combo_corr_method.addItems(['auto', 'welch', 'blackman-tukey'])
        form.addRow("Method:", self.combo_corr_method)

        self.combo_corr_scale.currentIndexChanged.connect(self._update_plot)
        self.combo_corr_method.currentIndexChanged.connect(self._update_plot)
        self.chk_corr_auto_mlags.toggled.connect(self._update_plot)
        self.spin_corr_mlags.valueChanged.connect(self._update_plot)

        return widget

    def _build_psd_params(self):
        widget = QWidget()
        form = QFormLayout(widget)

        self.chk_psd_auto_nlines = QCheckBox("Auto")
        self.chk_psd_auto_nlines.setChecked(True)
        self.spin_psd_nlines = QSpinBox()
        self.spin_psd_nlines.setRange(2, 10 ** 8)
        self.spin_psd_nlines.setValue(512)
        self.spin_psd_nlines.setEnabled(False)
        self.chk_psd_auto_nlines.toggled.connect(lambda c: self.spin_psd_nlines.setEnabled(not c))
        row = QHBoxLayout()
        row.addWidget(self.chk_psd_auto_nlines)
        row.addWidget(self.spin_psd_nlines)
        form.addRow("n_lines:", row)

        self.combo_psd_scale = QComboBox()
        self.combo_psd_scale.addItems(['db', 'power', 'rms', 'phase', 'svd'])
        self.combo_psd_scale.currentTextChanged.connect(self._on_psd_scale_changed)
        form.addRow("Scale:", self.combo_psd_scale)

        self.combo_psd_method = QComboBox()
        self.combo_psd_method.addItems(['auto', 'welch', 'blackman-tukey'])
        form.addRow("Method:", self.combo_psd_method)

        self.combo_psd_method.currentIndexChanged.connect(self._update_plot)
        self.chk_psd_auto_nlines.toggled.connect(self._update_plot)
        self.spin_psd_nlines.valueChanged.connect(self._update_plot)

        return widget

    def _on_psd_scale_changed(self, scale):
        # SignalPlot.plot_psd(scale='svd') ignores channel/reference
        # selection (and warns if refs are given), so grey those controls
        # out rather than let the user set values that get silently ignored.
        is_svd = (scale == 'svd')
        self.list_channels.setEnabled(not is_svd)
        self.chk_auto_ref.setEnabled(not is_svd)
        self._update_plot()

    def _build_overview_params(self):
        widget = QWidget()
        form = QFormLayout(widget)

        self.chk_overview_per_channel = QCheckBox("Separate axes per channel")
        self.chk_overview_per_channel.setChecked(True)
        form.addRow(self.chk_overview_per_channel)

        self.combo_overview_timescale = QComboBox()
        self.combo_overview_timescale.addItems(['time', 'samples', 'lags'])
        form.addRow("Time axis:", self.combo_overview_timescale)

        self.combo_overview_psdscale = QComboBox()
        self.combo_overview_psdscale.addItems(['db', 'power', 'rms', 'phase', 'svd'])
        form.addRow("Spectrum scale:", self.combo_overview_psdscale)

        # plot_signals() forwards n_lines/method to PreProcessSignals.psd()
        # and .correlation(). Both fall back to self._last_meth, which every
        # mutating call (filter/decimate/offset/...) resets to None via
        # _clear_spectral_values() -- and n_lines=None on a genuinely first
        # call raises. So this panel needs its own n_lines/method, same as
        # the PSD panel, rather than relying on whatever was last cached.
        self.chk_overview_auto_nlines = QCheckBox("Auto")
        self.chk_overview_auto_nlines.setChecked(False)
        self.spin_overview_nlines = QSpinBox()
        self.spin_overview_nlines.setRange(2, 10 ** 8)
        self.spin_overview_nlines.setValue(512)
        self.chk_overview_auto_nlines.toggled.connect(
            lambda c: self.spin_overview_nlines.setEnabled(not c))
        row_nlines = QHBoxLayout()
        row_nlines.addWidget(self.chk_overview_auto_nlines)
        row_nlines.addWidget(self.spin_overview_nlines)
        form.addRow("n_lines:", row_nlines)

        self.combo_overview_method = QComboBox()
        self.combo_overview_method.addItems(['welch', 'blackman-tukey'])
        form.addRow("Method:", self.combo_overview_method)

        self.chk_overview_per_channel.toggled.connect(self._update_plot)
        self.combo_overview_timescale.currentIndexChanged.connect(self._update_plot)
        self.combo_overview_psdscale.currentIndexChanged.connect(self._update_plot)
        self.chk_overview_auto_nlines.toggled.connect(self._update_plot)
        self.spin_overview_nlines.valueChanged.connect(self._update_plot)
        self.combo_overview_method.currentIndexChanged.connect(self._update_plot)

        return widget

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _update_plot(self):
        diagram = self.combo_diagram.currentText()
        self.canvas.figure.clear()
        try:
            if diagram == 'Time series':
                self._plot_timeseries()
            elif diagram == 'Correlation':
                self._plot_correlation()
            elif diagram == 'PSD':
                self._plot_psd()
            elif diagram == 'Signals overview':
                self._plot_overview()
        except Exception as exc:
            logger.exception("Plotting failed")
            self.canvas.figure.clear()
            ax = self.canvas.figure.subplots()
            ax.text(0.5, 0.5, f"Could not plot:\n{exc}",
                    ha='center', va='center', wrap=True, color='crimson',
                    transform=ax.transAxes)
        self.canvas.draw_idle()

    def _plot_timeseries(self):
        ax = self.canvas.figure.subplots()
        scale = self.combo_ts_scale.currentText()
        self.signal_plot.plot_timeseries(channels=self._selected_channels(), ax=ax, scale=scale)
        ax.legend(fontsize='small')

    def _plot_correlation(self):
        ax = self.canvas.figure.subplots()
        m_lags = None if self.chk_corr_auto_mlags.isChecked() else self.spin_corr_mlags.value()
        scale = self.combo_corr_scale.currentText()
        method = self.combo_corr_method.currentText()
        method = None if method == 'auto' else method
        self.signal_plot.plot_correlation(
            m_lags=m_lags, channels=self._selected_channels(), ax=ax,
            scale=scale, refs=self._selected_refs(), method=method)
        ax.legend(fontsize='small')

    def _plot_psd(self):
        ax = self.canvas.figure.subplots()
        n_lines = None if self.chk_psd_auto_nlines.isChecked() else self.spin_psd_nlines.value()
        scale = self.combo_psd_scale.currentText()
        method = self.combo_psd_method.currentText()
        method = None if method == 'auto' else method
        is_svd = (scale == 'svd')
        channels = None if is_svd else self._selected_channels()
        refs = None if is_svd else self._selected_refs()
        self.signal_plot.plot_psd(
            n_lines=n_lines, channels=channels, ax=ax,
            scale=scale, refs=refs, method=method)
        if not is_svd:
            ax.legend(fontsize='small')

    def _plot_overview(self):
        channels = self._selected_channels()
        channel_numbers = channels if channels is not None else \
            list(range(self.prep_signals.num_analised_channels))
        num_channels = len(channel_numbers)
        per_channel_axes = self.chk_overview_per_channel.isChecked()
        psd_scale = self.combo_overview_psdscale.currentText()
        timescale = self.combo_overview_timescale.currentText()
        n_lines = None if self.chk_overview_auto_nlines.isChecked() else self.spin_overview_nlines.value()
        method = self.combo_overview_method.currentText()

        axest, axesf = self._make_overview_axes(per_channel_axes, psd_scale, num_channels)

        self.signal_plot.plot_signals(
            channels=channels, axest=axest, axesf=axesf,
            per_channel_axes=per_channel_axes,
            timescale=timescale, psd_scale=psd_scale,
            n_lines=n_lines, method=method)

    def _make_overview_axes(self, per_channel_axes, psd_scale, num_channels):
        """Build axest/axesf on self.canvas.figure, mirroring
        SignalPlot._create_per_channel_axes / _create_shared_axes so the
        result renders on our embedded canvas instead of the stray pyplot
        figures those helpers would create if we let plot_signals build its
        own axes."""
        fig = self.canvas.figure
        if per_channel_axes:
            if psd_scale != 'svd':
                axes = fig.subplots(nrows=num_channels, ncols=2,
                                    sharey='col', sharex='col', squeeze=False)
                axest = axes[:, 0]
                axesf = axes[:, 1]
            else:
                nxn = int(np.ceil(np.sqrt(num_channels)))
                nrows_t = int(np.ceil(num_channels / nxn))
                gs = fig.add_gridspec(nrows_t + 1, nxn)
                axest = np.array([fig.add_subplot(gs[i // nxn, i % nxn]) for i in range(num_channels)])
                ax_svd = fig.add_subplot(gs[-1, :])
                axesf = np.repeat(ax_svd, num_channels)
        else:
            ax_t = fig.add_subplot(2, 1, 1)
            ax_f = fig.add_subplot(2, 1, 2)
            axest = np.repeat(ax_t, num_channels)
            axesf = np.repeat(ax_f, num_channels)
        return axest, axesf


def build_demo_prep_signals():
    """Small synthetic PreProcessSignals for manual GUI testing (no real
    measurement data required)."""
    _t, y, _omegas, _psd, _corr = SDOF_ambient()
    ch0 = y[:, np.newaxis]
    ch1 = ch0 * 0.6 + 0.05  # second channel so channel selection has something to show
    signals = np.hstack([ch0, ch1])
    return PreProcessSignals(signals, sampling_rate=128, channel_headers=['ch0', 'ch1'])


def start_preprocess_gui(prep_signals):
    global app
    app = QApplication.instance() or QApplication(sys.argv)

    form = PreProcessSignalsGUI(prep_signals)
    form.resize(1250, 820)

    loop = QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec()
    return


def main():
    prep_signals = build_demo_prep_signals()
    start_preprocess_gui(prep_signals)


if __name__ == '__main__':
    main()
