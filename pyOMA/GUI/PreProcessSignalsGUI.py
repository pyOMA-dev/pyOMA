# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive PyQt6 GUI for pyOMA.core.PreProcessingTools.

Wraps :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals` and
:class:`~pyOMA.core.PreProcessingTools.SignalPlot`: every pre-processing
action mutates the ``PreProcessSignals`` instance in place, and the plot
panel re-renders one of the ``SignalPlot`` diagrams against its current
state.

Widget layout lives in ``ui/preprocess_signals.ui`` (compiled to
``generated/ui_preprocess_signals.py`` by ``scripts/build_ui.py``); this
module only wires signals/slots and holds the plotting/pre-processing logic.
"""
import sys
import logging

import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QEventLoop

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from .generated.ui_preprocess_signals import Ui_PreProcessSignalsGUI
from ..core.PreProcessingTools import PreProcessSignals, SignalPlot, SDOF_ambient

logger = logging.getLogger(__name__)

app = None


class PreProcessSignalsGUI(QMainWindow, Ui_PreProcessSignalsGUI):
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

        self.setupUi(self)
        self._wire_canvas()
        self._wire_channel_box()
        self._wire_preprocessing_box()
        self._wire_diagram_box()

        self._refresh_channel_list()
        self._refresh_status()
        self._update_plot()
        self.show()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_canvas(self):
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plot_layout.insertWidget(0, self.toolbar)

    def _wire_channel_box(self):
        self.list_channels.itemSelectionChanged.connect(self._update_plot)
        self.btn_select_all.clicked.connect(self.list_channels.selectAll)
        self.btn_select_none.clicked.connect(self.list_channels.clearSelection)
        self.chk_auto_ref.stateChanged.connect(self._update_plot)

    def _wire_preprocessing_box(self):
        self.btn_correct_offset.clicked.connect(self._on_correct_offset)
        self.btn_precondition.clicked.connect(self._on_precondition)

        self.btn_add_noise.clicked.connect(self._on_add_noise)

        self.chk_lowpass.toggled.connect(self.spin_lowpass.setEnabled)
        self.chk_highpass.toggled.connect(self.spin_highpass.setEnabled)
        self.combo_ftype.currentTextChanged.connect(self._on_ftype_changed)
        self.chk_auto_order.toggled.connect(lambda checked: self.spin_order.setEnabled(not checked))
        self.btn_apply_filter.clicked.connect(self._on_filter)

        self.chk_decimate_highpass.toggled.connect(self.spin_decimate_highpass.setEnabled)
        self.btn_decimate.clicked.connect(self._on_decimate)

    def _wire_diagram_box(self):
        self.combo_diagram.currentIndexChanged.connect(self._on_diagram_type_changed)
        self.btn_refresh_plot.clicked.connect(self._update_plot)

        self.combo_ts_scale.currentIndexChanged.connect(self._update_plot)

        self.chk_corr_auto_mlags.toggled.connect(lambda c: self.spin_corr_mlags.setEnabled(not c))
        self.combo_corr_scale.currentIndexChanged.connect(self._update_plot)
        self.combo_corr_method.currentIndexChanged.connect(self._update_plot)
        self.chk_corr_auto_mlags.toggled.connect(self._update_plot)
        self.spin_corr_mlags.valueChanged.connect(self._update_plot)

        self.chk_psd_auto_nlines.toggled.connect(lambda c: self.spin_psd_nlines.setEnabled(not c))
        self.combo_psd_scale.currentTextChanged.connect(self._on_psd_scale_changed)
        self.combo_psd_method.currentIndexChanged.connect(self._update_plot)
        self.chk_psd_auto_nlines.toggled.connect(self._update_plot)
        self.spin_psd_nlines.valueChanged.connect(self._update_plot)

        self.chk_overview_per_channel.toggled.connect(self._update_plot)
        self.combo_overview_timescale.currentIndexChanged.connect(self._update_plot)
        self.combo_overview_psdscale.currentIndexChanged.connect(self._update_plot)
        self.chk_overview_auto_nlines.toggled.connect(
            lambda c: self.spin_overview_nlines.setEnabled(not c))
        self.chk_overview_auto_nlines.toggled.connect(self._update_plot)
        self.spin_overview_nlines.valueChanged.connect(self._update_plot)
        self.combo_overview_method.currentIndexChanged.connect(self._update_plot)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _refresh_status(self):
        self.lbl_num_channels.setText(str(self.prep_signals.num_analised_channels))
        self.lbl_sampling_rate.setText(f"{self.prep_signals.sampling_rate:.4g}")
        self.lbl_duration.setText(f"{self.prep_signals.duration:.4g}")

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------
    def _refresh_channel_list(self):
        self.list_channels.blockSignals(True)
        self.list_channels.clear()
        for idx, name in enumerate(self.prep_signals.channel_headers):
            self.list_channels.addItem(f"{idx}: {name}")
        self.list_channels.selectAll()
        self.list_channels.blockSignals(False)

    def _selected_channels(self):
        rows = [self.list_channels.row(item) for item in self.list_channels.selectedItems()]
        if not rows:
            return None  # PreProcessSignals/SignalPlot interpret None as "all channels"
        return sorted(rows)

    def _selected_refs(self):
        return 'auto' if self.chk_auto_ref.isChecked() else None

    # ------------------------------------------------------------------
    # Pre-processing actions
    # ------------------------------------------------------------------
    def _on_correct_offset(self):
        self.prep_signals.correct_offset()
        self._after_signal_mutation()

    def _on_precondition(self):
        self.prep_signals.precondition_signals(
            method=self.combo_precondition_method.currentText())
        self._after_signal_mutation()

    def _on_add_noise(self):
        amplitude = self.spin_noise_amplitude.value()
        snr = self.spin_noise_snr.value()
        if amplitude == 0 and snr == 0:
            QMessageBox.warning(self, "Add noise", "Set a non-zero amplitude or SNR first.")
            return
        self.prep_signals.add_noise(amplitude=amplitude, snr=snr)
        self._after_signal_mutation()

    def _on_ftype_changed(self, ftype):
        needs_rprs = ftype in ('cheby1', 'cheby2', 'ellip')
        self.spin_rp.setEnabled(needs_rprs)
        self.spin_rs.setEnabled(needs_rprs)
        self.lbl_rp.setEnabled(needs_rprs)
        self.lbl_rs.setEnabled(needs_rprs)

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

    # ------------------------------------------------------------------
    # Diagram controls
    # ------------------------------------------------------------------
    def _on_diagram_type_changed(self, index):
        self.stack_params.setCurrentIndex(index)
        self._update_plot()

    def _on_psd_scale_changed(self, scale):
        # SignalPlot.plot_psd(scale='svd') ignores channel/reference
        # selection (and warns if refs are given), so grey those controls
        # out rather than let the user set values that get silently ignored.
        is_svd = (scale == 'svd')
        self.list_channels.setEnabled(not is_svd)
        self.chk_auto_ref.setEnabled(not is_svd)
        self._update_plot()

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
