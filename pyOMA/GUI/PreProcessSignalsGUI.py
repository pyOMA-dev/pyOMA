# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive PyQt6 GUI for pyOMA.core.PreProcessingTools.

Wraps :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals` and
:class:`~pyOMA.core.PreProcessingTools.SignalPlot`: every pre-processing
action mutates the ``PreProcessSignals`` instance in place, and the two
plot widgets re-render (time domain on top, frequency domain at the
bottom) against its current state.

Widget layout lives in ``ui/preprocess_signals.ui`` (compiled to
``generated/ui_preprocess_signals.py`` by ``scripts/build_ui.py``); this
module only wires signals/slots and holds the plotting/pre-processing logic.
"""
import sys
import logging

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QComboBox, QCheckBox, QTableWidgetItem,
)
from PyQt6.QtCore import Qt, QEventLoop, QTimer

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from .generated.ui_preprocess_signals import Ui_PreProcessSignalsGUI
from ..core.PreProcessingTools import PreProcessSignals, SignalPlot, SDOF_ambient

logger = logging.getLogger(__name__)

app = None

_CHANNEL_TYPES = ('Acceleration', 'Velocity', 'Displacement')
_CHANNEL_TYPE_ATTR = {
    'Acceleration': 'accel_channels',
    'Velocity': 'velo_channels',
    'Displacement': 'disp_channels',
}


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
        self._wire_canvases()
        self._wire_channel_box()
        self._wire_preprocessing_box()
        self._wire_diagram_box()

        self._refresh_channel_table()
        self._refresh_status()
        self._update_both_plots()
        self.show()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_canvases(self):
        self.toolbar_time = NavigationToolbar2QT(self.canvas_time, self)
        self.plot_layout.insertWidget(0, self.toolbar_time)
        self.toolbar_freq = NavigationToolbar2QT(self.canvas_freq, self)
        self.plot_layout.insertWidget(2, self.toolbar_freq)

    def _wire_channel_box(self):
        self.channel_table.itemSelectionChanged.connect(self._update_both_plots)
        self.btn_select_all.clicked.connect(self.channel_table.selectAll)
        self.btn_select_none.clicked.connect(self.channel_table.clearSelection)
        self.btn_delete_channels.clicked.connect(self._on_delete_channels)
        self.chk_auto_ref.stateChanged.connect(self._update_both_plots)

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
        self.combo_time_diagram.currentIndexChanged.connect(self._on_time_diagram_changed)

        self.combo_ts_scale.currentIndexChanged.connect(self._update_time_plot)

        self.chk_corr_auto_mlags.toggled.connect(lambda c: self.spin_corr_mlags.setEnabled(not c))
        self.combo_corr_scale.currentIndexChanged.connect(self._update_time_plot)
        self.combo_corr_method.currentIndexChanged.connect(self._update_time_plot)
        self.chk_corr_auto_mlags.toggled.connect(self._update_time_plot)
        self.spin_corr_mlags.valueChanged.connect(self._update_time_plot)

        self.chk_psd_auto_nlines.toggled.connect(lambda c: self.spin_psd_nlines.setEnabled(not c))
        self.combo_psd_scale.currentTextChanged.connect(self._on_psd_scale_changed)
        self.combo_psd_method.currentIndexChanged.connect(self._update_freq_plot)
        self.chk_psd_auto_nlines.toggled.connect(self._update_freq_plot)
        self.spin_psd_nlines.valueChanged.connect(self._update_freq_plot)

        self.btn_refresh_plots.clicked.connect(self._update_both_plots)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _refresh_status(self):
        self.lbl_num_channels.setText(str(self.prep_signals.num_analised_channels))
        self.lbl_sampling_rate.setText(f"{self.prep_signals.sampling_rate:.4g}")
        self.lbl_duration.setText(f"{self.prep_signals.duration:.4g}")

    # ------------------------------------------------------------------
    # Channel table: selection, type (accel/velo/disp), reference, delete
    # ------------------------------------------------------------------
    def _channel_type_text(self, channel):
        if channel in self.prep_signals.accel_channels:
            return 'Acceleration'
        if channel in self.prep_signals.velo_channels:
            return 'Velocity'
        if channel in self.prep_signals.disp_channels:
            return 'Displacement'
        return 'Acceleration'

    def _refresh_channel_table(self):
        table = self.channel_table
        table.blockSignals(True)
        table.setRowCount(0)
        for channel in range(self.prep_signals.num_analised_channels):
            row = table.rowCount()
            table.insertRow(row)

            name_item = QTableWidgetItem(
                f"{channel}: {self.prep_signals.channel_headers[channel]}")
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name_item)

            combo = QComboBox()
            combo.addItems(_CHANNEL_TYPES)
            combo.setCurrentText(self._channel_type_text(channel))
            combo.currentTextChanged.connect(
                lambda text, ch=channel: self._on_channel_type_changed(ch, text))
            table.setCellWidget(row, 1, combo)

            checkbox = QCheckBox()
            checkbox.setChecked(channel in self.prep_signals.ref_channels)
            checkbox.toggled.connect(
                lambda checked, ch=channel: self._on_channel_ref_toggled(ch, checked))
            table.setCellWidget(row, 2, checkbox)
        table.selectAll()
        table.blockSignals(False)

    def _selected_channels(self):
        rows = [index.row() for index in self.channel_table.selectionModel().selectedRows()]
        if not rows:
            return None  # PreProcessSignals/SignalPlot interpret None as "all channels"
        return sorted(rows)

    def _selected_refs(self):
        return 'auto' if self.chk_auto_ref.isChecked() else None

    def _on_channel_type_changed(self, channel, type_text):
        attr = _CHANNEL_TYPE_ATTR[type_text]
        current = list(getattr(self.prep_signals, attr))
        if channel not in current:
            current.append(channel)
            setattr(self.prep_signals, attr, current)
        # Rebuilding the table now would destroy the combo box still
        # emitting this very signal - defer to the next event loop turn.
        QTimer.singleShot(0, self._refresh_channel_table)

    def _on_channel_ref_toggled(self, channel, checked):
        current = list(self.prep_signals.ref_channels)
        if checked and channel not in current:
            current.append(channel)
        elif not checked and channel in current:
            current.remove(channel)
        self.prep_signals.ref_channels = sorted(current)
        self._update_both_plots()

    def _on_delete_channels(self):
        rows = sorted(index.row() for index in self.channel_table.selectionModel().selectedRows())
        if not rows:
            QMessageBox.warning(self, "Delete channels", "Select at least one channel first.")
            return
        if len(rows) >= self.prep_signals.num_analised_channels:
            QMessageBox.warning(self, "Delete channels", "Cannot delete all channels.")
            return
        self.prep_signals.delete_channels(rows)
        self._refresh_channel_table()
        self._refresh_status()
        self._update_both_plots()

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
            self.canvas_time.figure.clear()
            self.canvas_freq.figure.clear()
            plot_ax = [self.canvas_time.figure.subplots(), self.canvas_freq.figure.subplots()]

        self.prep_signals.filter_signals(
            lowpass=lowpass, highpass=highpass, overwrite=overwrite,
            order=order, ftype=ftype, RpRs=RpRs, plot_ax=plot_ax)

        self._refresh_status()
        if show_response:
            self.canvas_time.draw_idle()  # filter response is shown instead of the signal plots this time
            self.canvas_freq.draw_idle()
        else:
            self._update_both_plots()

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
        # cleared by every mutating call above -> status and both plots need
        # a refresh.
        self._refresh_status()
        self._update_both_plots()

    # ------------------------------------------------------------------
    # Diagram controls
    # ------------------------------------------------------------------
    def _on_time_diagram_changed(self, index):
        self.stack_time_params.setCurrentIndex(index)
        self._update_time_plot()

    def _on_psd_scale_changed(self, scale):
        # SignalPlot.plot_psd(scale='svd') ignores channel/reference
        # selection (and warns if refs are given), so grey those controls
        # out rather than let the user set values that get silently ignored.
        is_svd = (scale == 'svd')
        self.channel_table.setEnabled(not is_svd)
        self.chk_auto_ref.setEnabled(not is_svd)
        self._update_freq_plot()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _update_both_plots(self):
        self._update_time_plot()
        self._update_freq_plot()

    def _update_time_plot(self):
        diagram = self.combo_time_diagram.currentText()
        self.canvas_time.figure.clear()
        try:
            if diagram == 'Time series':
                self._plot_timeseries()
            elif diagram == 'Correlation':
                self._plot_correlation()
        except Exception as exc:
            logger.exception("Time-domain plotting failed")
            self.canvas_time.figure.clear()
            ax = self.canvas_time.figure.subplots()
            ax.text(0.5, 0.5, f"Could not plot:\n{exc}",
                    ha='center', va='center', wrap=True, color='crimson',
                    transform=ax.transAxes)
        self.canvas_time.draw_idle()

    def _update_freq_plot(self):
        self.canvas_freq.figure.clear()
        try:
            self._plot_psd()
        except Exception as exc:
            logger.exception("Frequency-domain plotting failed")
            self.canvas_freq.figure.clear()
            ax = self.canvas_freq.figure.subplots()
            ax.text(0.5, 0.5, f"Could not plot:\n{exc}",
                    ha='center', va='center', wrap=True, color='crimson',
                    transform=ax.transAxes)
        self.canvas_freq.draw_idle()

    def _plot_timeseries(self):
        ax = self.canvas_time.figure.subplots()
        scale = self.combo_ts_scale.currentText()
        self.signal_plot.plot_timeseries(channels=self._selected_channels(), ax=ax, scale=scale)
        ax.legend(fontsize='small')

    def _plot_correlation(self):
        ax = self.canvas_time.figure.subplots()
        m_lags = None if self.chk_corr_auto_mlags.isChecked() else self.spin_corr_mlags.value()
        scale = self.combo_corr_scale.currentText()
        method = self.combo_corr_method.currentText()
        method = None if method == 'auto' else method
        self.signal_plot.plot_correlation(
            m_lags=m_lags, channels=self._selected_channels(), ax=ax,
            scale=scale, refs=self._selected_refs(), method=method)
        ax.legend(fontsize='small')

    def _plot_psd(self):
        ax = self.canvas_freq.figure.subplots()
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
