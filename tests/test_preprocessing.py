"""Tests for PreProcessingTools: GeometryProcessor and PreProcessSignals."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import GeometryProcessor, PreProcessSignals

TEST_FILES = Path(__file__).parent / 'files'


# ── GeometryProcessor ────────────────────────────────────────────────────────

class TestGeometryProcessor:
    def test_load_geometry_returns_instance(self, geometry_data):
        assert isinstance(geometry_data, GeometryProcessor)

    def test_nodes_are_loaded(self, geometry_data):
        assert len(geometry_data.nodes) > 0

    def test_lines_are_loaded(self, geometry_data):
        assert len(geometry_data.lines) > 0

    def test_node_coordinates_are_3d(self, geometry_data):
        for name, coords in geometry_data.nodes.items():
            assert len(coords) == 3, f"Node {name} has {len(coords)} coords, expected 3"

    def test_add_node_manually(self):
        geo = GeometryProcessor()
        geo.add_node('99', [1.0, 2.0, 3.0])
        assert '99' in geo.nodes
        assert tuple(geo.nodes['99']) == (1.0, 2.0, 3.0)

    def test_load_geometry_without_lines(self):
        geo = GeometryProcessor.load_geometry(nodes_file=TEST_FILES / 'grid.txt')
        assert len(geo.nodes) > 0
        assert len(geo.lines) == 0


# ── PreProcessSignals construction ────────────────────────────────────────────

class TestPreProcessSignalsInit:
    def test_basic_construction(self, prep_signals):
        assert prep_signals.sampling_rate == 128
        assert prep_signals.signals.shape[1] == 6
        assert prep_signals.total_time_steps == 8192

    def test_ref_channels_assigned(self, prep_signals):
        assert 5 in prep_signals.ref_channels

    def test_num_analised_channels(self, prep_signals):
        assert prep_signals.num_analised_channels == 6

    def test_num_ref_channels(self, prep_signals):
        assert prep_signals.num_ref_channels == 1

    def test_duration(self, prep_signals):
        expected = 8192 / 128
        assert abs(prep_signals.duration - expected) < 1e-10

    def test_dt(self, prep_signals):
        assert abs(prep_signals.dt - 1 / 128) < 1e-10

    def test_time_axis_length(self, prep_signals):
        assert prep_signals.t.shape == (8192,)

    def test_signals_shape_must_be_n_gt_channels(self):
        with pytest.raises((ValueError, AssertionError)):
            bad = np.random.randn(3, 10)
            PreProcessSignals(bad, 128)

    def test_init_from_config_real_data(self, prep_signals_real):
        assert isinstance(prep_signals_real, PreProcessSignals)
        assert prep_signals_real.sampling_rate == 256           # from config: Sampling Rate [Hz]: 256
        assert prep_signals_real.num_analised_channels == 5     # 6 channels − channel 5 deleted
        assert set(prep_signals_real.ref_channels) == {3, 4}   # Reference Channels: 3 4
        assert set(prep_signals_real.accel_channels) == {3, 4} # Accel. Channels: 3 4 5 → 5 deleted
        assert set(prep_signals_real.velo_channels) == {0, 1, 2}  # Velo. Channels: 0 1 2

    def test_channel_quantity_defaults_to_accel(self):
        sig = np.random.randn(1000, 4)
        ps = PreProcessSignals(sig, 100)
        # all channels should default to acceleration
        assert len(ps.accel_channels) == 4
        assert len(ps.velo_channels) == 0
        assert len(ps.disp_channels) == 0


# ── Signal pre-processing operations ─────────────────────────────────────────

class TestCorrectOffset:
    def test_mean_is_near_zero_after_offset_correction(self, prep_signals):
        prep_signals.signals[:, 0] += 5.0   # add artificial offset
        prep_signals.correct_offset()
        # correct_offset modifies self.signals in-place
        means = np.mean(prep_signals.signals, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)


class TestUndoStubs:
    """Single-step undo is not implemented yet - these lock in the stub
    contract (always-unavailable, NotImplementedError, no crash from the
    save_undo_snapshot() call sites already wired into the mutating
    methods) so the real implementation can land later without touching
    every call site again."""

    def test_undo_available_is_false(self, prep_signals):
        assert prep_signals.undo_available is False

    def test_undo_raises_not_implemented(self, prep_signals):
        with pytest.raises(NotImplementedError):
            prep_signals.undo()

    def test_save_undo_snapshot_is_a_harmless_noop(self, prep_signals):
        prep_signals.save_undo_snapshot()
        assert prep_signals.undo_available is False

    def test_mutating_methods_still_work_with_snapshot_call_sites_wired_in(self, prep_signals):
        prep_signals.correct_offset()
        prep_signals.add_noise(amplitude=0.01)
        prep_signals.filter_signals(lowpass=10.0)
        prep_signals.decimate_signals(2)
        prep_signals.delete_channels(0)
        assert prep_signals.undo_available is False


class TestChanDofAccessors:
    """get_chan_dof/set_chan_dof/remove_chan_dof replace the old, broken
    take_chan_dof (indexed into chan_dofs[j][2] as if it were a 3-tuple,
    but chan_dofs entries are flat [chan, node, az, elev, chan_name] -
    any real call would have raised TypeError; it had zero callers)."""

    def test_get_chan_dof_returns_none_when_unassigned(self, prep_signals):
        assert prep_signals.get_chan_dof(0) is None

    def test_set_then_get_chan_dof(self, prep_signals):
        prep_signals.set_chan_dof(0, 'N1', 90.0, 0.0)
        assert prep_signals.get_chan_dof(0) == ('N1', 90.0, 0.0)
        assert len(prep_signals.chan_dofs) == 1

    def test_set_chan_dof_replaces_not_duplicates(self, prep_signals):
        prep_signals.set_chan_dof(0, 'N1', 90.0, 0.0)
        prep_signals.set_chan_dof(0, 'N2', 0.0, 90.0)
        assert prep_signals.get_chan_dof(0) == ('N2', 0.0, 90.0)
        assert len(prep_signals.chan_dofs) == 1

    def test_set_chan_dof_stores_channel_name(self, prep_signals):
        prep_signals.set_chan_dof(0, 'N1', 90.0, 0.0)
        assert prep_signals.chan_dofs[0][4] == prep_signals.channel_headers[0]

    def test_remove_chan_dof(self, prep_signals):
        prep_signals.set_chan_dof(0, 'N1', 90.0, 0.0)
        prep_signals.set_chan_dof(1, 'N1', 180.0, 0.0)
        prep_signals.remove_chan_dof(0)
        assert prep_signals.get_chan_dof(0) is None
        assert prep_signals.get_chan_dof(1) == ('N1', 180.0, 0.0)

    def test_remove_chan_dof_on_unassigned_channel_is_a_noop(self, prep_signals):
        prep_signals.remove_chan_dof(0)
        assert prep_signals.chan_dofs == []


class TestDeleteChannels:
    def test_removes_channel_and_shrinks_signals(self, prep_signals):
        n_before = prep_signals.num_analised_channels
        prep_signals.delete_channels(0)
        assert prep_signals.num_analised_channels == n_before - 1
        assert prep_signals.signals.shape[1] == n_before - 1

    def test_reindexes_ref_channels(self, prep_signals):
        # SYN_REF = [5]; deleting channel 0 shifts the remaining ref down to 4
        assert prep_signals.ref_channels == [5]
        prep_signals.delete_channels(0)
        assert prep_signals.ref_channels == [4]

    def test_deleting_a_ref_channel_drops_it(self, prep_signals):
        prep_signals.delete_channels(5)  # the only reference channel
        assert prep_signals.ref_channels == []

    def test_channel_headers_and_factors_shrink(self, prep_signals):
        headers_before = list(prep_signals.channel_headers)
        prep_signals.delete_channels(1)
        assert len(prep_signals.channel_headers) == len(headers_before) - 1
        assert len(prep_signals.channel_factors) == len(prep_signals.channel_headers)

    def test_delete_by_name(self):
        sig = np.random.randn(1000, 4)
        ps = PreProcessSignals(sig, 100, channel_headers=['a', 'b', 'c', 'd'])
        ps.delete_channels('b')
        assert ps.channel_headers == ['a', 'c', 'd']

    def test_delete_multiple_at_once(self, prep_signals):
        n_before = prep_signals.num_analised_channels
        prep_signals.delete_channels([1, 4])
        assert prep_signals.num_analised_channels == n_before - 2

    def test_cannot_delete_all_channels(self, prep_signals):
        with pytest.raises(ValueError):
            prep_signals.delete_channels(list(range(prep_signals.num_analised_channels)))

    def test_clears_cached_spectra(self, prep_signals):
        prep_signals.psd(n_lines=256)
        assert prep_signals.psd_matrix is not None
        prep_signals.delete_channels(0)
        assert prep_signals.psd_matrix is None


class TestFilterSignals:
    def test_lowpass_reduces_high_frequency_energy(self, prep_signals):
        fs = prep_signals.sampling_rate
        # inject a pure sine at 90 % of Nyquist – should be attenuated
        f_high = 0.9 * fs / 2
        t = np.arange(prep_signals.total_time_steps) / fs
        prep_signals.signals[:, 0] += np.sin(2 * np.pi * f_high * t) * 100
        energy_before = np.var(prep_signals.signals[:, 0])
        prep_signals.filter_signals(lowpass=5.0)
        energy_after = np.var(prep_signals.signals_filtered[:, 0])
        assert energy_after < energy_before

    def test_filter_does_not_change_signal_length(self, prep_signals):
        n_before = prep_signals.total_time_steps
        prep_signals.filter_signals(lowpass=10.0)
        assert prep_signals.signals_filtered.shape[0] == n_before


class TestDecimateSignals:
    def test_halved_sampling_rate_after_decimate_by_2(self, prep_signals):
        fs_before = prep_signals.sampling_rate
        prep_signals.decimate_signals(2)
        assert prep_signals.sampling_rate == fs_before // 2

    def test_signal_length_halved_after_decimate_by_2(self, prep_signals):
        n_before = prep_signals.total_time_steps
        prep_signals.decimate_signals(2)
        assert prep_signals.total_time_steps == n_before // 2


# ── Spectral estimation ───────────────────────────────────────────────────────

class TestCorrelation:
    def test_welch_correlation_sets_m_lags(self, prep_signals):
        m = 100
        prep_signals.corr_welch(m_lags=m)
        assert prep_signals.m_lags == m

    def test_welch_corr_matrix_shape(self, prep_signals):
        n_ch = prep_signals.num_analised_channels
        n_ref = prep_signals.num_ref_channels
        m = 100
        prep_signals.corr_welch(m_lags=m)
        assert prep_signals.corr_matrix.shape == (n_ch, n_ref, m)

    def test_blackman_tukey_corr_sets_m_lags(self, prep_signals):
        m = 80
        prep_signals.corr_blackman_tukey(m_lags=m)
        assert prep_signals.m_lags == m

    def test_zero_lag_autocorr_is_positive(self, prep_signals):
        prep_signals.corr_welch(m_lags=100)
        # The zero-lag autocorrelation R(0) is the expected power of the signal.
        # It must be positive for any channel with non-zero variance.
        ref_idx = 0  # index into corr_matrix (only ref channels are stored)
        ref_ch = prep_signals.ref_channels[ref_idx]
        corr_0 = prep_signals.corr_matrix[ref_ch, ref_idx, 0]
        var = np.var(prep_signals.signals_filtered[:, ref_ch])
        assert var > 0, "Reference channel has zero variance – bad test data"
        assert corr_0 > 0, f"Zero-lag autocorrelation should be positive, got {corr_0}"


class TestPSD:
    def test_welch_psd_sets_n_lines(self, prep_signals):
        n = 256
        prep_signals.psd_welch(n_lines=n)
        assert prep_signals.n_lines is not None

    def test_psd_matrix_shape(self, prep_signals):
        n_ch = prep_signals.num_analised_channels
        n_ref = prep_signals.num_ref_channels
        n = 256
        prep_signals.psd_welch(n_lines=n)
        assert prep_signals.psd_matrix.shape == (n_ch, n_ref, n // 2 + 1)

    def test_psd_diagonal_is_real_and_positive(self, prep_signals):
        prep_signals.psd_welch(n_lines=256)
        for ch in range(prep_signals.num_ref_channels):
            diag = prep_signals.psd_matrix[ch, ch, :]
            assert np.all(np.isreal(diag)) or np.allclose(diag.imag, 0, atol=1e-12)
            assert np.all(diag.real >= 0)


# ── State persistence ─────────────────────────────────────────────────────────

class TestSaveLoadState:
    def test_round_trip_preserves_signals(self, prep_signals):
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            prep_signals.save_state(fname)
            loaded = PreProcessSignals.load_state(fname)
            np.testing.assert_array_equal(loaded.signals, prep_signals.signals)
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_round_trip_preserves_sampling_rate(self, prep_signals):
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            prep_signals.save_state(fname)
            loaded = PreProcessSignals.load_state(fname)
            assert loaded.sampling_rate == prep_signals.sampling_rate
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_round_trip_preserves_correlations(self, prep_signals):
        prep_signals.corr_welch(m_lags=80)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            prep_signals.save_state(fname)
            loaded = PreProcessSignals.load_state(fname)
            np.testing.assert_allclose(
                loaded.corr_matrix, prep_signals.corr_matrix, rtol=1e-10)
        finally:
            Path(fname).unlink(missing_ok=True)
