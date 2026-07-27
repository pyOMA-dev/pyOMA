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

    def test_take_parent_child_by_value_removes_matching_entry(self):
        # Regression: the matching loop used to seed its accumulator with
        # `b = False` and then compute `b = b and ...`, which is always
        # False - take_parent_child(ms=...) could never find a match.
        geo = GeometryProcessor()
        geo.add_node('1', [0.0, 0.0, 0.0])
        geo.add_node('2', [1.0, 0.0, 0.0])
        ms = ('1', 1.0, 0.0, 0.0, '2', 1.0, 0.0, 0.0)
        geo.add_parent_child(ms)
        assert ms in geo.parent_childs
        geo.take_parent_child(ms=ms)
        assert ms not in geo.parent_childs


class TestGeometryProcessorSaveLoad:
    def test_nodes_saver_round_trip(self, geometry_data, tmp_path):
        # nodes_loader() returns coordinates as lists; geometry_data.nodes
        # (populated via add_node()) stores them as tuples - compare as lists.
        fname = tmp_path / 'nodes.txt'
        GeometryProcessor.nodes_saver(fname, geometry_data.nodes)
        reloaded = GeometryProcessor.nodes_loader(fname)
        assert reloaded.keys() == geometry_data.nodes.keys()
        for name, coords in geometry_data.nodes.items():
            assert list(reloaded[name]) == list(coords)

    def test_lines_saver_round_trip(self, geometry_data, tmp_path):
        # lines_loader() returns tuples; geometry_data.lines (populated via
        # add_line()) stores lists - compare as lists.
        fname = tmp_path / 'lines.txt'
        GeometryProcessor.lines_saver(fname, geometry_data.lines)
        reloaded = GeometryProcessor.lines_loader(fname)
        assert [list(line) for line in reloaded] == [list(line) for line in geometry_data.lines]

    def test_parent_childs_saver_round_trip(self, geometry_data, tmp_path):
        fname = tmp_path / 'parent_childs.txt'
        GeometryProcessor.parent_childs_saver(fname, geometry_data.parent_childs)
        reloaded = GeometryProcessor.parent_childs_loader(fname)
        assert reloaded == geometry_data.parent_childs

    def test_save_geometry_round_trip(self, geometry_data, tmp_path):
        nodes_file = tmp_path / 'nodes.txt'
        lines_file = tmp_path / 'lines.txt'
        parent_childs_file = tmp_path / 'parent_childs.txt'
        geometry_data.save_geometry(nodes_file, lines_file, parent_childs_file)

        reloaded = GeometryProcessor.load_geometry(
            nodes_file=nodes_file,
            lines_file=lines_file,
            parent_childs_file=parent_childs_file,
        )
        assert reloaded.nodes == geometry_data.nodes
        assert reloaded.lines == geometry_data.lines
        assert reloaded.parent_childs == geometry_data.parent_childs

    def test_save_geometry_without_lines_or_parent_childs(self, geometry_data, tmp_path):
        nodes_file = tmp_path / 'nodes.txt'
        geometry_data.save_geometry(nodes_file)
        assert nodes_file.exists()
        assert not (tmp_path / 'lines.txt').exists()


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

    def test_signals_accepts_npy_filename(self, tmp_path):
        array = np.random.randn(1000, 4)
        fname = tmp_path / 'signals.npy'
        np.save(fname, array)
        from_file = PreProcessSignals(str(fname), 100)
        from_array = PreProcessSignals(array, 100)
        np.testing.assert_array_equal(from_file.signals, from_array.signals)
        assert from_file.sampling_rate == from_array.sampling_rate == 100

    def test_signals_accepts_npy_pathlib_path(self, tmp_path):
        array = np.random.randn(1000, 4)
        fname = tmp_path / 'signals.npy'
        np.save(fname, array)
        ps = PreProcessSignals(fname, 100)
        np.testing.assert_array_equal(ps.signals, array)

    def test_signals_npz_filename_rejected_with_pointer_to_load_state(self, tmp_path):
        fname = tmp_path / 'session.npz'
        np.savez(fname, **{'self.signals': np.random.randn(1000, 4)})
        with pytest.raises(ValueError, match='load_state'):
            PreProcessSignals(str(fname), 100)

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


class TestDefaultLoadMeasurementFile:
    """Regression: PreProcessSignals.load_measurement_file() used to always
    raise NotImplementedError unless a script/conftest monkeypatched it -
    which the 'pyoma' launcher's Load Config action never does, so Load
    Config was unusable out of the box. It must now warn and fall back to
    np.load() instead.

    Run in a subprocess: tests/conftest.py permanently monkeypatches
    PreProcessSignals.load_measurement_file = staticmethod(np.load) for the
    whole session (module-level, not via the `monkeypatch` fixture), so the
    real default is otherwise unreachable from within this test process.
    """

    def test_falls_back_to_npy_and_rejects_npz(self, tmp_path):
        import subprocess
        import sys
        array = np.random.randn(50, 3)
        npy_fname = tmp_path / 'signals.npy'
        np.save(npy_fname, array)
        npz_fname = tmp_path / 'signals.npz'
        np.savez(npz_fname, **{'self.signals': array})

        code = f"""
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
from pyOMA.core.PreProcessingTools import PreProcessSignals

loaded = PreProcessSignals.load_measurement_file({str(npy_fname)!r})
assert isinstance(loaded, np.ndarray), type(loaded)
assert loaded.shape == {array.shape!r}

try:
    PreProcessSignals.load_measurement_file({str(npz_fname)!r})
except ValueError as exc:
    assert 'load_state' in str(exc)
else:
    raise AssertionError("expected ValueError for .npz")

print("OK")
"""
        result = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'OK' in result.stdout
        assert 'has not been overridden' in result.stderr


class TestInitFromConfigDeleteChannels:
    """Regression tests for _apply_delete_channels (index-based deletion,
    chan_dofs entries attached opportunistically rather than required)."""

    CONFIG_TEMPLATE = (
        'Setup Name:\n{name}\n'
        'Sampling Rate [Hz]:\n{fs}\n'
        'Reference Channels:\n{ref}\n'
        'Delete Channels:\n{delete}\n'
        'Accel. Channels:\n{accel}\n'
        'Velo. Channels:\n{velo}\n'
        'Disp. Channels:\n{disp}\n'
    )

    def _write_config(self, tmp_path, **kwargs):
        fname = tmp_path / 'setup_info.txt'
        fname.write_text(self.CONFIG_TEMPLATE.format(**kwargs))
        return fname

    def test_delete_channels_without_chan_dofs_file_does_not_raise(self, tmp_path):
        meas_dir = TEST_FILES / 'measurement_1'
        conf_file = self._write_config(
            tmp_path, name='measurement_1', fs=256, ref='3 4', delete='5',
            accel='3 4 5', velo='0 1 2', disp='')

        prep_signals = PreProcessSignals.init_from_config(
            conf_file=conf_file,
            meas_file=meas_dir / 'measurement_1.npy',
            chan_dofs_file=None,
        )
        assert prep_signals.signals.shape[1] == len(prep_signals.channel_headers)
        assert prep_signals.num_analised_channels == 5

    def test_delete_channels_with_partial_chan_dofs_file(self, tmp_path):
        # chan_dofs file omits channel 2 (a surviving channel, not the
        # deleted one) - it must stay in headers/signals regardless.
        meas_dir = TEST_FILES / 'measurement_1'
        full_lines = (meas_dir / 'channel_dofs.txt').read_text().splitlines()
        partial_lines = [line for line in full_lines
                          if not line.startswith('2\t')]
        chan_dofs_file = tmp_path / 'channel_dofs_partial.txt'
        chan_dofs_file.write_text('\n'.join(partial_lines) + '\n')

        conf_file = self._write_config(
            tmp_path, name='measurement_1', fs=256, ref='3 4', delete='5',
            accel='3 4 5', velo='0 1 2', disp='')

        prep_signals = PreProcessSignals.init_from_config(
            conf_file=conf_file,
            meas_file=meas_dir / 'measurement_1.npy',
            chan_dofs_file=chan_dofs_file,
        )
        assert prep_signals.signals.shape[1] == len(prep_signals.channel_headers)
        assert prep_signals.num_analised_channels == 5
        # channel 2 survived deletion despite missing chan_dofs entry
        assert prep_signals.get_chan_dof(2) is None


class TestSaveConfig:
    def test_round_trip_preserves_values(self, prep_signals, tmp_path):
        fname = tmp_path / 'setup_info.txt'
        prep_signals.save_config(fname, delete_channels=[2, 3])

        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(fname)
        assert cfg.float('Sampling Rate [Hz]') == prep_signals.sampling_rate
        assert cfg.int_list('Reference Channels') == prep_signals.ref_channels
        assert cfg.int_list('Delete Channels') == [2, 3]
        assert cfg.int_list('Accel. Channels') == prep_signals.accel_channels
        assert cfg.int_list('Velo. Channels') == prep_signals.velo_channels
        assert cfg.int_list('Disp. Channels') == prep_signals.disp_channels

    def test_round_trip_without_delete_channels(self, prep_signals, tmp_path):
        fname = tmp_path / 'setup_info.txt'
        prep_signals.save_config(fname)

        from pyOMA.core.Helpers import ConfigFile
        cfg = ConfigFile(fname)
        assert cfg.int_list('Delete Channels') == []


class TestSaveChanDofs:
    def test_round_trip(self, prep_signals, tmp_path):
        prep_signals.add_chan_dofs([
            [0, '1', 0.0, 90.0, 'a'],
            [1, None, 45.0, 0.0, 'b'],
        ])
        fname = tmp_path / 'channel_dofs.txt'
        prep_signals.save_chan_dofs(fname)

        reloaded = PreProcessSignals.load_chan_dofs(fname)
        assert reloaded == prep_signals.chan_dofs


# ── Signal pre-processing operations ─────────────────────────────────────────

class TestCorrectOffset:
    def test_mean_is_near_zero_after_offset_correction(self, prep_signals):
        prep_signals.signals[:, 0] += 5.0   # add artificial offset
        prep_signals.correct_offset()
        # correct_offset modifies self.signals in-place
        means = np.mean(prep_signals.signals, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)


class TestUndo:
    """Bounded (at most 5 steps) undo history for signal-mutating actions
    (filtering, decimation, offset correction, noise addition, channel
    deletion, ...) - all of which already call save_undo_snapshot()."""

    def test_undo_available_is_false_initially(self, prep_signals):
        assert prep_signals.undo_available is False

    def test_undo_raises_runtime_error_when_nothing_to_undo(self, prep_signals):
        with pytest.raises(RuntimeError):
            prep_signals.undo()

    def test_save_undo_snapshot_makes_undo_available(self, prep_signals):
        prep_signals.save_undo_snapshot()
        assert prep_signals.undo_available is True

    def test_undo_restores_signals_after_correct_offset(self, prep_signals):
        original = prep_signals.signals.copy()
        prep_signals.signals[:, 0] += 5.0
        before_correct = prep_signals.signals.copy()
        prep_signals.correct_offset()
        assert not np.allclose(prep_signals.signals, before_correct)

        assert prep_signals.undo_available is True
        prep_signals.undo()
        np.testing.assert_array_equal(prep_signals.signals, before_correct)
        assert prep_signals.undo_available is False

    def test_undo_restores_sampling_rate_after_decimate(self, prep_signals):
        original_rate = prep_signals.sampling_rate
        prep_signals.decimate_signals(2)
        assert prep_signals.sampling_rate == original_rate / 2

        prep_signals.undo()
        assert prep_signals.sampling_rate == original_rate

    def test_mutating_methods_all_wire_into_undo(self, prep_signals):
        prep_signals.correct_offset()
        prep_signals.add_noise(amplitude=0.01)
        prep_signals.filter_signals(lowpass=10.0)
        prep_signals.decimate_signals(2)
        prep_signals.delete_channels(0)
        assert prep_signals.undo_available is True

        # 5 mutating actions above -> 5 successful undos, then empty again.
        for _ in range(5):
            prep_signals.undo()
        assert prep_signals.undo_available is False
        with pytest.raises(RuntimeError):
            prep_signals.undo()

    def test_undo_history_is_capped_at_5_steps(self, prep_signals):
        for _ in range(7):
            prep_signals.add_noise(amplitude=0.01)

        undone = 0
        while prep_signals.undo_available:
            prep_signals.undo()
            undone += 1
        assert undone == 5


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


class TestRenameChannel:
    def test_rename_updates_channel_headers(self, prep_signals):
        prep_signals.rename_channel(0, 'new_name')
        assert prep_signals.channel_headers[0] == 'new_name'

    def test_rename_rejects_duplicate_name(self, prep_signals):
        other_name = prep_signals.channel_headers[1]
        with pytest.raises(ValueError):
            prep_signals.rename_channel(0, other_name)

    def test_rename_to_same_name_is_allowed(self, prep_signals):
        prep_signals.rename_channel(0, 'chan0')
        prep_signals.rename_channel(0, 'chan0')
        assert prep_signals.channel_headers[0] == 'chan0'

    def test_rename_updates_existing_chan_dof_annotation(self, prep_signals):
        prep_signals.set_chan_dof(0, 'N1', 90.0, 0.0)
        prep_signals.rename_channel(0, 'new_name')
        assert prep_signals.chan_dofs[0][4] == 'new_name'

    def test_rename_invalid_channel_raises(self, prep_signals):
        with pytest.raises(ValueError):
            prep_signals.rename_channel(prep_signals.num_analised_channels, 'x')


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
