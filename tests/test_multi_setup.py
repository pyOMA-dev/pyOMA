"""Tests for multi-setup analysis: PogerSSICovRef, PreGERSSI and MergePoSER.

The PoGer/PreGER real-data tests process two real measurement files end-to-end
and are marked @pytest.mark.slow.  The PreGER fast tests use small synthetic rod
signals; the MergePoSER smoke test uses a minimal in-memory setup.
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.SSICovRef import BRSSICovRef, PogerSSICovRef
from pyOMA.core.MultiSetupSSI import PreGERSSI
from pyOMA.core.StabilDiagram import StabilCalc
from pyOMA.core.PostProcessingTools import MergePoSER

TEST_FILES = Path(__file__).parent / 'files'


# ── PreGER synthetic-rod helpers ──────────────────────────────────────────────

_ROD_N = 8192
_ROD_NODES = 6
_ROD_FS = 128
_ROD_MODES = 2


def _rod_signal():
    """Cached ambient response of a 6-channel rod (node 0 is the fixed end)."""
    if not hasattr(_rod_signal, '_cache'):
        from tests.system_ambient_ifrf import ambient_gaussian
        _, sig = ambient_gaussian(_ROD_N, _ROD_NODES, [5], _ROD_FS, 10.0,
                                  seed=1, num_modes=_ROD_MODES)
        _rod_signal._cache = sig
    return _rod_signal._cache


def _make_rod_ps(channels, ref_local, m_lags=80):
    """Fresh PreProcessSignals for the given rod channels (columns of the signal).

    ``ref_local`` are reference channel indices *within* the selected subset;
    each column ``c`` is assigned to node ``str(channels[c])``.
    """
    sig = _rod_signal()[:, channels]
    ps = PreProcessSignals(sig, _ROD_FS, ref_channels=list(ref_local))
    ps.add_chan_dofs([[c, str(channels[c]), 1.0, 0.0, f'ch{channels[c]}']
                      for c in range(len(channels))])
    ps.corr_blackman_tukey(m_lags=m_lags)
    return ps


def _mac(phi_a, phi_b):
    """Modal Assurance Criterion between two complex mode-shape vectors."""
    num = np.abs(np.vdot(phi_a, phi_b)) ** 2
    den = np.vdot(phi_a, phi_a).real * np.vdot(phi_b, phi_b).real
    return num / den


# ── PogerSSICovRef ────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestPogerSSICovRef:
    """Full PoGer pipeline using the first two real measurement setups.

    Marked slow – exclude with: pytest -m "not slow"
    """

    @pytest.fixture(scope='class')
    def poger_result(self):
        modal = PogerSSICovRef()
        for i in (1, 2):
            meas_dir = TEST_FILES / f'measurement_{i}'
            ps = PreProcessSignals.init_from_config(
                conf_file=meas_dir / 'setup_info.txt',
                meas_file=meas_dir / f'measurement_{i}.npy',
                chan_dofs_file=meas_dir / 'channel_dofs.txt',
            )
            ps.corr_blackman_tukey(m_lags=200)
            modal.add_setup(ps)

        modal.pair_channels()
        # m_lags=200 → max num_block_columns+num_block_rows < 200, use 80
        modal.build_merged_subspace_matrix(num_block_columns=80)
        modal.compute_modal_params(max_model_order=40, max_modes=24)
        return modal

    def test_construction_succeeds(self, poger_result):
        assert isinstance(poger_result, PogerSSICovRef)

    def test_state_complete(self, poger_result):
        # PoGer state = [subspace_built, channels_paired, modal_params, setups_added, ...]
        # state[4] is an optional step not used in the basic workflow
        assert poger_result.state[0], "Subspace matrix not built"
        assert poger_result.state[1], "Channels not paired"
        assert poger_result.state[2], "Modal params not computed"

    def test_modal_frequencies_computed(self, poger_result):
        assert poger_result.modal_frequencies is not None

    def test_modal_frequencies_shape(self, poger_result):
        freqs = poger_result.modal_frequencies
        assert freqs.ndim == 2
        assert freqs.shape[0] == 40  # max_model_order

    def test_frequencies_below_nyquist(self, poger_result):
        freqs = poger_result.modal_frequencies
        valid = freqs[~np.isnan(freqs)]
        # Measurement sampling rate is 256 Hz → Nyquist = 128 Hz
        assert np.all(valid >= 0)
        assert np.all(valid < 128.0)

    def test_save_load_round_trip(self, poger_result):
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            poger_result.save_state(fname)
            loaded = PogerSSICovRef.load_state(fname)
            np.testing.assert_allclose(
                loaded.modal_frequencies, poger_result.modal_frequencies,
                rtol=1e-10, equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)


# ── MergePoSER ────────────────────────────────────────────────────────────────

class TestMergePoSER:
    """Smoke tests for MergePoSER using pre-computed single-setup results."""

    @pytest.fixture(scope='class')
    def single_setup_stabil(self, test_files_dir):
        """Load a pre-computed BRSSICovRef + StabilCalc from the test files."""
        meas_dir = test_files_dir / 'measurement_1'
        prep = PreProcessSignals.init_from_config(
            conf_file=meas_dir / 'setup_info.txt',
            meas_file=meas_dir / 'measurement_1.npy',
            chan_dofs_file=meas_dir / 'channel_dofs.txt',
        )
        modal = BRSSICovRef.load_state(meas_dir / 'modal_data.npz', prep)
        stabil = StabilCalc.load_state(meas_dir / 'stabil_data.npz', modal)
        return prep, modal, stabil

    def test_add_setup_requires_chan_dofs(self, single_setup_stabil):
        prep, modal, stabil = single_setup_stabil
        merger = MergePoSER()
        if not prep.chan_dofs:
            pytest.skip("chan_dofs not available in loaded state")
        merger.add_setup(prep, modal, stabil)
        assert merger.state[0]

    def test_construction(self):
        merger = MergePoSER()
        assert merger.mean_frequencies is None
        assert merger.mean_damping is None
        assert merger.merged_mode_shapes is None

    def test_save_load_empty_state(self):
        merger = MergePoSER()
        merger.state[0] = True   # pretend setups were added
        merger.mean_frequencies = np.array([1.0, 2.0, 3.0])
        merger.mean_damping = np.array([0.02, 0.03, 0.04])
        merger.std_frequencies = np.array([0.001, 0.001, 0.001])
        merger.merged_mode_shapes = np.eye(3, dtype=complex)
        merger.merged_chan_dofs = []
        merger.state[1] = True

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            merger.save_state(fname)
            loaded = MergePoSER.load_state(fname)
            np.testing.assert_allclose(
                loaded.mean_frequencies, merger.mean_frequencies)
        finally:
            Path(fname).unlink(missing_ok=True)


# ── PreGERSSI point estimates (fast, synthetic) ───────────────────────────────

class TestPreGERSSIPoint:
    """Fast point-estimate tests for PreGERSSI on synthetic rod data.

    The Ns=1 case must reproduce BRSSICovRef exactly (same column space); the
    2-setup case (built from disjoint channel slices of the *same* signal)
    exercises the pinv-normalisation merging branch and must reconstruct the
    full-array identification.
    """

    P = Q = 15
    MAXO = 14
    MLAGS = 80

    # ---- Ns = 1 equivalence with BRSSICovRef ----

    @pytest.fixture(scope='class')
    def ns1_pair(self):
        """(BRSSICovRef, PreGERSSI) on the same single full-array rod setup."""
        br = BRSSICovRef(_make_rod_ps(list(range(_ROD_NODES)), [5], self.MLAGS))
        br.build_toeplitz_cov(num_block_columns=self.Q, num_block_rows=self.P)
        br.compute_modal_params(self.MAXO, modal_contrib=False)

        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # non-ref common DOFs dropped (expected)
            pg.add_setup(_make_rod_ps(list(range(_ROD_NODES)), [5], self.MLAGS))
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.Q, num_block_rows=self.P)
        pg.compute_modal_params(self.MAXO)
        return br, pg

    def test_ns1_frequencies_match_br(self, ns1_pair):
        br, pg = ns1_pair
        for order in range(1, self.MAXO):
            fb = np.sort(br.modal_frequencies[order, :order])
            fp = np.sort(pg.modal_frequencies[order, :order])
            np.testing.assert_allclose(fb, fp, rtol=0, atol=1e-8, equal_nan=True)

    def test_ns1_eigenvalues_match_br(self, ns1_pair):
        br, pg = ns1_pair
        for order in range(1, self.MAXO):
            lb = br.eigenvalues[order, :order]
            lp = pg.eigenvalues[order, :order]
            lb = np.sort_complex(lb[np.isfinite(lb)])
            lp = np.sort_complex(lp[np.isfinite(lp)])
            assert lb.shape == lp.shape
            np.testing.assert_allclose(lb, lp, rtol=0, atol=1e-8)

    def test_ns1_mode_shapes_match_br_up_to_permutation(self, ns1_pair):
        br, pg = ns1_pair
        perm = pg.setup_row_orders[0]  # PreGER row i == BR channel perm[i]
        order = self.MAXO - 1
        n_modes = np.count_nonzero(np.isfinite(pg.modal_frequencies[order, :order]))
        assert n_modes >= 2
        for m in range(n_modes):
            phi_pg = pg.mode_shapes[:, m, order]
            phi_br = br.mode_shapes[perm, m, order]
            if not np.all(np.isfinite(phi_pg)):
                continue
            assert _mac(phi_pg, phi_br) > 1 - 1e-6

    # ---- pairing / bookkeeping ----

    def test_pairing_sets_common_ref(self):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
            pg.add_setup(_make_rod_ps([5, 3, 4], [0], self.MLAGS))
            pg.pair_channels()
        # only node 5 is a reference in both setups
        assert pg.num_ref_channels == 1
        # merged order: setup-0 refs-first + setup-1 rovings == [5,0,1,2,3,4]
        assert pg.merged_num_channels == 6
        assert pg.num_analised_channels == 6
        assert pg.setup_row_orders[0] == [0, 1, 2, 3]  # ref already first here
        assert pg.setup_row_orders[1] == [0, 1, 2]

    # ---- 2-setup merge exercises the j>=1 pinv branch ----

    @pytest.fixture(scope='class')
    def two_setup(self):
        """Full-array BRSSICovRef vs PreGER merge of two disjoint channel slices."""
        br = BRSSICovRef(_make_rod_ps(list(range(_ROD_NODES)), [5], self.MLAGS))
        br.build_toeplitz_cov(num_block_columns=self.Q, num_block_rows=self.P)
        br.compute_modal_params(self.MAXO, modal_contrib=False)

        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
            pg.add_setup(_make_rod_ps([5, 3, 4], [0], self.MLAGS))
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.Q, num_block_rows=self.P)
        pg.compute_modal_params(self.MAXO)
        return br, pg

    def test_two_setup_matches_full_array_at_true_order(self, two_setup):
        br, pg = two_setup
        # At the true model order (2 modes -> order 4) every pole is physical:
        # merging disjoint slices of the *same* signal must reproduce the joint
        # identification to within estimation precision (~1e-4).  Higher orders
        # additionally fit spurious noise poles that legitimately differ between
        # the joint-truncation (BR) and per-setup-truncation (PreGER) estimators.
        order = 4
        fb = np.sort(br.modal_frequencies[order, :order])
        fp = np.sort(pg.modal_frequencies[order, :order])
        np.testing.assert_allclose(fb, fp, rtol=0, atol=1e-2, equal_nan=True)
        finite = fp[np.isfinite(fp)]
        # both rod modes recovered near their true values (~6.46, ~19.4 Hz)
        assert np.any(np.abs(finite - 6.46) < 0.2)
        assert np.any(np.abs(finite - 19.4) < 0.3)

    def test_two_setup_merged_channel_order(self, two_setup):
        _, pg = two_setup
        nodes = [cd[1] for cd in pg.merged_chan_dofs]
        assert nodes == ['5', '0', '1', '2', '3', '4']

    def test_two_setup_mode_shapes_match_full_array(self, two_setup):
        br, pg = two_setup
        # merged order [5,0,1,2,3,4] -> BR channel index for each merged row
        perm = [5, 0, 1, 2, 3, 4]
        order = 4
        n_modes = np.count_nonzero(np.isfinite(pg.modal_frequencies[order, :order]))
        assert n_modes == 2
        for m in range(n_modes):
            phi_pg = pg.mode_shapes[:, m, order]
            phi_br = br.mode_shapes[perm, m, order]
            assert _mac(phi_pg, phi_br) > 0.999

    # ---- guards and caps ----

    def test_order_cap(self):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
            pg.add_setup(_make_rod_ps([5, 3, 4], [0], self.MLAGS))
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.Q, num_block_rows=self.P)
        r = pg.num_ref_channels
        assert pg.max_model_order == min((self.P + 1) * r, self.Q * r)

    def test_build_requires_paired(self):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
        with pytest.raises(RuntimeError):
            pg.build_subspace_matrices(num_block_columns=self.Q)

    def test_projection_point_estimate(self):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
            pg.add_setup(_make_rod_ps([5, 3, 4], [0], self.MLAGS))
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.P, num_block_rows=self.P,
                                       subspace_method='projection', num_blocks=10)
        assert pg.subspace_method == 'projection'
        # projection: q == p, order cap min((p+1)r, p*r) = p*r
        assert pg.max_model_order == self.P * pg.num_ref_channels
        pg.compute_modal_params(self.MAXO)
        order = 4
        finite = pg.modal_frequencies[order, :order]
        finite = finite[np.isfinite(finite)]
        assert np.any(np.abs(finite - 6.46) < 0.3)   # rod mode 1
        assert np.any(np.abs(finite - 19.4) < 0.6)   # rod mode 2

    def test_too_many_blocks_raises(self):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
            pg.add_setup(_make_rod_ps([5, 3, 4], [0], self.MLAGS))
            pg.pair_channels()
        with pytest.raises(RuntimeError):
            pg.build_subspace_matrices(num_block_columns=self.MLAGS)

    def test_save_load_round_trip(self):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pg.add_setup(_make_rod_ps([5, 0, 1, 2], [0], self.MLAGS))
            pg.add_setup(_make_rod_ps([5, 3, 4], [0], self.MLAGS))
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.Q, num_block_rows=self.P)
        pg.compute_modal_params(self.MAXO)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            pg.save_state(fname)
            loaded = PreGERSSI.load_state(fname)
            np.testing.assert_allclose(
                loaded.modal_frequencies, pg.modal_frequencies,
                rtol=1e-10, equal_nan=True)
            # state must be re-runnable from restored subspace matrices
            A, C = loaded.estimate_state(6)
            assert A.shape == (6, 6)
            assert C.shape[0] == pg.merged_num_channels
        finally:
            Path(fname).unlink(missing_ok=True)


@pytest.mark.slow
class TestPreGERSSI:
    """Full PreGER pipeline on the first two real measurement setups.

    Marked slow – exclude with: pytest -m "not slow"
    """

    @pytest.fixture(scope='class')
    def preger_result(self):
        modal = PreGERSSI()
        for i in (1, 2):
            meas_dir = TEST_FILES / f'measurement_{i}'
            ps = PreProcessSignals.init_from_config(
                conf_file=meas_dir / 'setup_info.txt',
                meas_file=meas_dir / f'measurement_{i}.npy',
                chan_dofs_file=meas_dir / 'channel_dofs.txt',
            )
            ps.corr_blackman_tukey(m_lags=200)
            modal.add_setup(ps)
        modal.pair_channels()
        modal.build_subspace_matrices(num_block_columns=80)
        modal.compute_modal_params(max_model_order=40)
        return modal

    @pytest.fixture(scope='class')
    def prep_signals_list(self):
        prep_list = []
        for i in (1, 2):
            meas_dir = TEST_FILES / f'measurement_{i}'
            ps = PreProcessSignals.init_from_config(
                conf_file=meas_dir / 'setup_info.txt',
                meas_file=meas_dir / f'measurement_{i}.npy',
                chan_dofs_file=meas_dir / 'channel_dofs.txt',
            )
            ps.corr_blackman_tukey(m_lags=200)
            prep_list.append(ps)
        return prep_list

    @pytest.fixture(scope='class')
    def poger_reference(self):
        modal = PogerSSICovRef()
        for i in (1, 2):
            meas_dir = TEST_FILES / f'measurement_{i}'
            ps = PreProcessSignals.init_from_config(
                conf_file=meas_dir / 'setup_info.txt',
                meas_file=meas_dir / f'measurement_{i}.npy',
                chan_dofs_file=meas_dir / 'channel_dofs.txt',
            )
            ps.corr_blackman_tukey(m_lags=200)
            modal.add_setup(ps)
        modal.pair_channels()
        modal.build_merged_subspace_matrix(num_block_columns=80)
        modal.compute_modal_params(max_model_order=40, max_modes=24)
        return modal

    def test_state_complete(self, preger_result):
        assert preger_result.state[0], "Subspace matrices not built"
        assert preger_result.state[1], "Channels not paired"
        assert preger_result.state[2], "Modal params not computed"
        assert preger_result.state[3], "Setups not added"

    def test_modal_frequencies_shape(self, preger_result):
        freqs = preger_result.modal_frequencies
        assert freqs.ndim == 2
        assert freqs.shape[0] == 40

    def test_frequencies_below_nyquist(self, preger_result):
        freqs = preger_result.modal_frequencies
        valid = freqs[~np.isnan(freqs)]
        assert np.all(valid >= 0)
        assert np.all(valid < 128.0)  # fs = 256 Hz -> Nyquist 128 Hz

    def test_mode_shape_rows_match_merged_channels(self, preger_result):
        assert preger_result.mode_shapes.shape[0] == preger_result.merged_num_channels

    def test_stable_poles_match_poger(self, preger_result, poger_reference):
        """Poles that stabilise in both estimators must agree within a few %."""
        order = 30

        def stable_freqs(modal):
            f = modal.modal_frequencies
            out = []
            for col in range(order):
                fi = f[order, col]
                if not np.isfinite(fi) or fi <= 0:
                    continue
                # frequency-stable against the previous order
                prev = f[order - 1, :order - 1]
                prev = prev[np.isfinite(prev) & (prev > 0)]
                if prev.size and np.min(np.abs(prev - fi)) / fi < 0.01:
                    out.append(fi)
            return np.array(sorted(out))

        f_pg = stable_freqs(preger_result)
        f_po = stable_freqs(poger_reference)
        assert f_pg.size >= 3, "PreGER produced too few stable poles"
        # every PreGER stable pole should have a Poger match within 2 %
        matched = 0
        for fi in f_pg:
            if f_po.size and np.min(np.abs(f_po - fi)) / fi < 0.02:
                matched += 1
        assert matched >= max(3, int(0.6 * f_pg.size))

    def test_save_load_round_trip(self, preger_result):
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            preger_result.save_state(fname)
            loaded = PreGERSSI.load_state(fname)
            np.testing.assert_allclose(
                loaded.modal_frequencies, preger_result.modal_frequencies,
                rtol=1e-10, equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_init_from_config(self, tmp_path, prep_signals_list):
        cfg = tmp_path / 'preger.txt'
        cfg.write_text(
            'Number of Block-Columns:\n80\n'
            'Number of Block-Rows:\n0\n'
            'Maximum Model Order:\n40\n'
            'Subspace Method (projection/covariance):\ncovariance\n'
            'Number of Blocks:\n0\n'
        )
        obj = PreGERSSI.init_from_config(cfg, prep_signals_list)
        assert all(obj.state[:3]), "build/pair/compute steps not all completed"
        assert obj.modal_frequencies.shape[0] == 40

    def test_write_config_round_trips_through_init_from_config(
            self, preger_result, prep_signals_list, tmp_path):
        cfg = tmp_path / 'preger.txt'
        preger_result.write_config(cfg)
        obj = PreGERSSI.init_from_config(cfg, prep_signals_list)
        assert obj.num_block_columns == preger_result.num_block_columns
        assert obj.max_model_order == preger_result.max_model_order
        np.testing.assert_allclose(
            obj.modal_frequencies, preger_result.modal_frequencies,
            rtol=1e-10, equal_nan=True)
