"""Tests for VarPreGERSSI: first-order modal-parameter uncertainties.

Fast tests finite-difference-validate the two perturbation-propagation branches
(setup-0 ripple and setup->=1 direct pinv branch), check point-estimate
invariance vs PreGERSSI, StabilCalc std pickup and save/load.  The slow
Monte-Carlo tests validate the predicted standard deviations against ensemble
scatter, with a guard proving the gate is not vacuous.
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.MultiSetupSSI import PreGERSSI, VarPreGERSSI
from pyOMA.core.StabilDiagram import StabilCalc

from tests.system_ambient_ifrf import ambient_gaussian, ambient_ifrf

_FS = 128
_NODES = 6


def _rod_signal(seed, n=8192, generator=ambient_gaussian):
    _, sig = generator(n, _NODES, [5], _FS, 10.0, seed=seed, num_modes=2)
    return sig


def _mkps(sig, channels, ref_local, m_lags, n_segments):
    ps = PreProcessSignals(sig[:, channels], _FS, ref_channels=list(ref_local))
    ps.add_chan_dofs([[c, str(channels[c]), 1.0, 0.0, f'ch{channels[c]}']
                      for c in range(len(channels))])
    ps.corr_blackman_tukey(m_lags=m_lags, n_segments=n_segments)
    return ps


def _build_var(signals, chan_specs, m_lags, n_segments, p, q,
               subspace_method='covariance', num_blocks=None):
    """signals: list of arrays (one per setup); chan_specs: [(channels, ref_local), ...]."""
    v = VarPreGERSSI()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for sig, (ch, ref) in zip(signals, chan_specs):
            v.add_setup(_mkps(sig, ch, ref, m_lags, n_segments))
        v.pair_channels()
        v.build_subspace_matrices(num_block_columns=q, num_block_rows=p,
                                  subspace_method=subspace_method, num_blocks=num_blocks)
    return v


def _freq_damp(lam):
    a = np.abs(np.arctan2(lam.imag, lam.real))
    b = np.log(np.abs(lam))
    return (np.sqrt(a ** 2 + b ** 2) * _FS / 2 / np.pi,
            100 * np.abs(b) / np.sqrt(a ** 2 + b ** 2))


def _point_shape(v, C, phi, freq):
    """Integrated + rotation-only mode shape, via the class's own normaliser."""
    return v._normalize_mode_shape(C @ phi, freq)


def _predict_columns(v, order):
    """Return (dA over all K columns, C, dC, A, w, vl, vr, phys) at *order*."""
    factors = v._prepare_perturbation_factors(order)
    ctx = v._var_context(order)
    A, C = ctx['A'], ctx['C']
    dO_up, dO_down, dC = v._observability_perturbations(order, ctx, factors)
    Kinv = np.linalg.inv(ctx['O_up'].T @ ctx['O_up'])
    resid = ctx['O_down'] - ctx['O_up'] @ A
    dA = np.einsum('ij,jmK->imK', Kinv,
                   np.einsum('rnK,rm->nmK', dO_up, resid)
                   + np.einsum('rn,rmK->nmK', ctx['O_up'],
                               dO_down - np.einsum('rnK,nm->rmK', dO_up, A)))
    w, vl, vr = scipy.linalg.eig(A, left=True, right=True)
    phys = v.remove_conjugates(w, vr, vl, inds_only=True)
    return dA, C, dC, A, w, vl, vr, phys


def _predict_col(v, order, dA, C, dC, A, w, vl, vr, col, i):
    lam = w[i]
    phi = vr[:, i]
    chi = vl[:, i]
    dAk = dA[:, :, col]
    dlam = (chi.conj() @ dAk @ phi) / (chi.conj() @ phi)
    dfd = v._jacobian_freq_damp(lam, _FS) @ np.array([dlam.real, dlam.imag])
    proj = np.eye(order) - np.outer(phi, chi.conj()) / (chi.conj() @ phi)
    dphi = np.linalg.pinv(lam * np.eye(order) - A) @ (proj @ (dAk @ phi))
    dpsi_raw = C @ dphi + dC[:, :, col] @ phi
    freq = _freq_damp(lam)[0]
    dpsi = v._mode_shape_perturbation(
        C @ phi, dpsi_raw[:, None], freq, np.array([dfd[0]]))[:, 0]
    return dfd[0], dfd[1], dpsi


# ── finite-difference validation of the perturbation chain ────────────────────

class TestVarPreGERSSIFiniteDifference:
    """Per-column FD validation of both propagation branches (definitive)."""

    MLAGS, NSEG, P, Q, ORDER = 40, 6, 10, 10, 4
    CHS = [([5, 0, 1, 2], [0]), ([5, 3, 4], [0])]

    def _fd_point(self, signals, s_pert, k, eps):
        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for si, (sig, (ch, ref)) in enumerate(zip(signals, self.CHS)):
                ps = _mkps(sig, ch, ref, self.MLAGS, self.NSEG)
                if si == s_pert:
                    dirn = ps.corr_matrices[k] - ps.corr_matrix
                    ps.corr_matrix_bt = ps.corr_matrix + eps * dirn
                pg.add_setup(ps)
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.Q, num_block_rows=self.P)
        Ap, Cp = pg.estimate_state(self.ORDER)
        wp, vrp = np.linalg.eig(Ap)
        return wp, vrp, Cp

    @pytest.mark.parametrize('s_pert', [0, 1])
    def test_per_column_fd(self, s_pert):
        signals = [_rod_signal(7), _rod_signal(7)]  # same signal sliced -> both setups
        v = _build_var(signals, self.CHS, self.MLAGS, self.NSEG, self.P, self.Q)
        dA, C, dC, A, w, vl, vr, phys = _predict_columns(v, self.ORDER)
        col = int(np.sum(v.setup_n_b[:s_pert]))  # block 0 of setup s_pert
        scale = np.sqrt(v.setup_n_b[s_pert] * (v.setup_n_b[s_pert] - 1))
        eps = 1e-6
        wp, vrp, Cp = self._fd_point(signals, s_pert, 0, eps)

        assert len(phys) >= 2
        for i in phys:
            lam = w[i]
            f0, x0 = _freq_damp(lam)
            df_pr, dxi_pr, dpsi_pr = _predict_col(v, self.ORDER, dA, C, dC, A, w, vl, vr, col, i)
            jj = np.argmin(np.abs(wp - lam))
            f1, x1 = _freq_damp(wp[jj])
            df_fd = (f1 - f0) / eps / scale
            dxi_fd = (x1 - x0) / eps / scale
            # FD of the *integrated* (displacement) shape, at each mode's own freq
            dpsi_fd = (_point_shape(v, Cp, vrp[:, jj], f1)
                       - _point_shape(v, C, vr[:, i], f0)) / eps / scale
            assert abs(df_pr - df_fd) <= 1e-3 * abs(df_fd) + 1e-6
            assert abs(dxi_pr - dxi_fd) <= 1e-3 * abs(dxi_fd) + 1e-6
            assert (np.linalg.norm(dpsi_pr - dpsi_fd)
                    <= 1e-3 * np.linalg.norm(dpsi_fd) + 1e-6)


# ── projection (data-driven) variance FD validation ───────────────────────────

class TestVarPreGERSSIProjectionFiniteDifference:
    """Per-column FD validation of the variance chain in projection mode.

    The perturbation direction is injected directly into a setup's data-driven
    subspace matrix ``H^dat`` (re-deriving the point estimate through the class's
    own :meth:`estimate_state`), so the same subspace-agnostic chain is exercised
    with the projection source.
    """

    P, NB, ORDER = 8, 8, 4
    CHS = [([5, 0, 1, 2], [0]), ([5, 3, 4], [0])]

    def _build(self):
        v = VarPreGERSSI()
        signals = [_rod_signal(7), _rod_signal(7)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for sig, (ch, ref) in zip(signals, self.CHS):
                ps = PreProcessSignals(sig[:, ch], _FS, ref_channels=list(ref))
                ps.add_chan_dofs([[c, str(ch[c]), 1.0, 0.0, f'ch{ch[c]}']
                                  for c in range(len(ch))])
                v.add_setup(ps)   # projection: no correlations needed
            v.pair_channels()
            v.build_subspace_matrices(num_block_columns=self.P, num_block_rows=self.P,
                                      subspace_method='projection', num_blocks=self.NB)
        return v

    def _estimate_perturbed(self, v, s, h_pert, order):
        """Re-derive (A, C) after injecting a perturbed H^dat for setup s."""
        p, r = v.num_block_rows, v.num_ref_channels
        n_l = v.setup_n_l[s]
        saved = (v.svd_h1, v.svd_refs[s], v.hankels_mov[s])
        try:
            if s == 0:
                v.svd_h1 = scipy.linalg.svd(h_pert, full_matrices=False)
            else:
                ref_idx = v._block_row_selection(p + 1, n_l, 0, r)
                mov_idx = v._block_row_selection(p + 1, n_l, r, n_l)
                v.svd_refs[s] = scipy.linalg.svd(h_pert[ref_idx], full_matrices=False)
                v.hankels_mov[s] = h_pert[mov_idx]
            return v.estimate_state(order)
        finally:
            v.svd_h1, v.svd_refs[s], v.hankels_mov[s] = saved

    @pytest.mark.parametrize('s_pert', [0, 1])
    def test_per_column_fd_projection(self, s_pert):
        v = self._build()
        dA, C, dC, A, w, vl, vr, phys = _predict_columns(v, self.ORDER)
        assert len(phys) >= 2
        col = int(np.sum(v.setup_n_b[:s_pert]))     # block 0 of setup s_pert
        scale = np.sqrt(v.setup_n_b[s_pert] * (v.setup_n_b[s_pert] - 1))

        h_mean, h_blocks = v._projection_subspace(s_pert, self.P, self.NB)
        direction = h_blocks[0] - h_mean
        eps = 1e-6
        Ap, Cp = self._estimate_perturbed(v, s_pert, h_mean + eps * direction, self.ORDER)
        wp, vrp = np.linalg.eig(Ap)

        for i in phys:
            lam = w[i]
            f0, x0 = _freq_damp(lam)
            df_pr, dxi_pr, dpsi_pr = _predict_col(v, self.ORDER, dA, C, dC, A, w, vl, vr, col, i)
            jj = np.argmin(np.abs(wp - lam))
            f1, x1 = _freq_damp(wp[jj])
            df_fd = (f1 - f0) / eps / scale
            dxi_fd = (x1 - x0) / eps / scale
            dpsi_fd = (_point_shape(v, Cp, vrp[:, jj], f1)
                       - _point_shape(v, C, vr[:, i], f0)) / eps / scale
            assert abs(df_pr - df_fd) <= 1e-3 * abs(df_fd) + 1e-6
            assert abs(dxi_pr - dxi_fd) <= 1e-3 * abs(dxi_fd) + 1e-6
            assert (np.linalg.norm(dpsi_pr - dpsi_fd)
                    <= 1e-3 * np.linalg.norm(dpsi_fd) + 1e-6)

    def test_projection_point_estimate_invariance(self):
        v = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v.compute_modal_params(self.ORDER + 2)
            pg = PreGERSSI()
            signals = [_rod_signal(7), _rod_signal(7)]
            for sig, (ch, ref) in zip(signals, self.CHS):
                ps = PreProcessSignals(sig[:, ch], _FS, ref_channels=list(ref))
                ps.add_chan_dofs([[c, str(ch[c]), 1.0, 0.0, f'ch{ch[c]}']
                                  for c in range(len(ch))])
                pg.add_setup(ps)
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.P, num_block_rows=self.P,
                                       subspace_method='projection', num_blocks=self.NB)
            pg.compute_modal_params(self.ORDER + 2)
        np.testing.assert_allclose(v.modal_frequencies, pg.modal_frequencies,
                                   rtol=0, atol=1e-9, equal_nan=True)


# ── basics: invariance, positivity, StabilCalc, save/load ─────────────────────

class TestVarPreGERSSIBasics:

    MLAGS, NSEG, P, Q, MAXO = 60, 8, 15, 15, 10
    CHS = [([5, 0, 1, 2], [0]), ([5, 3, 4], [0])]

    @pytest.fixture(scope='class')
    def var_and_point(self):
        signals = [_rod_signal(11), _rod_signal(11)]
        v = _build_var(signals, self.CHS, self.MLAGS, self.NSEG, self.P, self.Q)
        v.compute_modal_params(self.MAXO)

        pg = PreGERSSI()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for sig, (ch, ref) in zip(signals, self.CHS):
                pg.add_setup(_mkps(sig, ch, ref, self.MLAGS, self.NSEG))
            pg.pair_channels()
            pg.build_subspace_matrices(num_block_columns=self.Q, num_block_rows=self.P)
        pg.compute_modal_params(self.MAXO)
        return v, pg

    def test_point_estimate_invariance(self, var_and_point):
        v, pg = var_and_point
        np.testing.assert_allclose(v.modal_frequencies, pg.modal_frequencies,
                                   rtol=0, atol=1e-9, equal_nan=True)
        np.testing.assert_allclose(v.modal_damping, pg.modal_damping,
                                   rtol=0, atol=1e-9, equal_nan=True)

    def test_state_and_std_shapes(self, var_and_point):
        v, _ = var_and_point
        assert v.state[4]
        assert v.std_frequencies.shape == (self.MAXO, self.MAXO)
        assert v.std_mode_shapes.shape[0] == v.merged_num_channels

    def test_std_positive_finite_for_physical_modes(self, var_and_point):
        v, _ = var_and_point
        order = 4
        n = np.count_nonzero(np.isfinite(v.modal_frequencies[order, :order])
                             & (v.modal_frequencies[order, :order] > 0))
        assert n >= 2
        sf = v.std_frequencies[order, :n]
        sd = v.std_damping[order, :n]
        assert np.all(np.isfinite(sf)) and np.all(sf > 0)
        assert np.all(np.isfinite(sd)) and np.all(sd > 0)

    def test_stabilcalc_std_capability(self, var_and_point):
        v, _ = var_and_point
        sc = StabilCalc(v)
        assert sc.capabilities['std'] is True
        assert v.num_blocks == min(v.setup_n_b)

    def test_save_load_std(self, var_and_point):
        v, _ = var_and_point
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            v.save_state(fname)
            loaded = VarPreGERSSI.load_state(fname)
            assert loaded.state[4]
            np.testing.assert_allclose(loaded.std_frequencies, v.std_frequencies,
                                       rtol=1e-10, equal_nan=True)
            np.testing.assert_allclose(loaded.std_mode_shapes, v.std_mode_shapes,
                                       rtol=1e-10, equal_nan=True)
            assert loaded.num_blocks == v.num_blocks
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_init_from_config(self, tmp_path):
        """VarPreGERSSI inherits PreGERSSI.init_from_config unmodified; cls()
        must resolve polymorphically so the returned object is a VarPreGERSSI
        with variances computed, not a plain PreGERSSI."""
        signals = [_rod_signal(11), _rod_signal(11)]
        prep_signals_list = [
            _mkps(sig, ch, ref, self.MLAGS, self.NSEG)
            for sig, (ch, ref) in zip(signals, self.CHS)
        ]
        cfg = tmp_path / 'preger.txt'
        cfg.write_text(
            f'Number of Block-Columns:\n{self.Q}\n'
            f'Number of Block-Rows:\n{self.P}\n'
            f'Maximum Model Order:\n{self.MAXO}\n'
            'Subspace Method (projection/covariance):\ncovariance\n'
            'Number of Blocks:\n0\n'
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            obj = VarPreGERSSI.init_from_config(cfg, prep_signals_list)
        assert isinstance(obj, VarPreGERSSI)
        assert all(obj.state)
        assert obj.std_frequencies is not None
        assert obj.std_frequencies.shape == (self.MAXO, self.MAXO)


# ── Monte-Carlo validation (slow) ─────────────────────────────────────────────

@pytest.mark.slow
class TestVarPreGERSSIMonteCarlo:
    """std_predicted vs ensemble scatter over independent realizations.

    Each run draws *independent* ambient_gaussian realizations for the two
    setups (changing excitation across setups is exactly what PreGER absorbs).
    """

    N = 8192
    MLAGS, NSEG, P, Q, ORDER = 100, 20, 8, 8, 4
    CHS = [([5, 1, 2], [0]), ([5, 3, 4], [0])]

    def _run_ensemble(self, n_runs, generator, method='covariance', num_blocks=None):
        freqs, damps, shapes = [], [], []
        pred_f, pred_d, pred_s = [], [], []
        for run in range(n_runs):
            sigA = _rod_signal(2 * run, self.N, generator)
            sigB = _rod_signal(2 * run + 1, self.N, generator)
            v = _build_var([sigA, sigB], self.CHS, self.MLAGS, self.NSEG, self.P, self.Q,
                           subspace_method=method, num_blocks=num_blocks)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v.compute_modal_params(self.ORDER + 1)
            o = self.ORDER
            f = v.modal_frequencies[o, :o]
            valid = np.isfinite(f) & (f > 0)
            # order the two physical modes by frequency
            idx = np.where(valid)[0]
            if idx.size < 2:
                continue
            idx = idx[np.argsort(f[idx])][:2]
            freqs.append(f[idx])
            damps.append(v.modal_damping[o, idx])
            shapes.append(np.abs(v.mode_shapes[:, idx, o]))
            pred_f.append(v.std_frequencies[o, idx])
            pred_d.append(v.std_damping[o, idx])
            pred_s.append(np.abs(v.std_mode_shapes[:, idx, o]))
        return (np.array(freqs), np.array(damps), np.array(shapes),
                np.array(pred_f), np.array(pred_d), np.array(pred_s))

    def test_mc_std_matches_prediction(self):
        freqs, damps, shapes, pred_f, pred_d, pred_s = self._run_ensemble(
            100, ambient_gaussian)
        assert freqs.shape[0] >= 80
        mc_f = np.std(freqs, axis=0, ddof=1)
        mc_d = np.std(damps, axis=0, ddof=1)
        ratio_f = mc_f / np.mean(pred_f, axis=0)
        ratio_d = mc_d / np.mean(pred_d, axis=0)
        # loose band: MC SEM (~0.08 at n=100) + block-correlation slack
        assert np.all(ratio_f > 0.8) and np.all(ratio_f < 1.3), f'freq ratio {ratio_f}'
        assert np.all(ratio_d > 0.7) and np.all(ratio_d < 1.4), f'damp ratio {ratio_d}'
        # mode-shape amplitude scatter vs predicted std, over well-excited comps
        mc_s = np.std(shapes, axis=0, ddof=1)
        pr_s = np.mean(pred_s, axis=0)
        well = pr_s > 0.05 * pr_s.max()
        ratio_s = np.median(mc_s[well] / pr_s[well])
        assert 0.7 < ratio_s < 1.4, f'shape ratio {ratio_s}'

    def test_mc_std_matches_prediction_projection(self):
        # same validation with the data-driven (projection) subspace matrix
        freqs, damps, _, pred_f, pred_d, _ = self._run_ensemble(
            100, ambient_gaussian, method='projection', num_blocks=self.NSEG)
        assert freqs.shape[0] >= 80
        ratio_f = np.std(freqs, axis=0, ddof=1) / np.mean(pred_f, axis=0)
        ratio_d = np.std(damps, axis=0, ddof=1) / np.mean(pred_d, axis=0)
        assert np.all(ratio_f > 0.8) and np.all(ratio_f < 1.3), f'proj freq ratio {ratio_f}'
        assert np.all(ratio_d > 0.7) and np.all(ratio_d < 1.4), f'proj damp ratio {ratio_d}'

    def test_gate_is_not_vacuous(self):
        # ambient_ifrf is a multisine: realization scatter is far below the
        # block-based prediction, so the ratio must fall well outside the band.
        freqs, _, _, pred_f, _, _ = self._run_ensemble(40, ambient_ifrf)
        assert freqs.shape[0] >= 30
        mc_f = np.std(freqs, axis=0, ddof=1)
        pf = np.mean(pred_f, axis=0)
        ratio_f = mc_f / pf
        assert np.any(ratio_f < 0.6), f'multisine ratio should be << 1, got {ratio_f}'
