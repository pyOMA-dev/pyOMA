# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for build-time and post-hoc block weighting of VarPreGERSSI.

The weighting generalizes the per-setup block averaging of the PreGER merger,
exactly as ``weights``/``apply_block_weights`` do for
:class:`~pyOMA.core.VarSSIRef.VarSSIRef` and
:class:`~pyOMA.core.VarPLSCF.VarPLSCF`, with weights given *per setup*:

* pure-math properties of the reweighting matrix ``W(w)`` (identity at uniform
  weights, build-time equivalence, leave-one-out, PSD, convention ratio);
* build-time weighting (``build_subspace_matrices(weights=...)``): the point
  estimate becomes the weighted mean and the deviation factors follow it;
* the binding T-level identity ``devs_unweighted @ W == devs_weighted_build``;
* post-hoc reweighting, both the recompute path
  (``compute_modal_params_weighted``, Tier B) and the cached one
  (``apply_block_weights``, Tier A), including their agreement;
* the guard rails (weighted build, missing cache, malformed weights) and
  save/load round trips.
"""

import warnings

import numpy as np
import pytest

from pyOMA.core.MultiSetupSSI import VarPreGERSSI
from pyOMA.core.StabilDiagram import StabilCalc

from tests.system_ambient_ifrf import ambient_gaussian
from tests.test_preger_variance import _rod_signal, _mkps

MLAGS, NSEG, P, Q, MAXO = 60, 8, 15, 15, 10
CHS = [([5, 0, 1, 2], [0]), ([5, 3, 4], [0])]

# per-setup, non-uniform weight vectors (NSEG blocks each)
W_A = np.array([0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04])
W_B = np.array([0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25])
WEIGHTS = [W_A, W_B]
UNIFORM = [np.full(NSEG, 1.0 / NSEG), np.full(NSEG, 1.0 / NSEG)]

LOGGER = 'pyOMA.core.MultiSetupSSI'


def _build(weights=None, subspace_method='covariance', num_blocks=None,
           seed=11, **ctor_kwargs):
    """An identified-ready VarPreGERSSI over two setups of the same rod signal."""
    obj = VarPreGERSSI(**ctor_kwargs)
    signals = [_rod_signal(seed), _rod_signal(seed)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for sig, (channels, ref) in zip(signals, CHS):
            obj.add_setup(_mkps(sig, channels, ref, MLAGS, NSEG))
        obj.pair_channels()
        obj.build_subspace_matrices(
            num_block_columns=Q, num_block_rows=P,
            subspace_method=subspace_method, num_blocks=num_blocks,
            weights=weights)
    return obj


def _identify(obj, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        obj.compute_modal_params(MAXO, **kwargs)
    return obj


# ── pure-math properties of W(w) ─────────────────────────────────────────────

N_FEAT, N_BLOCKS = 12, 5


def _uniform_factor(X):
    """Unweighted deviation factor, as the deviation helpers build it."""
    n_b = X.shape[1]
    return (X - X.mean(axis=1, keepdims=True)) / np.sqrt(n_b * (n_b - 1))


def _build_time_factor(X, weights):
    """Build-time weighted deviation factor, as the deviation helpers build it."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    return (X - (X @ w)[:, np.newaxis]) * (np.sqrt(w) / np.sqrt(n_eff - 1.0))


@pytest.fixture
def rng():
    return np.random.default_rng(20260725)


class TestBlockWeightFactor:

    def test_substitution_matches_build_time_factor(self, rng):
        X = rng.standard_normal((N_FEAT, N_BLOCKS))
        w = rng.uniform(0.1, 1.0, N_BLOCKS)
        F = _uniform_factor(X)
        W = VarPreGERSSI._block_weight_factor(w, N_BLOCKS, 'substitution')
        assert np.allclose(F @ W, _build_time_factor(X, w), rtol=1e-11, atol=1e-13)

    def test_uniform_weights_act_as_identity(self, rng):
        X = rng.standard_normal((N_FEAT, N_BLOCKS))
        F = _uniform_factor(X)
        for convention in ('substitution', 'reliability', 'precision'):
            W_none = VarPreGERSSI._block_weight_factor(None, N_BLOCKS, convention)
            W_flat = VarPreGERSSI._block_weight_factor(
                np.full(N_BLOCKS, 1.0 / N_BLOCKS), N_BLOCKS, convention)
            assert np.allclose(F @ W_none, F, rtol=1e-11, atol=1e-13)
            assert np.allclose(F @ W_flat, F, rtol=1e-11, atol=1e-13)

    def test_zero_weight_is_leave_one_out(self, rng):
        X = rng.standard_normal((N_FEAT, N_BLOCKS))
        k = 2
        w = np.full(N_BLOCKS, 1.0 / (N_BLOCKS - 1))
        w[k] = 0.0
        FW = _uniform_factor(X) @ VarPreGERSSI._block_weight_factor(
            w, N_BLOCKS, 'substitution')
        assert np.allclose(FW[:, k], 0.0)
        F_del = _uniform_factor(np.delete(X, k, axis=1))
        assert np.allclose(FW @ FW.T, F_del @ F_del.T, rtol=1e-10, atol=1e-12)

    def test_reweighted_covariance_is_psd(self, rng):
        F = rng.standard_normal((7, N_BLOCKS))
        w = rng.uniform(0.1, 1.0, N_BLOCKS)
        for convention in ('substitution', 'reliability', 'precision'):
            FW = F @ VarPreGERSSI._block_weight_factor(w, N_BLOCKS, convention)
            cov = FW @ FW.T
            assert np.allclose(cov, cov.T)
            assert np.all(np.einsum('ij,ij->i', FW, FW) >= 0.0)
            assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_convention_variance_ratio(self, rng):
        """var(substitution) / var(reliability) == n_b / n_eff, elementwise."""
        X = rng.standard_normal((N_FEAT, N_BLOCKS))
        w = rng.uniform(0.1, 1.0, N_BLOCKS)
        n_eff = 1.0 / np.sum((w / w.sum()) ** 2)
        F = _uniform_factor(X)
        FWs = F @ VarPreGERSSI._block_weight_factor(w, N_BLOCKS, 'substitution')
        FWr = F @ VarPreGERSSI._block_weight_factor(w, N_BLOCKS, 'reliability')
        var_s = np.einsum('ij,ij->i', FWs, FWs)
        var_r = np.einsum('ij,ij->i', FWr, FWr)
        mask = var_r > 1e-12
        assert np.allclose(var_s[mask] / var_r[mask], N_BLOCKS / n_eff, rtol=1e-10)

    def test_single_block_mass(self, caplog, rng):
        w = np.zeros(N_BLOCKS)
        w[1] = 1.0
        with pytest.raises(ValueError):
            VarPreGERSSI._block_weight_factor(w, N_BLOCKS, 'reliability')
        with caplog.at_level('WARNING', logger=LOGGER):
            W = VarPreGERSSI._block_weight_factor(w, N_BLOCKS, 'substitution')
        assert np.all(W == 0.0)
        assert any('n_eff' in rec.message for rec in caplog.records)
        # 'precision' degenerates on its own, without NaNs
        F = rng.standard_normal((7, N_BLOCKS))
        W = VarPreGERSSI._block_weight_factor(w, N_BLOCKS, 'precision')
        assert np.all(np.isfinite(W))
        assert np.allclose(F @ W, 0.0)

    def test_argument_validation(self):
        with pytest.raises(ValueError):
            VarPreGERSSI._block_weight_factor(None, N_BLOCKS, 'bogus')
        with pytest.raises(ValueError):
            VarPreGERSSI._block_weight_factor([1.0, 1.0, 1.0], N_BLOCKS)
        with pytest.raises(ValueError):
            VarPreGERSSI._block_weight_factor([-0.1, 0.5, 0.2, 0.2, 0.2], N_BLOCKS)
        with pytest.raises(ValueError):  # complex weights are not admissible
            VarPreGERSSI._validate_weights(np.full(N_BLOCKS, 1 + 1j), N_BLOCKS)
        with pytest.raises(ValueError):  # all-zero mass
            VarPreGERSSI._validate_weights(np.zeros(N_BLOCKS), N_BLOCKS)
        assert VarPreGERSSI._validate_weights(None, N_BLOCKS) == (None, float(N_BLOCKS))
        weights, n_eff = VarPreGERSSI._validate_weights([2.0, 1.0, 1.0], 3)
        assert np.allclose(weights, [0.5, 0.25, 0.25])
        assert np.isclose(n_eff, 1.0 / (0.25 + 0.0625 + 0.0625))


# ── per-setup weight structure ───────────────────────────────────────────────

class TestSetupWeightStructure:

    @pytest.fixture(scope='class')
    def obj(self):
        return _build()

    def test_wrong_number_of_setups(self, obj):
        with pytest.raises(ValueError, match='one entry per setup'):
            obj._as_setup_weight_list([W_A])

    def test_scalar_entry_rejected(self, obj):
        with pytest.raises(ValueError, match='one-dimensional'):
            obj._as_setup_weight_list([0.5, 0.5])

    def test_non_sequence_rejected(self, obj):
        with pytest.raises(ValueError, match='sequence'):
            obj._as_setup_weight_list(0.5)

    def test_none_entries_are_uniform(self, obj):
        resolved = obj._as_setup_weight_list([W_A, None])
        assert resolved[1] is None
        assert np.allclose(resolved[0], W_A)

    def test_wrong_block_count_raises_at_build(self):
        with pytest.raises(ValueError):
            _build(weights=[np.ones(NSEG + 1), np.ones(NSEG)])


# ── build-time weighting ─────────────────────────────────────────────────────

class TestBuildTimeWeights:

    def test_uniform_weights_reproduce_unweighted_build(self):
        ref = _identify(_build())
        uni = _identify(_build(weights=UNIFORM))
        assert uni.weights is not None
        for j in range(len(ref.setup_n_b)):
            assert np.isclose(uni.n_eff[j], NSEG)
            assert np.allclose(uni.hankel_devs[j], ref.hankel_devs[j],
                               rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(uni.modal_frequencies, ref.modal_frequencies,
                                   rtol=1e-9, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(uni.std_frequencies, ref.std_frequencies,
                                   rtol=1e-7, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(uni.std_damping, ref.std_damping,
                                   rtol=1e-7, atol=1e-14, equal_nan=True)
        assert uni.num_blocks == ref.num_blocks == NSEG

    def test_nonuniform_weights_move_point_estimate_and_std(self):
        ref = _identify(_build())
        wgt = _identify(_build(weights=WEIGHTS))
        order = 4
        valid = np.where(np.isfinite(ref.modal_frequencies[order, :order]))[0]
        assert valid.size >= 2
        # the weighted mean Hankel is a different point estimate ...
        assert not np.allclose(wgt.modal_frequencies[order, valid],
                               ref.modal_frequencies[order, valid])
        # ... and the variance follows the reduced effective block count
        assert np.all(wgt.std_frequencies[order, valid] > 0)
        assert not np.allclose(wgt.std_frequencies[order, valid],
                               ref.std_frequencies[order, valid])
        for j, n_b in enumerate(wgt.setup_n_b):
            assert np.isclose(wgt.n_eff[j], 1.0 / np.sum(
                (WEIGHTS[j] / WEIGHTS[j].sum()) ** 2))
            assert wgt.n_eff[j] < n_b

    def test_weighted_mean_is_the_point_estimate(self):
        """Setup 0's weighted Hankel equals the weighted mean of its blocks."""
        obj = _build(weights=WEIGHTS)
        setup = obj.setups[0]
        corr = np.tensordot(obj.weights[0], setup['corr_matrices'], axes=(0, 0))
        corr = corr[obj.setup_row_orders[0], :, :][:, obj.ssi_ref_channels[0], :]
        expected = obj._hankel_from_corr(corr, obj.num_block_rows, obj.num_block_columns)
        U1, S1, V1_T = obj.svd_h1
        assert np.allclose((U1 * S1) @ V1_T, expected, rtol=1e-10, atol=1e-14)

    def test_mixed_none_entry(self):
        ref = _build()
        obj = _build(weights=[W_A, None])
        assert obj.weights[1] is None
        assert np.isclose(obj.n_eff[1], NSEG)
        # the unweighted setup keeps its unweighted deviations bit for bit
        assert np.allclose(obj.hankel_devs[1], ref.hankel_devs[1])
        assert not np.allclose(obj.hankel_devs[0], ref.hankel_devs[0])

    @pytest.mark.parametrize('subspace_method', ['covariance', 'projection'])
    def test_reweighted_devs_match_build_time_weighted(self, subspace_method):
        """Binding T-level identity: devs @ W(w) == build-time weighted devs.

        Holds exactly for both subspace sources -- the ``(I - w 1^T)`` factor
        re-centres the deviations on the new weighted mean, so the centring
        constant of the unweighted build cancels.
        """
        num_blocks = None if subspace_method == 'covariance' else NSEG
        ref = _build(subspace_method=subspace_method, num_blocks=num_blocks)
        wgt = _build(weights=WEIGHTS, subspace_method=subspace_method,
                     num_blocks=num_blocks)
        for j, n_b in enumerate(ref.setup_n_b):
            W = VarPreGERSSI._block_weight_factor(WEIGHTS[j], n_b, 'substitution')
            assert np.allclose(ref.hankel_devs[j] @ W, wgt.hankel_devs[j],
                               rtol=1e-9, atol=1e-14)

    def test_all_mass_on_one_block_raises(self):
        collapsed = np.zeros(NSEG)
        collapsed[0] = 1.0
        with pytest.raises(ValueError, match='single block'):
            _build(weights=[collapsed, None])

    def test_low_neff_warns(self, caplog):
        mild = np.zeros(NSEG)
        mild[:2] = 0.5
        with caplog.at_level('WARNING', logger=LOGGER):
            _build(weights=[mild, None])
        assert any('n_eff' in rec.message for rec in caplog.records)

    def test_num_blocks_follows_effective_count(self):
        obj = _build(weights=WEIGHTS)
        expected = min(1.0 / np.sum((w / w.sum()) ** 2) for w in WEIGHTS)
        assert obj.num_blocks == max(2, int(np.floor(expected)))
        assert obj.num_blocks < NSEG
        # StabilCalc still sees a usable std capability and dof
        assert StabilCalc(_identify(obj)).capabilities['std'] is True


# ── post-hoc reweighting: Tier B (recompute) ─────────────────────────────────

class TestPostHocRecompute:

    @pytest.mark.parametrize('subspace_method', ['covariance', 'projection'])
    def test_uniform_reweight_reproduces_unweighted(self, subspace_method):
        num_blocks = None if subspace_method == 'covariance' else NSEG
        obj = _identify(_build(subspace_method=subspace_method,
                               num_blocks=num_blocks))
        ref_f = obj.std_frequencies.copy()
        ref_d = obj.std_damping.copy()
        ref_ms = obj.std_mode_shapes.copy()
        ref_freq = obj.modal_frequencies.copy()

        obj.compute_modal_params_weighted(None, max_model_order=MAXO)

        np.testing.assert_allclose(obj.std_frequencies, ref_f,
                                   rtol=1e-9, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(obj.std_damping, ref_d,
                                   rtol=1e-9, atol=1e-14, equal_nan=True)
        # mode-shape stds pass through pinv(lambda I - A), whose retained
        # near-null direction is only cancelled by 'proj' to working precision;
        # its condition number (~1e15) turns the 1e-15 agreement of the
        # reweighted perturbation columns into ~1e-3 here, for a few modes
        np.testing.assert_allclose(obj.std_mode_shapes, ref_ms,
                                   rtol=5e-3, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(obj.modal_frequencies, ref_freq, equal_nan=True)
        assert obj.block_weights is None
        assert obj.num_blocks == NSEG

    def test_reweight_changes_std_not_point_estimates(self):
        obj = _identify(_build())
        f0 = obj.modal_frequencies.copy()
        d0 = obj.modal_damping.copy()
        ms0 = obj.mode_shapes.copy()
        std_f0 = obj.std_frequencies.copy()

        obj.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO)

        np.testing.assert_allclose(obj.modal_frequencies, f0, equal_nan=True)
        np.testing.assert_allclose(obj.modal_damping, d0, equal_nan=True)
        np.testing.assert_allclose(obj.mode_shapes, ms0, equal_nan=True)
        mask = np.isfinite(std_f0) & (std_f0 > 0)
        assert not np.allclose(obj.std_frequencies[mask], std_f0[mask])
        assert np.all(obj.std_frequencies[mask] >= 0)
        for j in range(len(WEIGHTS)):
            assert np.allclose(obj.block_weights[j],
                               WEIGHTS[j] / WEIGHTS[j].sum())
        assert obj.block_weight_convention == 'substitution'
        assert obj.num_blocks < NSEG

    def test_convention_ordering(self):
        """var(substitution) / var(reliability) == n_b / n_eff, per setup.

        Both setups get the same n_b/n_eff ratio here, so the modal variances
        scale by that single constant.
        """
        obj = _identify(_build())
        weights = [W_A, W_A]
        obj.compute_modal_params_weighted(weights, convention='substitution',
                                          max_model_order=MAXO)
        std_sub = obj.std_frequencies.copy()
        obj.compute_modal_params_weighted(weights, convention='reliability',
                                          max_model_order=MAXO)
        std_rel = obj.std_frequencies.copy()
        mask = np.isfinite(std_sub) & (std_sub > 0)
        n_eff = 1.0 / np.sum((W_A / W_A.sum()) ** 2)
        assert np.allclose((std_sub[mask] / std_rel[mask]) ** 2, NSEG / n_eff,
                           rtol=1e-8)

    def test_zero_weight_drops_a_block(self):
        """A zero weight eliminates that block's perturbation column exactly."""
        obj = _identify(_build())
        weights = np.full(NSEG, 1.0 / (NSEG - 1))
        weights[3] = 0.0
        obj.compute_modal_params_weighted([weights, weights],
                                          max_model_order=MAXO)
        factors, _, _ = obj._setup_weight_factors([weights, weights],
                                                  'substitution')
        for j, n_b in enumerate(obj.setup_n_b):
            assert np.allclose(obj.hankel_devs[j] @ factors[j][:, 3], 0.0)
            assert factors[j].shape == (n_b, n_b)

    def test_orders_restriction(self):
        obj = _identify(_build())
        obj.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO,
                                          orders=[6])
        assert np.any(obj.modal_frequencies[6, :] > 0)
        untouched = np.delete(obj.modal_frequencies, 6, axis=0)
        assert np.all(untouched == 0)
        with pytest.raises(ValueError):
            obj.compute_modal_params_weighted(None, max_model_order=MAXO,
                                              orders=[MAXO])

    def test_weighted_build_rejects_posthoc(self):
        obj = _identify(_build(weights=WEIGHTS))
        with pytest.raises(NotImplementedError):
            obj.compute_modal_params_weighted(UNIFORM, max_model_order=MAXO)
        with pytest.raises(NotImplementedError):
            obj.apply_block_weights(UNIFORM)

    def test_requires_built_subspace(self):
        obj = VarPreGERSSI()
        with pytest.raises(RuntimeError):
            obj.compute_modal_params_weighted(None)
        with pytest.raises(RuntimeError):
            obj.apply_block_weights(None)

    def test_all_mass_one_block_zeroes_std(self, caplog):
        obj = _identify(_build())
        collapsed = np.zeros(NSEG)
        collapsed[1] = 1.0
        with caplog.at_level('WARNING', logger=LOGGER):
            obj.compute_modal_params_weighted([collapsed, collapsed],
                                              max_model_order=MAXO)
        valid = np.isfinite(obj.std_frequencies)
        assert np.allclose(obj.std_frequencies[valid], 0.0)
        assert any('n_eff' in rec.message for rec in caplog.records)
        with pytest.raises(ValueError):
            obj.compute_modal_params_weighted(
                [collapsed, collapsed], convention='reliability',
                max_model_order=MAXO)


# ── post-hoc reweighting: Tier A (cached per-mode factors) ───────────────────

class TestPostHocCached:

    @pytest.mark.parametrize('subspace_method', ['covariance', 'projection'])
    def test_tier_a_equals_tier_b(self, subspace_method):
        num_blocks = None if subspace_method == 'covariance' else NSEG
        obj_b = _identify(_build(subspace_method=subspace_method,
                                 num_blocks=num_blocks))
        obj_b.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO)

        obj_a = _identify(
            _build(subspace_method=subspace_method, num_blocks=num_blocks),
            cache_variance_factors=True, cache_dtype=np.float64)
        obj_a.apply_block_weights(WEIGHTS)

        np.testing.assert_allclose(obj_a.std_frequencies, obj_b.std_frequencies,
                                   rtol=1e-10, atol=1e-13, equal_nan=True)
        np.testing.assert_allclose(obj_a.std_damping, obj_b.std_damping,
                                   rtol=1e-10, atol=1e-13, equal_nan=True)
        # see TestPostHocRecompute.test_uniform_reweight_reproduces_unweighted
        # for the ill-conditioned mode-shape chain behind the looser tolerance
        np.testing.assert_allclose(obj_a.std_mode_shapes.real,
                                   obj_b.std_mode_shapes.real,
                                   rtol=5e-3, atol=1e-13, equal_nan=True)
        np.testing.assert_allclose(obj_a.std_mode_shapes.imag,
                                   obj_b.std_mode_shapes.imag,
                                   rtol=5e-3, atol=1e-13, equal_nan=True)

    def test_uniform_restore(self):
        obj = _identify(_build(), cache_variance_factors=True,
                        cache_dtype=np.float64)
        ref_f = obj.std_frequencies.copy()
        ref_d = obj.std_damping.copy()
        ref_ms = obj.std_mode_shapes.copy()

        obj.apply_block_weights(WEIGHTS)       # reweight ...
        assert obj.num_blocks < NSEG
        obj.apply_block_weights(None)          # ... then restore

        np.testing.assert_allclose(obj.std_frequencies, ref_f,
                                   rtol=1e-10, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(obj.std_damping, ref_d,
                                   rtol=1e-10, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(obj.std_mode_shapes, ref_ms,
                                   rtol=1e-10, atol=1e-14, equal_nan=True)
        assert obj.block_weights is None
        assert obj.num_blocks == NSEG

    def test_float32_default_precision(self):
        obj_b = _identify(_build())
        obj_b.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO)
        obj_a = _identify(_build(), cache_variance_factors=True)
        assert obj_a.U_fixi_cache[6].dtype == np.float32
        obj_a.apply_block_weights(WEIGHTS)
        valid = np.isfinite(obj_b.std_frequencies) & (obj_b.std_frequencies > 0)
        assert np.allclose(obj_a.std_frequencies[valid],
                           obj_b.std_frequencies[valid], rtol=1e-5)

    def test_freqdamp_mode(self):
        obj = _identify(_build(), cache_variance_factors=True, cache='freqdamp',
                        cache_dtype=np.float64)
        assert obj.U_fixi_cache is not None
        assert obj.U_phii_cache is None
        ms_before = obj.std_mode_shapes.copy()

        obj_b = _identify(_build())
        obj_b.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO)
        obj.apply_block_weights(WEIGHTS)

        np.testing.assert_allclose(obj.std_frequencies, obj_b.std_frequencies,
                                   rtol=1e-9, atol=1e-13, equal_nan=True)
        # mode-shape stds keep the weighting they were computed under
        np.testing.assert_array_equal(obj.std_mode_shapes.real, ms_before.real)
        np.testing.assert_array_equal(obj.std_mode_shapes.imag, ms_before.imag)

    def test_cache_shapes_and_padding(self):
        obj = _identify(_build(), cache_variance_factors=True,
                        cache_dtype=np.float64)
        n_l = obj.num_analised_channels
        num_cols = int(np.sum(obj.setup_n_b))
        for order, U_fixi in obj.U_fixi_cache.items():
            assert U_fixi.shape == (order, 2, num_cols)
            assert obj.U_phii_cache[order].shape == (order, 2 * n_l, num_cols)
            # non-physical slots stay nan, exactly as the std_* arrays do
            nan_rows = np.all(np.isnan(U_fixi), axis=(1, 2))
            assert np.array_equal(nan_rows,
                                  np.isnan(obj.std_frequencies[order, :order]))

    def test_missing_cache_raises(self):
        obj = _identify(_build())
        assert obj.U_fixi_cache is None
        with pytest.raises(RuntimeError):
            obj.apply_block_weights(WEIGHTS)

    def test_constructor_default_enables_cache(self):
        obj = _identify(_build(cache_variance_factors=True))
        assert obj.U_fixi_cache is not None
        obj.apply_block_weights(WEIGHTS)   # works without a per-call opt-in

    def test_recompute_path_keeps_the_caches(self):
        """Tier B holds unweighted factors, so the caches stay valid across it."""
        obj = _identify(_build(), cache_variance_factors=True,
                        cache_dtype=np.float64)
        cached = {order: U.copy() for order, U in obj.U_fixi_cache.items()}
        obj.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO)
        assert obj.U_fixi_cache is not None
        for order, U in cached.items():
            np.testing.assert_allclose(obj.U_fixi_cache[order], U, equal_nan=True)
        std_b = obj.std_frequencies.copy()
        obj.apply_block_weights(WEIGHTS)   # Tier A on top reproduces Tier B
        np.testing.assert_allclose(obj.std_frequencies, std_b,
                                   rtol=1e-10, atol=1e-13, equal_nan=True)

    def test_fresh_identification_resets_cache(self):
        obj = _identify(_build(), cache_variance_factors=True)
        obj.apply_block_weights(WEIGHTS)
        assert obj.block_weights is not None
        _identify(obj)
        assert obj.block_weights is None
        assert obj.block_weight_convention is None

    def test_error_paths(self):
        obj = _identify(_build(), cache_variance_factors=True)
        with pytest.raises(ValueError):     # wrong per-setup structure
            obj.apply_block_weights([W_A])
        with pytest.raises(ValueError):     # wrong block count
            obj.apply_block_weights([np.ones(NSEG + 1), np.ones(NSEG)])
        negative = np.full(NSEG, 0.2)
        negative[0] = -0.1
        with pytest.raises(ValueError):
            obj.apply_block_weights([negative, negative])
        with pytest.raises(ValueError):
            obj.apply_block_weights(WEIGHTS, convention='bogus')
        with pytest.raises(ValueError):
            obj.compute_modal_params(MAXO, cache_variance_factors=True,
                                     cache='bogus')
        with pytest.raises(ValueError):
            VarPreGERSSI(cache='bogus')


# ── persistence ──────────────────────────────────────────────────────────────

class TestPersistence:

    # a mixed list packs to a 1-D object array, an all-vectors one to a 2-D
    # object array; both must survive the round trip
    @pytest.mark.parametrize('weights', [[W_A, None], WEIGHTS])
    def test_round_trip_with_build_time_weights(self, tmp_path, weights):
        obj = _identify(_build(weights=weights))
        fname = str(tmp_path / 'weighted.npz')
        obj.save_state(fname)
        loaded = VarPreGERSSI.load_state(fname)
        for j, this_weights in enumerate(weights):
            if this_weights is None:
                assert loaded.weights[j] is None
            else:
                assert np.allclose(loaded.weights[j],
                                   this_weights / this_weights.sum())
        assert np.allclose(loaded.n_eff, obj.n_eff)
        assert loaded.num_blocks == obj.num_blocks
        np.testing.assert_allclose(loaded.std_frequencies, obj.std_frequencies,
                                   rtol=1e-10, equal_nan=True)
        # post-hoc reweighting stays rejected for a weighted build
        with pytest.raises(NotImplementedError):
            loaded.compute_modal_params_weighted(None, max_model_order=MAXO)

    def test_round_trip_preserves_tier_a_cache(self, tmp_path):
        obj = _identify(_build(), cache_variance_factors=True,
                        cache_dtype=np.float64)
        obj.apply_block_weights(WEIGHTS)
        ref_f = obj.std_frequencies.copy()

        fname = str(tmp_path / 'tier_a.npz')
        obj.save_state(fname)
        loaded = VarPreGERSSI.load_state(fname)

        assert loaded.U_fixi_cache is not None
        assert loaded.U_phii_cache is not None
        assert loaded.setup_n_b == obj.setup_n_b
        assert loaded.block_weight_convention == 'substitution'
        for j in range(len(WEIGHTS)):
            assert np.allclose(loaded.block_weights[j],
                               WEIGHTS[j] / WEIGHTS[j].sum())
        loaded.apply_block_weights(WEIGHTS)   # reweight from the restored cache
        np.testing.assert_allclose(loaded.std_frequencies, ref_f,
                                   rtol=1e-10, atol=1e-14, equal_nan=True)

    def test_round_trip_without_cache(self, tmp_path):
        obj = _identify(_build())
        fname = str(tmp_path / 'plain.npz')
        obj.save_state(fname)
        with np.load(fname, allow_pickle=True) as archive:
            assert 'self.U_fixi_cache' not in archive
            assert 'self.weights' not in archive
        loaded = VarPreGERSSI.load_state(fname)
        assert loaded.weights is None
        assert loaded.U_fixi_cache is None
        assert loaded.block_weights is None
        # Tier A unavailable ...
        with pytest.raises(RuntimeError):
            loaded.apply_block_weights(WEIGHTS)
        # ... and so is Tier B, whose block deviations are not persisted
        with pytest.raises(RuntimeError, match='deviations'):
            loaded.compute_modal_params_weighted(WEIGHTS, max_model_order=MAXO)


# ── Monte-Carlo acceptance gate for the build-time weighted estimator ────────

@pytest.mark.slow
class TestBuildTimeWeightedMonteCarlo:
    """Predicted std of the *weighted* estimator vs its ensemble scatter.

    Same design as ``TestVarPreGERSSIMonteCarlo`` in ``test_preger_variance``,
    but every realization is identified twice from the same signals: once
    unweighted, and once with half of every setup's blocks weighted to zero
    (``n_eff = n_b / 2``).  The weighted point estimate then scatters by about
    ``sqrt(2)`` more than the unweighted one, so the ensemble is a gate the
    build-time weighted covariance factor can only pass if the weights really
    enter it -- the unweighted prediction on the same ensemble is checked to
    fall *outside* the accepted band, which is what makes the gate non-vacuous.
    """

    N = 8192
    MLAGS, NSEG_MC, P_MC, ORDER = 100, 20, 8, 4
    CHS_MC = [([5, 1, 2], [0]), ([5, 3, 4], [0])]

    def _weights(self):
        """Keep every second block of every setup: n_eff == n_b / 2 exactly."""
        keep = np.zeros(self.NSEG_MC)
        keep[::2] = 1.0
        return [keep.copy(), keep.copy()]

    def _run(self, n_runs):
        weights = self._weights()
        freqs, damps, pred_f, pred_d, pred_f_uni = [], [], [], [], []
        for run in range(n_runs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                prep_signals = []
                for offset, (channels, ref) in enumerate(self.CHS_MC):
                    sig = _rod_signal(2 * run + offset, self.N, ambient_gaussian)
                    prep_signals.append(
                        _mkps(sig, channels, ref, self.MLAGS, self.NSEG_MC))
                objects = []
                for build_weights in (weights, None):
                    obj = VarPreGERSSI()
                    for prep in prep_signals:
                        obj.add_setup(prep)
                    obj.pair_channels()
                    obj.build_subspace_matrices(
                        num_block_columns=self.P_MC, num_block_rows=self.P_MC,
                        weights=build_weights)
                    obj.compute_modal_params(self.ORDER + 1)
                    objects.append(obj)
            wgt, uni = objects
            order = self.ORDER
            f = wgt.modal_frequencies[order, :order]
            valid = np.where(np.isfinite(f) & (f > 0))[0]
            if valid.size < 2:
                continue
            valid = valid[np.argsort(f[valid])][:2]
            freqs.append(f[valid])
            damps.append(wgt.modal_damping[order, valid])
            pred_f.append(wgt.std_frequencies[order, valid])
            pred_d.append(wgt.std_damping[order, valid])
            # the unweighted prediction, paired mode by mode with the weighted one
            f_uni = uni.modal_frequencies[order, :order]
            valid_uni = np.where(np.isfinite(f_uni) & (f_uni > 0))[0]
            valid_uni = valid_uni[np.argsort(f_uni[valid_uni])][:2]
            pred_f_uni.append(uni.std_frequencies[order, valid_uni]
                              if valid_uni.size == 2 else [np.nan, np.nan])
        return (np.array(freqs), np.array(damps), np.array(pred_f),
                np.array(pred_d), np.array(pred_f_uni))

    def test_weighted_std_matches_ensemble_scatter(self):
        freqs, damps, pred_f, pred_d, pred_f_uni = self._run(100)
        assert freqs.shape[0] >= 80
        mc_f = np.std(freqs, axis=0, ddof=1)
        ratio_f = mc_f / np.mean(pred_f, axis=0)
        ratio_d = np.std(damps, axis=0, ddof=1) / np.mean(pred_d, axis=0)
        # same loose band as the unweighted gate: MC SEM plus block-correlation slack
        assert np.all(ratio_f > 0.8) and np.all(ratio_f < 1.3), f'freq ratio {ratio_f}'
        assert np.all(ratio_d > 0.7) and np.all(ratio_d < 1.4), f'damp ratio {ratio_d}'

        # non-vacuous: the weights genuinely scale the prediction, by the
        # sqrt(n_b / n_eff) == sqrt(2) of halving the effective block count.
        # Both predictions come from the same data, so their ensemble-mean ratio
        # is far tighter than the Monte-Carlo scatter itself.
        pred_ratio = np.mean(pred_f, axis=0) / np.nanmean(pred_f_uni, axis=0)
        assert np.allclose(pred_ratio, np.sqrt(2.0), rtol=0.15), \
            f'weighted/unweighted prediction ratio {pred_ratio}'
        # ... so predicting this ensemble with the unweighted factor underestimates
        ratio_uni = mc_f / np.nanmean(pred_f_uni, axis=0)
        assert np.all(ratio_uni > 1.2), f'unweighted prediction ratio {ratio_uni}'
