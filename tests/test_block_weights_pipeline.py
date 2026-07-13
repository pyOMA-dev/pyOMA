# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline-level tests for post-hoc block reweighting of VarSSIRef.

Phase 1 (Tier B): ``compute_modal_params_weighted`` reruns the modal-order loop
with the cached fast variance factors right-multiplied by ``W(w)``.  These tests
exercise the full build -> state matrices -> sensitivities -> modal loop chain on
the synthetic fixture and check:

* uniform weights reproduce the unweighted ``std_*`` arrays (binding invariant),
  for both ``lsq_method`` and both ``subspace_method`` values;
* reweighting changes the variances but leaves the point estimates untouched;
* the reweighted Hankel covariance factor ``T @ W`` matches a fresh build-time
  weighted run (isolates the T-level identity at pipeline scale);
* the guard rails (weighted build, slow algorithm, missing sensitivities).
"""

import numpy as np
import pytest

from pyOMA.core.VarSSIRef import VarSSIRef

NBC = 15          # num_block_columns
NUM_BLOCKS = 6
MAX_ORDER = 8


def _prepare(prep_signals, lsq_method='pinv', subspace_method='covariance',
             num_blocks=NUM_BLOCKS, variance_algo='fast'):
    """Build an *unweighted* fast-variance pipeline ready for reweighting."""
    obj = VarSSIRef(prep_signals)
    obj.build_subspace_mat(
        num_block_columns=NBC, num_blocks=num_blocks,
        subspace_method=subspace_method)
    obj.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method=lsq_method)
    obj.prepare_sensitivities(variance_algo=variance_algo)
    return obj


@pytest.mark.parametrize('lsq_method', ['pinv', 'qr'])
@pytest.mark.parametrize('subspace_method', ['covariance', 'projection'])
def test_uniform_reweight_reproduces_unweighted(
        prep_signals_with_corr, lsq_method, subspace_method):
    obj = _prepare(prep_signals_with_corr, lsq_method, subspace_method)
    obj.compute_modal_params()
    ref_f = obj.std_frequencies.copy()
    ref_d = obj.std_damping.copy()
    ref_ms = obj.std_mode_shapes.copy()
    ref_freq = obj.modal_frequencies.copy()

    obj.compute_modal_params_weighted(None)

    assert np.allclose(obj.std_frequencies, ref_f, rtol=1e-10, atol=1e-12)
    assert np.allclose(obj.std_damping, ref_d, rtol=1e-10, atol=1e-12)
    assert np.allclose(obj.std_mode_shapes, ref_ms, rtol=1e-10, atol=1e-12)
    # point estimates are recomputed identically
    assert np.allclose(obj.modal_frequencies, ref_freq)
    assert obj.block_weights is None


def test_reweight_changes_std_not_point_estimates(prep_signals_with_corr):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params()
    f0 = obj.modal_frequencies.copy()
    d0 = obj.modal_damping.copy()
    ms0 = obj.mode_shapes.copy()
    std_f0 = obj.std_frequencies.copy()

    w = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    obj.compute_modal_params_weighted(w)

    # point estimates identical, variances genuinely changed
    assert np.allclose(obj.modal_frequencies, f0)
    assert np.allclose(obj.modal_damping, d0)
    assert np.allclose(obj.mode_shapes, ms0)
    mask = std_f0 > 0
    assert not np.allclose(obj.std_frequencies[mask], std_f0[mask])
    assert np.all(obj.std_frequencies[mask] >= 0)
    assert np.allclose(obj.block_weights, w / w.sum())
    assert obj.block_weight_convention == 'substitution'


def test_reweighted_T_matches_build_time_weighted(prep_signals_with_corr):
    """T @ W (substitution) == the branch's build-time weighted Hankel cov."""
    obj = _prepare(prep_signals_with_corr)
    T = obj.hankel_cov_matrix  # uniform T
    w = np.array([0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    W = VarSSIRef._block_weight_factor(w, NUM_BLOCKS, 'substitution')

    obj_w = VarSSIRef(prep_signals_with_corr)
    obj_w.build_subspace_mat(
        num_block_columns=NBC, num_blocks=NUM_BLOCKS,
        subspace_method='covariance', weights=w)
    obj_w.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method='pinv')
    obj_w.prepare_sensitivities(variance_algo='fast')

    assert np.allclose(T @ W, obj_w.hankel_cov_matrix, rtol=1e-9, atol=1e-12)


def test_convention_ordering_substitution_vs_reliability(prep_signals_with_corr):
    """var(substitution) >= var(reliability): substitution is the conservative one."""
    obj = _prepare(prep_signals_with_corr)
    w = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    obj.compute_modal_params_weighted(w, convention='substitution')
    std_sub = obj.std_frequencies.copy()
    obj.compute_modal_params_weighted(w, convention='reliability')
    std_rel = obj.std_frequencies.copy()
    mask = std_sub > 0
    # ratio of variances is the constant n_b / n_eff > 1
    ratio = (std_sub[mask] / std_rel[mask]) ** 2
    n_eff = 1.0 / np.sum((w / w.sum()) ** 2)
    assert np.allclose(ratio, NUM_BLOCKS / n_eff, rtol=1e-8)


def test_orders_restriction(prep_signals_with_corr):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params_weighted(np.linspace(1.0, 2.0, NUM_BLOCKS), orders=[6])
    assert np.any(obj.modal_frequencies[6, :] > 0)
    untouched = np.delete(obj.modal_frequencies, 6, axis=0)
    assert np.all(untouched == 0)
    with pytest.raises(ValueError):
        obj.compute_modal_params_weighted(None, orders=[MAX_ORDER])


def test_weighted_build_rejects_posthoc_reweight(prep_signals_with_corr):
    obj = VarSSIRef(prep_signals_with_corr)
    obj.build_subspace_mat(
        num_block_columns=NBC, num_blocks=4, subspace_method='covariance',
        weights=[0.4, 0.3, 0.2, 0.1])
    obj.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method='pinv')
    obj.prepare_sensitivities(variance_algo='fast')
    with pytest.raises(NotImplementedError):
        obj.compute_modal_params_weighted([0.25, 0.25, 0.25, 0.25])


def test_slow_algo_rejects_posthoc_reweight(prep_signals_with_corr):
    # Tier B is fast-only; slow-algorithm reweighting is not implemented here.
    # Flip variance_algo on an already fast-prepared object to hit the guard
    # (the slow covariance path is unrelated to reweighting and out of scope).
    obj = _prepare(prep_signals_with_corr)
    obj.variance_algo = 'slow'
    with pytest.raises(NotImplementedError):
        obj.compute_modal_params_weighted(np.linspace(1.0, 2.0, NUM_BLOCKS))


def test_reweight_requires_sensitivities(prep_signals_with_corr):
    obj = VarSSIRef(prep_signals_with_corr)
    obj.build_subspace_mat(
        num_block_columns=NBC, num_blocks=4, subspace_method='covariance')
    obj.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method='pinv')
    with pytest.raises(RuntimeError):
        obj.compute_modal_params_weighted(None)


# ── Phase 2: Tier A (per-mode cache + apply_block_weights) ──────────────────

RANDOM_W = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])


@pytest.mark.parametrize('lsq_method', ['pinv', 'qr'])
@pytest.mark.parametrize('subspace_method', ['covariance', 'projection'])
def test_tier_a_equals_tier_b(prep_signals_with_corr, lsq_method, subspace_method):
    """apply_block_weights (U @ W) == compute_modal_params_weighted (recompute)."""
    obj_b = _prepare(prep_signals_with_corr, lsq_method, subspace_method)
    obj_b.compute_modal_params_weighted(RANDOM_W)

    obj_a = _prepare(prep_signals_with_corr, lsq_method, subspace_method)
    obj_a.compute_modal_params(cache_variance_factors=True, cache_dtype=np.float64)
    obj_a.apply_block_weights(RANDOM_W)

    valid = ~np.isnan(obj_b.std_frequencies)
    assert np.allclose(obj_a.std_frequencies[valid], obj_b.std_frequencies[valid],
                       rtol=1e-9, atol=1e-13)
    assert np.allclose(obj_a.std_damping[valid], obj_b.std_damping[valid],
                       rtol=1e-9, atol=1e-13)
    assert np.allclose(obj_a.std_mode_shapes.real, obj_b.std_mode_shapes.real,
                       rtol=1e-9, atol=1e-13)
    assert np.allclose(obj_a.std_mode_shapes.imag, obj_b.std_mode_shapes.imag,
                       rtol=1e-9, atol=1e-13)


def test_tier_a_uniform_restore(prep_signals_with_corr):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params(cache_variance_factors=True, cache_dtype=np.float64)
    ref_f = obj.std_frequencies.copy()
    ref_ms = obj.std_mode_shapes.copy()

    obj.apply_block_weights(RANDOM_W)          # reweight ...
    obj.apply_block_weights(None)              # ... then restore uniform

    assert np.allclose(obj.std_frequencies, ref_f, rtol=1e-10, atol=1e-14)
    assert np.allclose(obj.std_mode_shapes.real, ref_ms.real, rtol=1e-10, atol=1e-14)
    assert np.allclose(obj.std_mode_shapes.imag, ref_ms.imag, rtol=1e-10, atol=1e-14)
    assert obj.block_weights is None


def test_tier_a_float32_default_precision(prep_signals_with_corr):
    obj_b = _prepare(prep_signals_with_corr)
    obj_b.compute_modal_params_weighted(RANDOM_W)
    obj_a = _prepare(prep_signals_with_corr)
    obj_a.compute_modal_params(cache_variance_factors=True)  # float32 default
    assert obj_a.U_fixi_cache[6].dtype == np.float32
    obj_a.apply_block_weights(RANDOM_W)
    valid = obj_b.std_frequencies > 0
    assert np.allclose(obj_a.std_frequencies[valid], obj_b.std_frequencies[valid],
                       rtol=1e-5)


def test_tier_a_freqdamp_mode(prep_signals_with_corr):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params(
        cache_variance_factors=True, cache='freqdamp', cache_dtype=np.float64)
    assert obj.U_fixi_cache is not None
    assert obj.U_phii_cache is None
    ms_before = obj.std_mode_shapes.copy()

    obj_b = _prepare(prep_signals_with_corr)
    obj_b.compute_modal_params_weighted(RANDOM_W)

    obj.apply_block_weights(RANDOM_W)
    valid = ~np.isnan(obj_b.std_frequencies)
    # frequency/damping reweighted exactly; mode-shape stds left untouched
    assert np.allclose(obj.std_frequencies[valid], obj_b.std_frequencies[valid],
                       rtol=1e-9, atol=1e-13)
    assert np.array_equal(obj.std_mode_shapes, ms_before)


def test_apply_block_weights_without_cache_raises(prep_signals_with_corr):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params()  # no caching
    assert obj.U_fixi_cache is None
    with pytest.raises(RuntimeError):
        obj.apply_block_weights(RANDOM_W)


def test_constructor_cache_default(prep_signals_with_corr):
    obj = VarSSIRef(prep_signals_with_corr, cache_variance_factors=True)
    obj.build_subspace_mat(
        num_block_columns=NBC, num_blocks=NUM_BLOCKS, subspace_method='covariance')
    obj.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method='pinv')
    obj.prepare_sensitivities(variance_algo='fast')
    obj.compute_modal_params()  # constructor default enables caching
    assert obj.U_fixi_cache is not None
    obj.apply_block_weights(RANDOM_W)  # works without RuntimeError


def test_apply_block_weights_rejects_weighted_build(prep_signals_with_corr):
    obj = VarSSIRef(prep_signals_with_corr, cache_variance_factors=True)
    obj.build_subspace_mat(
        num_block_columns=NBC, num_blocks=4, subspace_method='covariance',
        weights=[0.4, 0.3, 0.2, 0.1])
    obj.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method='pinv')
    obj.prepare_sensitivities(variance_algo='fast')
    obj.compute_modal_params()  # caches the weighted factors
    with pytest.raises(NotImplementedError):
        obj.apply_block_weights([0.25, 0.25, 0.25, 0.25])


def test_save_load_preserves_tier_a_cache(prep_signals_with_corr, tmp_path):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params(cache_variance_factors=True, cache_dtype=np.float64)
    obj.apply_block_weights(RANDOM_W)
    ref_f = obj.std_frequencies.copy()

    fname = tmp_path / 'tier_a.npz'
    obj.save_state(str(fname))
    loaded = VarSSIRef.load_state(str(fname), prep_signals_with_corr)

    assert loaded.U_fixi_cache is not None
    assert loaded.U_phii_cache is not None
    assert np.allclose(loaded.block_weights, RANDOM_W / RANDOM_W.sum())
    assert loaded.block_weight_convention == 'substitution'
    loaded.apply_block_weights(RANDOM_W)  # reweight from the restored cache
    assert np.allclose(loaded.std_frequencies, ref_f, rtol=1e-10, atol=1e-14)


def test_load_legacy_archive_without_cache(prep_signals_with_corr, tmp_path):
    obj = _prepare(prep_signals_with_corr)
    obj.compute_modal_params()  # no cache
    fname = tmp_path / 'legacy.npz'
    obj.save_state(str(fname))
    with np.load(str(fname), allow_pickle=True) as arch:
        assert 'self.U_fixi_cache' not in arch

    loaded = VarSSIRef.load_state(str(fname), prep_signals_with_corr)
    assert loaded.U_fixi_cache is None
    # Tier A unavailable ...
    with pytest.raises(RuntimeError):
        loaded.apply_block_weights(RANDOM_W)
    # ... but the Tier B recompute path still works from the restored Q factors
    loaded.compute_modal_params_weighted(RANDOM_W)
    valid = ~np.isnan(loaded.std_frequencies)
    assert np.all(loaded.std_frequencies[valid] >= 0)
