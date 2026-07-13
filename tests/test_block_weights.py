# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-0 unit tests for ``VarSSIRef._block_weight_factor`` (post-hoc block
reweighting matrix ``W(w)``).

These are pure-math tests on synthetic random data -- no pyOMA pipeline is
built.  They pin the verified properties of Eq. (I) of the task brief:

1. ``T @ W`` (substitution) reproduces the build-time weighted Hankel
   covariance factor ``T_w`` for the same weights, machine precision.
2. Uniform weights make ``W`` act as the identity on an already-centered ``T``.
3. A zero weight on block ``k`` gives, at covariance level, the from-scratch
   leave-one-out estimate on the remaining ``n_b - 1`` blocks.
4. ``cov_w = (F W)(F W)^T`` is PSD, so row variances are non-negative.

and the §2.2 convention ratio ``var(substitution) / var(reliability) = n_b / n_eff``.
"""

import numpy as np
import pytest

from pyOMA.core.VarSSIRef import VarSSIRef

N_FEAT = 12
N_BLOCKS = 5


def _uniform_T(X):
    """Uniform (unweighted) Hankel covariance factor from block columns ``X``.

    Mirrors :meth:`VarSSIRef._compute_hankel_cov_matrix` in the unweighted
    branch: centre on the uniform mean, then scale by ``1/sqrt(n_b**2 (n_b-1))``.
    """
    n_b = X.shape[1]
    T = X - X.mean(axis=1, keepdims=True)
    if n_b > 1:
        T = T / np.sqrt(n_b ** 2 * (n_b - 1))
    return T


def _branch_weighted_T(X, weights):
    """Build-time weighted Hankel covariance factor (substitution convention).

    Mirrors the weighted branch of :meth:`_compute_hankel_cov_matrix`: centre on
    the *weighted* mean, scale columns by ``sqrt(w)``, divide by
    ``sqrt(n_eff (n_eff-1))``.
    """
    w = np.asarray(weights, float)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    T = X - (X @ w)[:, np.newaxis]
    T = T * np.sqrt(w)[np.newaxis, :]
    if n_eff > 1:
        T = T / np.sqrt(n_eff * (n_eff - 1))
    else:
        T = np.zeros_like(T)
    return T


@pytest.fixture
def rng():
    return np.random.default_rng(20260713)


def test_substitution_matches_branch_weighted_T(rng):
    """Property 1: T @ W == build-time weighted T (substitution)."""
    X = rng.standard_normal((N_FEAT, N_BLOCKS))
    w = rng.uniform(0.1, 1.0, N_BLOCKS)
    T = _uniform_T(X)
    W = VarSSIRef._block_weight_factor(w, N_BLOCKS, 'substitution')
    assert np.allclose(T @ W, _branch_weighted_T(X, w), rtol=1e-11, atol=1e-13)


def test_uniform_weights_act_as_identity_on_T(rng):
    """Property 2: uniform weights (None or explicit) leave a centered T fixed."""
    X = rng.standard_normal((N_FEAT, N_BLOCKS))
    T = _uniform_T(X)
    for convention in ('substitution', 'reliability', 'precision'):
        W_none = VarSSIRef._block_weight_factor(None, N_BLOCKS, convention)
        W_flat = VarSSIRef._block_weight_factor(
            np.full(N_BLOCKS, 1.0 / N_BLOCKS), N_BLOCKS, convention)
        assert np.allclose(T @ W_none, T, rtol=1e-11, atol=1e-13)
        assert np.allclose(T @ W_flat, T, rtol=1e-11, atol=1e-13)


def test_zero_weight_is_leave_one_out(rng):
    """Property 3: w_k = 0 reproduces the from-scratch (n_b-1)-block covariance."""
    X = rng.standard_normal((N_FEAT, N_BLOCKS))
    k = 2
    w = np.full(N_BLOCKS, 1.0 / (N_BLOCKS - 1))
    w[k] = 0.0
    W = VarSSIRef._block_weight_factor(w, N_BLOCKS, 'substitution')
    TW = _uniform_T(X) @ W
    # the eliminated column vanishes and the covariance matches the fresh run
    assert np.allclose(TW[:, k], 0.0)
    T_del = _uniform_T(np.delete(X, k, axis=1))
    assert np.allclose(TW @ TW.T, T_del @ T_del.T, rtol=1e-10, atol=1e-12)


def test_reweighted_covariance_is_psd(rng):
    """Property 4: cov_w = (F W)(F W)^T is symmetric PSD; variances >= 0."""
    F = rng.standard_normal((7, N_BLOCKS))
    w = rng.uniform(0.1, 1.0, N_BLOCKS)
    for convention in ('substitution', 'reliability', 'precision'):
        FW = F @ VarSSIRef._block_weight_factor(w, N_BLOCKS, convention)
        cov = FW @ FW.T
        assert np.allclose(cov, cov.T)
        assert np.all(np.einsum('ij,ij->i', FW, FW) >= 0.0)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)


def test_convention_variance_ratio(rng):
    """§2.2: var(substitution) / var(reliability) == n_b / n_eff, elementwise."""
    X = rng.standard_normal((N_FEAT, N_BLOCKS))
    w = rng.uniform(0.1, 1.0, N_BLOCKS)
    n_eff = 1.0 / np.sum((w / w.sum()) ** 2)
    T = _uniform_T(X)
    TWs = T @ VarSSIRef._block_weight_factor(w, N_BLOCKS, 'substitution')
    TWr = T @ VarSSIRef._block_weight_factor(w, N_BLOCKS, 'reliability')
    var_s = np.einsum('ij,ij->i', TWs, TWs)
    var_r = np.einsum('ij,ij->i', TWr, TWr)
    mask = var_r > 1e-12
    assert np.allclose(var_s[mask] / var_r[mask], N_BLOCKS / n_eff, rtol=1e-10)


def test_reliability_raises_on_single_block_mass():
    w = np.zeros(N_BLOCKS)
    w[1] = 1.0
    with pytest.raises(ValueError):
        VarSSIRef._block_weight_factor(w, N_BLOCKS, 'reliability')


def test_substitution_zeroes_on_single_block_mass(caplog):
    w = np.zeros(N_BLOCKS)
    w[1] = 1.0
    with caplog.at_level('WARNING', logger='pyOMA.core.VarSSIRef'):
        W = VarSSIRef._block_weight_factor(w, N_BLOCKS, 'substitution')
    assert np.all(W == 0.0)
    assert any('n_eff' in rec.message for rec in caplog.records)


def test_single_block_mass_gives_zero_covariance_all_conventions(rng):
    """Even without the guard, collapsed mass yields a zero factor (no NaN)."""
    F = rng.standard_normal((7, N_BLOCKS))
    w = np.zeros(N_BLOCKS)
    w[3] = 1.0
    for convention in ('substitution', 'precision'):  # reliability raises
        W = VarSSIRef._block_weight_factor(w, N_BLOCKS, convention)
        assert np.all(np.isfinite(W))
        assert np.allclose(F @ W, 0.0)


def test_invalid_convention_raises():
    with pytest.raises(ValueError):
        VarSSIRef._block_weight_factor(None, N_BLOCKS, 'bogus')


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        VarSSIRef._block_weight_factor([1.0, 1.0, 1.0], N_BLOCKS)


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        VarSSIRef._block_weight_factor([-0.1, 0.5, 0.2, 0.2, 0.2], N_BLOCKS)
