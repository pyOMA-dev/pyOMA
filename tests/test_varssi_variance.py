# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-validation of the VarSSIRef variance algorithms against each other
and against the Doehler & Mevel (2013, MSSP 38) reference formulation.

These tests pin three fixes (2026-07):

* ``_compute_slow_sigma_r_s3`` now builds ``sigma_R`` from the correlation
  lags 1..p+q in the ``S3``-compatible vec ordering and with pyOMA's
  ``1/(n_b^2 (n_b-1))`` normalization, so ``S3 @ sigma_R @ S3.T == T @ T.T``
  exactly (paper Eq. 17-19).
* ``_fast_joht_per_order`` builds ``B_i,1`` per paper Eq. (29); previously a
  misplaced bracket moved the correction term onto the wrong block, making
  all fast-path standard deviations wrong by large factors.
* ``_run_modal_order_loop`` passes the eigenvectors to ``remove_conjugates``
  in the signature order ``(eigval, eigvec_r, eigvec_l)``; previously the
  left/right eigenvectors came back swapped, corrupting the mode shapes
  (``C @ chi`` instead of ``C @ phi``) and all variance Jacobians.

With all three in place, the slow (naive, Lemma 1-3) and fast (Prop. 7 /
Eq. 38) algorithms must agree to numerical precision for every method
combination -- they are algebraically identical propagations of the same
``sigma_H``.
"""

import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.VarSSIRef import VarSSIRef, vectorize

NBC = 15          # num_block_columns
NUM_BLOCKS = 6
MAX_ORDER = 8


@pytest.fixture(scope='module')
def ps_var():
    """Fresh, module-local signals: the session-scoped conftest fixture can be
    mutated by other test modules (recomputed Welch segments), which changes
    the data conditioning these precision comparisons rely on."""
    from tests.system_ambient_ifrf import ambient_ifrf
    _, sig = ambient_ifrf(8192, 6, [5], 128, 10, seed=42, num_modes=2)
    ps = PreProcessSignals(sig, 128, ref_channels=[5])
    ps.corr_welch(m_lags=200)
    return ps


def _assert_std_close(a, b, rtol=1e-6, atol_scale=1e-9):
    """Assert two std arrays agree, with an absolute floor scaled to the array
    magnitude (entries that are mathematically zero carry float noise)."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b))
    atol = atol_scale * scale if scale > 0 else atol_scale
    dev = np.abs(a - b)
    margin = dev - (atol + rtol * np.abs(b))
    worst = np.argmax(margin)
    assert np.all(margin <= 0), (
        f'max deviation {dev.flat[worst]:.3e} at flat index {worst}: '
        f'a={a.flat[worst]:.6e} b={b.flat[worst]:.6e} (scale {scale:.3e})')


def _prepare(prep_signals, lsq_method='pinv', subspace_method='covariance',
             variance_algo='fast'):
    obj = VarSSIRef(prep_signals)
    obj.build_subspace_mat(
        num_block_columns=NBC, num_blocks=NUM_BLOCKS,
        subspace_method=subspace_method)
    obj.compute_state_matrices(max_model_order=MAX_ORDER, lsq_method=lsq_method)
    obj.prepare_sensitivities(variance_algo=variance_algo)
    obj.compute_modal_params()
    return obj


@pytest.mark.parametrize('lsq_method', ['pinv', 'qr'])
@pytest.mark.parametrize('subspace_method', ['covariance', 'projection'])
def test_slow_matches_fast(ps_var, subspace_method, lsq_method):
    """Slow (naive) and fast variance algorithms agree to numerical precision."""
    fast = _prepare(ps_var, lsq_method, subspace_method, 'fast')
    slow = _prepare(ps_var, lsq_method, subspace_method, 'slow')

    assert np.allclose(slow.modal_frequencies, fast.modal_frequencies)
    assert np.allclose(slow.modal_damping, fast.modal_damping)
    assert np.allclose(slow.mode_shapes, fast.mode_shapes)
    _assert_std_close(slow.std_frequencies, fast.std_frequencies)
    _assert_std_close(slow.std_damping, fast.std_damping)
    _assert_std_close(slow.std_mode_shapes.real, fast.std_mode_shapes.real)
    _assert_std_close(slow.std_mode_shapes.imag, fast.std_mode_shapes.imag)


def test_sigma_r_reconstructs_hankel_covariance(ps_var):
    """Paper Eq. (17)/(18): S3 @ vec(R_j) rebuilds vec(H_j) and
    S3 @ sigma_R @ S3.T equals the fast path's T @ T.T exactly."""
    fast = _prepare(ps_var, 'pinv', 'covariance', 'fast')
    slow = _prepare(ps_var, 'pinv', 'covariance', 'slow')
    T = fast.hankel_cov_matrix
    S3 = slow.S3
    ps = ps_var
    n_l = ps.num_analised_channels
    n_r = ps.num_ref_channels
    n_lags = slow.num_block_rows + slow.num_block_columns

    for j in range(NUM_BLOCKS):
        R = np.transpose(ps.corr_matrices[j][:, :, 1:n_lags + 1],
                         (2, 0, 1)).reshape(n_lags * n_l, n_r)
        assert np.allclose(S3.dot(vectorize(R) * NUM_BLOCKS),
                           vectorize(slow.subspace_matrices[j]),
                           rtol=1e-12, atol=1e-14)

    lhs = np.asarray(S3.dot(slow.sigma_R).dot(S3.T.toarray()))
    rhs = T @ T.T
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-16)


def test_fast_pinv_matches_fast_qr(ps_var):
    """The pinv- and qr-based least-squares sensitivities are algebraically
    identical; disagreement indicates broken eigenvector pairing."""
    pinv = _prepare(ps_var, 'pinv', 'covariance', 'fast')
    qr = _prepare(ps_var, 'qr', 'covariance', 'fast')
    assert np.allclose(pinv.std_frequencies, qr.std_frequencies,
                       rtol=1e-6, atol=1e-12)
    assert np.allclose(pinv.std_damping, qr.std_damping,
                       rtol=1e-6, atol=1e-12)


def test_mode_shapes_match_analytic_rod(ps_var):
    """Identified mode shapes are C @ phi (right eigenvector): they must match
    the analytic rod shapes closely.  With swapped left/right eigenvectors the
    MAC degrades to ~0.94-0.97."""
    obj = _prepare(ps_var, 'pinv', 'covariance', 'fast')

    length = 200.0
    x = np.linspace(0, length, 6)
    j = np.arange(1, 3)
    true_ms = np.sin((2 * j[np.newaxis, :] - 1) / 2 * np.pi / length
                     * x[:, np.newaxis])
    true_f = np.array([6.465, 19.396])

    order = 4
    checked = 0
    for i in range(order):
        f = obj.modal_frequencies[order, i]
        if f <= 0:
            continue
        k = np.argmin(np.abs(true_f - f))
        if abs(f - true_f[k]) / true_f[k] > 0.05:
            continue
        ms = obj.mode_shapes[:, i, order]
        ref = true_ms[:, k]
        mac = (np.abs(ms.conj() @ ref) ** 2 /
               ((ms.conj() @ ms).real * (ref @ ref)))
        assert mac > 0.98, f'mode at {f:.3f} Hz: MAC {mac:.4f}'
        checked += 1
    assert checked == 2
