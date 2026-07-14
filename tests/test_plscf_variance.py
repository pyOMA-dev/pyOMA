"""Variance estimation tests for the pLSCF method (VarPLSCF)."""
import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.VarPLSCF import VarPLSCF

FS = 128
M_LAGS = 200
NUM_BLOCKS = 20
MAX_ORDER = 6

# cond(M_aa) is ~3e3 here but reaches 1e9 by order 4 and 1e11 by order 20: the
# normal-equation formulation squares the condition number, which sets the floor
# for any finite difference of the solved coefficients.
FD_ORDER = 2
# minimum of the truncation (~eps^2) / roundoff (~1/eps) trade-off at FD_ORDER
FD_EPS = 0.01


@pytest.fixture(scope='module')
def prep_signals_blocks():
    """Module-scoped signals, hermetic: other test modules mutate shared fixtures."""
    from tests.system_ambient_ifrf import ambient_ifrf
    _, sig = ambient_ifrf(8192, 6, [5], FS, 10, seed=42, num_modes=2)
    return PreProcessSignals(sig, FS, ref_channels=[4, 5])


@pytest.fixture(scope='module')
def var_plscf(prep_signals_blocks):
    obj = VarPLSCF(prep_signals_blocks)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    return obj


@pytest.fixture(scope='module')
def fd_plscf(prep_signals_blocks):
    """Own instance: the finite-difference tests overwrite pos_half_spectra."""
    obj = VarPLSCF(prep_signals_blocks)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    return obj


@pytest.fixture(scope='module')
def var_plscf_computed(prep_signals_blocks):
    obj = VarPLSCF(prep_signals_blocks)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
    return obj


def _max_rel_err(predicted, finite_difference):
    """Max-norm relative error.

    An entrywise rtol is meaningless here: several scores are exactly zero by
    construction (e.g. the normalised participation component).
    """
    scale = max(np.max(np.abs(finite_difference)), 1e-30)
    return np.max(np.abs(predicted - finite_difference)) / scale


def _m_theta(obj, order, alpha_0, half_spectra):
    """M(h) @ alpha, with alpha frozen: the quantity whose derivative is dM.theta."""
    obj.pos_half_spectra = half_spectra
    return obj._assemble_normal_equations(order).M @ alpha_0


def _stage1_quantities(obj, order, half_spectra):
    """Re-identify from perturbed half-spectra and return the Stage-1 outputs."""
    n_r = obj.prep_signals.num_ref_channels
    obj.pos_half_spectra = half_spectra
    ctx = obj._assemble_normal_equations(order)
    frequencies, damping, _, eigenvalues = obj.modal_analysis_residuals(
        ctx.alpha, ctx.beta_l_i)
    return {'theta': ctx.alpha[:order * n_r,:].copy(),
            'lambda': eigenvalues,
            'freq': frequencies,
            'damp': damping,
            'part': obj._participation_vectors.copy()}


def test_build_half_spectra_requires_num_blocks(prep_signals_blocks):
    obj = VarPLSCF(prep_signals_blocks)
    with pytest.raises(ValueError, match='num_blocks'):
        obj.build_half_spectra(nperseg=M_LAGS)


def test_block_factor_shape_and_block_axis_last(var_plscf):
    n_l = var_plscf.prep_signals.num_analised_channels
    n_r = var_plscf.prep_signals.num_ref_channels
    assert var_plscf.spec_block_factor.shape == (
        n_l, n_r, var_plscf.num_omega, NUM_BLOCKS)

    # guard the tests below, several of which a zero factor would satisfy trivially
    scale = np.abs(var_plscf.pos_half_spectra).mean()
    assert np.abs(var_plscf.spec_block_factor).mean() > 1e-4 * scale


def test_block_deviations_are_centred(var_plscf):
    """Deviations are taken about the block mean, so they must sum to zero."""
    total = var_plscf.spec_block_factor.sum(axis=-1)
    scale = np.abs(var_plscf.pos_half_spectra).mean()
    np.testing.assert_allclose(total, 0.0, atol=1e-10 * scale)


def test_mean_of_block_spectra_is_the_identified_half_spectrum(var_plscf):
    """The windowed rFFT is linear, so averaging blocks in the correlation domain
    and in the spectral domain must agree exactly.

    The covariance factor describes the scatter about this mean, so if the two
    disagreed, the factor would describe a different estimator than the one fitted.
    """
    obj = var_plscf
    corr_blocks = obj.prep_signals.corr_matrices_bt[obj.training_blocks, ..., :obj.nperseg]
    _, block_spectra, _ = obj._windowed_half_spectrum(
        corr_blocks, obj.nperseg, obj.window_decay,
        obj.begin_frequency, obj.end_frequency)

    np.testing.assert_allclose(
        np.mean(block_spectra, axis=0), obj.pos_half_spectra, rtol=1e-10, atol=1e-12)


def test_factor_reproduces_the_sample_covariance_of_the_estimate(var_plscf):
    """cov(h_of) must equal B_of @ B_of.T for the re/im-stacked slice.

    Checked against the sample covariance of the block mean computed directly,
    i.e. the sample covariance of the blocks divided by n_b.
    """
    obj = var_plscf
    i_l, i_r, i_f = 2, 1, 10

    factor = obj.spec_block_factor[i_l, i_r, i_f, :]
    B = np.vstack([np.real(factor), np.imag(factor)])
    cov_factor = B @ B.T

    corr_blocks = obj.prep_signals.corr_matrices_bt[obj.training_blocks, ..., :obj.nperseg]
    _, block_spectra, _ = obj._windowed_half_spectrum(
        corr_blocks, obj.nperseg, obj.window_decay,
        obj.begin_frequency, obj.end_frequency)
    samples = block_spectra[:, i_l, i_r, i_f]
    stacked = np.vstack([np.real(samples), np.imag(samples)])
    # ddof=1 sample covariance of a block, divided by n_b -> covariance of the mean
    cov_direct = np.cov(stacked, ddof=1) / NUM_BLOCKS

    np.testing.assert_allclose(cov_factor, cov_direct, rtol=1e-10, atol=1e-20)


def test_covariance_of_any_contraction_is_positive_semidefinite(var_plscf):
    """B B^T is PSD by construction; verify no scaling step broke that."""
    obj = var_plscf
    for i_f in (0, obj.num_omega // 2, obj.num_omega - 1):
        factor = obj.spec_block_factor[1, 0, i_f, :]
        B = np.vstack([np.real(factor), np.imag(factor)])
        eigvals = np.linalg.eigvalsh(B @ B.T)
        assert eigvals.min() >= -1e-20, f'negative eigenvalue {eigvals.min()} at line {i_f}'


def test_training_blocks_restrict_the_factor(prep_signals_blocks):
    """Only the blocks that entered the point estimate may enter its covariance."""
    training = [0, 1, 2, 3, 4, 5, 6, 7]
    obj = VarPLSCF(prep_signals_blocks)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS, training_blocks=training)

    assert obj.spec_block_factor.shape[-1] == len(training)

    corr_blocks = obj.prep_signals.corr_matrices_bt[np.array(training), ..., :obj.nperseg]
    _, block_spectra, _ = obj._windowed_half_spectrum(
        corr_blocks, obj.nperseg, obj.window_decay,
        obj.begin_frequency, obj.end_frequency)
    np.testing.assert_allclose(
        np.mean(block_spectra, axis=0), obj.pos_half_spectra, rtol=1e-10, atol=1e-12)


def test_theta_score_matches_the_exact_derivative_of_the_normal_equations(fd_plscf):
    """M is exactly quadratic in the half-spectra, so a central difference of
    M(h) @ theta is its derivative *exactly*, at any step size and with no
    truncation error.

    That makes this the sharp test of the dM.theta assembly, which is where the
    sign, conjugation and Kronecker-structure bookkeeping lives.  The full chain
    cannot be pinned this tightly (see the end-to-end test below), so this one
    carries the weight.
    """
    obj = fd_plscf
    n_r = obj.prep_signals.num_ref_channels
    half_spectra = obj.pos_half_spectra
    try:
        for order in (2, 4, 8):
            ctx = obj._assemble_normal_equations(order)
            alpha_0 = ctx.alpha.copy()
            # -[dM theta]_{:n*n_r}, recovered from the score by multiplying M_aa
            # back on; that direction is stable, unlike solving with it
            rhs_predicted = np.einsum('pq,qsj->psj', ctx.M_aa, obj._theta_scores(ctx))

            for i_b in (0, 5, 11, NUM_BLOCKS - 1):
                deviation = obj.spec_block_factor[:,:,:, i_b]
                # step size 1: exact for a quadratic, and large enough that the
                # difference does not cancel itself away
                fd = -(_m_theta(obj, order, alpha_0, half_spectra + deviation)
                       - _m_theta(obj, order, alpha_0, half_spectra - deviation)) / 2
                obj.pos_half_spectra = half_spectra
                err = _max_rel_err(rhs_predicted[:,:, i_b], fd[:order * n_r,:])
                assert err < 1e-9, f'order {order}, block {i_b}: rel. error {err:.3e}'
    finally:
        obj.pos_half_spectra = half_spectra


def test_stage1_scores_match_central_finite_differences(fd_plscf):
    """Every Stage-1 score against a central difference of the actual estimator.

    Run at a low order, where cond(M_aa) is still moderate; the accuracy of the
    finite difference itself, not of the propagation, is what limits the
    tolerance here.
    """
    obj = fd_plscf
    half_spectra = obj.pos_half_spectra
    try:
        obj.pos_half_spectra = half_spectra
        ctx = obj._assemble_normal_equations(FD_ORDER)
        _, _, _, eigenvalues = obj.modal_analysis_residuals(ctx.alpha, ctx.beta_l_i)
        scores = obj._stage1_scores(ctx, obj._modal_ctx, eigenvalues)

        for i_b in (0, 5, 11, NUM_BLOCKS - 1):
            deviation = FD_EPS * obj.spec_block_factor[:,:,:, i_b]
            plus = _stage1_quantities(obj, FD_ORDER, half_spectra + deviation)
            minus = _stage1_quantities(obj, FD_ORDER, half_spectra - deviation)

            for name, predicted in (
                    ('theta', scores.U_theta[:,:, i_b]),
                    ('lambda', scores.d_lambda[:, i_b]),
                    ('freq', scores.d_frequency[:, i_b]),
                    ('damp', scores.d_damping[:, i_b]),
                    ('part', scores.d_participation[:,:, i_b])):
                fd = (plus[name] - minus[name]) / (2 * FD_EPS)
                err = _max_rel_err(predicted, fd)
                assert err < 1e-5, f'{name} score, block {i_b}: rel. error {err:.3e}'
    finally:
        obj.pos_half_spectra = half_spectra


def test_scores_stay_paired_with_the_modes_they_belong_to(fd_plscf):
    """Guard the in-band filter and frequency sort feeding the scores.

    Run at an order where several modes are in band, so that a score attached to
    the wrong mode shows up.  cond(M_aa) is ~1e9 here, which limits the finite
    difference to a few digits -- but a mis-pairing is an O(1) error, so a loose
    bound still catches it.
    """
    obj = fd_plscf
    order = 4
    half_spectra = obj.pos_half_spectra
    try:
        obj.pos_half_spectra = half_spectra
        ctx = obj._assemble_normal_equations(order)
        frequencies, _, _, eigenvalues = obj.modal_analysis_residuals(
            ctx.alpha, ctx.beta_l_i)
        assert len(frequencies) > 2, 'test needs several in-band modes to be meaningful'
        scores = obj._stage1_scores(ctx, obj._modal_ctx, eigenvalues)

        for i_b in (0, 11):
            deviation = FD_EPS * obj.spec_block_factor[:,:,:, i_b]
            plus = _stage1_quantities(obj, order, half_spectra + deviation)
            minus = _stage1_quantities(obj, order, half_spectra - deviation)
            fd = (plus['freq'] - minus['freq']) / (2 * FD_EPS)
            err = _max_rel_err(scores.d_frequency[:, i_b], fd)
            assert err < 1e-2, f'block {i_b}: rel. error {err:.3e}'
    finally:
        obj.pos_half_spectra = half_spectra


def test_compute_modal_params_rejects_complex_coefficients(var_plscf):
    with pytest.raises(NotImplementedError, match='real'):
        var_plscf.compute_modal_params(4, complex_coefficients=True)


def test_compute_modal_params_rejects_the_state_space_algorithm(var_plscf):
    with pytest.raises(NotImplementedError, match='residuals'):
        var_plscf.compute_modal_params(4, algo='state-space')


def test_std_arrays_have_the_shapes_of_their_point_estimates(var_plscf_computed):
    """StabilDiagram consumes std_frequencies/std_damping by name and shape."""
    obj = var_plscf_computed
    assert obj.std_frequencies.shape == obj.modal_frequencies.shape
    assert obj.std_damping.shape == obj.modal_damping.shape
    assert obj.std_participation_vectors.shape == obj.participation_vectors.shape


def test_standard_deviations_are_finite_and_positive(var_plscf_computed):
    obj = var_plscf_computed
    identified = obj.modal_frequencies > 0
    assert identified.any(), 'no modes identified, test would be vacuous'
    assert np.all(np.isfinite(obj.std_frequencies))
    assert np.all(np.isfinite(obj.std_damping))
    assert np.all(np.isfinite(obj.std_participation_vectors))
    assert np.all(obj.std_frequencies[identified] > 0)
    assert np.all(obj.std_damping[identified] > 0)


def test_the_normalised_participation_component_carries_no_uncertainty(var_plscf_computed):
    """The largest component is scaled to one, so it cannot vary.

    The paper notes the same rank deficiency in Sect. 3.1.3.
    """
    obj = var_plscf_computed
    checked = 0
    for order in range(1, MAX_ORDER):
        n_modes = int(np.sum(obj.modal_frequencies[order,:] > 0))
        for i in range(n_modes):
            std = obj.std_participation_vectors[:, i, order]
            k = np.argmax(np.abs(obj.participation_vectors[:, i, order]))
            assert np.abs(std[k]) < 1e-12 * np.max(np.abs(std))
            checked += 1
    assert checked > 0, 'no modes identified, test would be vacuous'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
