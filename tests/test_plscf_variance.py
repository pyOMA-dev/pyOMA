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


@pytest.fixture(scope='module')
def fd_plscf_mixed():
    """Mixed channel types, which the default all-accelerometer setup cannot test.

    ``integrate_quantities`` scales a channel by ``omega**-p``, p = 2/1/0 for
    accelerations/velocities/displacements.  When every channel has the same p
    the factor is common and cancels against the mode-shape normalisation, so
    its dependence on omega is unobservable; only a mixed setup exercises it.
    """
    from tests.system_ambient_ifrf import ambient_ifrf
    _, sig = ambient_ifrf(8192, 6, [5], FS, 10, seed=42, num_modes=2)
    prep = PreProcessSignals(sig, FS, ref_channels=[4, 5], accel_channels=[0, 1],
                             velo_channels=[2, 3], disp_channels=[4, 5])
    obj = VarPLSCF(prep)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    return obj


def _max_rel_err(predicted, finite_difference):
    """Max-norm relative error.

    An entrywise rtol is meaningless here: several scores are exactly zero by
    construction (e.g. the normalised participation component).
    """
    scale = max(np.max(np.abs(finite_difference)), 1e-30)
    return np.max(np.abs(predicted - finite_difference)) / scale


def _max_rel_err_per_channel(predicted, finite_difference):
    """Max-norm relative error, each channel scaled by its own magnitude.

    With mixed channel types the integration factors span orders of magnitude
    (1/omega**2 against 1), so a global max-norm only ever sees the displacement
    channels -- exactly the ones the integration leaves untouched.  Scaling per
    channel keeps the suppressed channels observable.
    """
    worst = 0.0
    overall = np.max(np.abs(finite_difference))
    for channel in range(finite_difference.shape[0]):
        scale = np.max(np.abs(finite_difference[channel,:]))
        if scale < 1e-10 * overall:
            continue  # the normalised component, which is exactly zero
        worst = max(worst, np.max(
            np.abs(predicted[channel,:] - finite_difference[channel,:])) / scale)
    return worst


def _mode_shapes_at(obj, order, half_spectra):
    """Re-identify from perturbed half-spectra and return the mode shapes."""
    obj.pos_half_spectra = half_spectra
    ctx = obj._assemble_normal_equations(order)
    return obj.modal_analysis_residuals(ctx.alpha, ctx.beta_l_i)[2]


def _predicted_mode_shape_scores(obj, order):
    """The Stage-2 linearisation at the unperturbed half-spectra."""
    ctx = obj._assemble_normal_equations(order)
    _, _, _, eigenvalues = obj.modal_analysis_residuals(ctx.alpha, ctx.beta_l_i)
    scores = obj._stage1_scores(ctx, obj._modal_ctx, eigenvalues, obj.spec_block_factor)
    return obj._mode_shape_scores(obj._lsfd_ctx, scores, obj.spec_block_factor)


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
            rhs_predicted = np.einsum('pq,qsj->psj', ctx.M_aa,
                                      obj._theta_scores(ctx, obj.spec_block_factor))

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
        scores = obj._stage1_scores(ctx, obj._modal_ctx, eigenvalues, obj.spec_block_factor)

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
        scores = obj._stage1_scores(ctx, obj._modal_ctx, eigenvalues, obj.spec_block_factor)

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


def test_mode_shape_scores_match_central_finite_differences(fd_plscf):
    """The Stage-2 chain against a finite difference of the actual estimator.

    Guards the T1 (direct, through h) and T2 (through A) paths jointly,
    including the Stage-1 pole/participation scores coupling into A.
    """
    obj = fd_plscf
    half_spectra_0 = obj.pos_half_spectra.copy()
    factor = obj.spec_block_factor
    try:
        predicted = _predicted_mode_shape_scores(obj, FD_ORDER)
        for i_b in (0, 5, 11, 19):
            deviation = FD_EPS * factor[..., i_b]
            finite_difference = (
                _mode_shapes_at(obj, FD_ORDER, half_spectra_0 + deviation)
                - _mode_shapes_at(obj, FD_ORDER, half_spectra_0 - deviation)
            ) / (2 * FD_EPS)
            err = _max_rel_err(predicted[:,:, i_b], finite_difference)
            assert err < 1e-5, f'block {i_b}: max rel err {err:.3e}'
    finally:
        obj.pos_half_spectra = half_spectra_0


def test_mode_shape_scores_track_the_integration_of_mixed_channel_types(fd_plscf_mixed):
    """The mode shapes are integrated at each mode's own omega, which varies.

    With uniform channel types this is untestable (see fd_plscf_mixed); freezing
    the integration factors passes there while being wrong by ~10% here.
    """
    obj = fd_plscf_mixed
    half_spectra_0 = obj.pos_half_spectra.copy()
    factor = obj.spec_block_factor
    try:
        predicted = _predicted_mode_shape_scores(obj, FD_ORDER)
        for i_b in (0, 11):
            deviation = FD_EPS * factor[..., i_b]
            finite_difference = (
                _mode_shapes_at(obj, FD_ORDER, half_spectra_0 + deviation)
                - _mode_shapes_at(obj, FD_ORDER, half_spectra_0 - deviation)
            ) / (2 * FD_EPS)
            err = _max_rel_err_per_channel(predicted[:,:, i_b], finite_difference)
            assert err < 1e-5, f'block {i_b}: max per-channel rel err {err:.3e}'
    finally:
        obj.pos_half_spectra = half_spectra_0


def test_std_mode_shapes_has_the_shape_of_the_point_estimate(var_plscf_computed):
    obj = var_plscf_computed
    assert obj.std_mode_shapes.shape == obj.mode_shapes.shape
    assert np.iscomplexobj(obj.std_mode_shapes)


def test_std_mode_shapes_are_finite_and_positive(var_plscf_computed):
    """Informative channels get a positive std; silent ones get exactly zero.

    Channel 0 of this system is the grounded degree of freedom and carries no
    signal, so its block deviations -- and every linear functional of them --
    are identically zero.
    """
    obj = var_plscf_computed
    assert (obj.modal_frequencies > 0).any(), 'no modes identified, test would be vacuous'
    assert np.all(np.isfinite(obj.std_mode_shapes))

    silent = np.max(np.abs(obj.spec_block_factor), axis=(1, 2, 3)) == 0
    assert silent.any() and not silent.all(), 'fixture no longer discriminates'

    checked = 0
    for order in range(1, MAX_ORDER):
        for i in range(int(np.sum(obj.modal_frequencies[order,:] > 0))):
            std = np.abs(obj.std_mode_shapes[:, i, order])
            assert np.all(std[silent] == 0)
            informative = ~silent
            # the largest component is normalised to one, so it cannot vary
            informative[np.argmax(np.abs(obj.mode_shapes[:, i, order]))] = False
            assert np.all(std[informative] > 0)
            checked += 1
    assert checked > 0, 'no modes identified, test would be vacuous'


def test_the_normalised_mode_shape_component_carries_no_uncertainty(var_plscf_computed):
    """The largest component is scaled to one, so it cannot vary."""
    obj = var_plscf_computed
    checked = 0
    for order in range(1, MAX_ORDER):
        n_modes = int(np.sum(obj.modal_frequencies[order,:] > 0))
        for i in range(n_modes):
            std = obj.std_mode_shapes[:, i, order]
            k = np.argmax(np.abs(obj.mode_shapes[:, i, order]))
            assert np.abs(std[k]) < 1e-12 * np.max(np.abs(std))
            checked += 1
    assert checked > 0, 'no modes identified, test would be vacuous'


def test_save_load_round_trip_preserves_the_standard_deviations(var_plscf_computed, tmp_path):
    obj = var_plscf_computed
    fname = str(tmp_path / 'varplscf.npz')
    obj.save_state(fname)
    loaded = VarPLSCF.load_state(fname, obj.prep_signals)

    for name in ('modal_frequencies', 'modal_damping', 'mode_shapes',
                 'participation_vectors', 'std_frequencies', 'std_damping',
                 'std_participation_vectors', 'std_mode_shapes'):
        original, restored = getattr(obj, name), getattr(loaded, name)
        assert restored is not None, f'{name} was not restored'
        np.testing.assert_allclose(restored, original, err_msg=name)


def test_loading_a_plscf_archive_leaves_the_standard_deviations_empty(
        var_plscf_computed, tmp_path):
    """An archive written by PLSCF carries no variances; loading must not fail."""
    from pyOMA.core.PLSCF import PLSCF

    obj = var_plscf_computed
    plain = PLSCF(obj.prep_signals)
    plain.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    plain.compute_modal_params(MAX_ORDER, modal_contrib=False)
    fname = str(tmp_path / 'plscf_legacy.npz')
    plain.save_state(fname)

    loaded = VarPLSCF.load_state(fname, obj.prep_signals)
    assert loaded.std_frequencies is None
    assert loaded.std_damping is None
    assert loaded.std_participation_vectors is None
    assert loaded.std_mode_shapes is None
    np.testing.assert_allclose(loaded.modal_frequencies, plain.modal_frequencies)


def test_a_loaded_state_refuses_to_recompute_without_the_block_factor(
        var_plscf_computed, tmp_path):
    """The block factor is deliberately not archived; the error must say so."""
    obj = var_plscf_computed
    fname = str(tmp_path / 'varplscf.npz')
    obj.save_state(fname)
    loaded = VarPLSCF.load_state(fname, obj.prep_signals)

    assert loaded.spec_block_factor is None
    with pytest.raises(RuntimeError, match='build_half_spectra'):
        loaded.compute_modal_params(MAX_ORDER, modal_contrib=False)


# ── Monte-Carlo acceptance gate ───────────────────────────────────────────────
#
# The finite-difference tests above pin every Jacobian to the true derivative of
# the estimator, but they are blind to the question of whether the *entry*
# covariance model is right -- whether the scatter of the half-spectra across
# blocks predicts their scatter across realisations.  Only Monte-Carlo answers
# that, so this gate is not redundant with them.
#
# It needs an excitation whose ensemble statistics match what covariance
# propagation assumes, which ``ambient_ifrf`` does not provide (see
# ``ambient_gaussian``).  Order 2 identifies exactly the system's two modes and
# is the well-conditioned end of the pLSCF normal equations.

MC_ORDER = 2
MC_REALIZATIONS = 150
# Measured ratios here span 0.91 to 1.11 (f and zeta, both modes).  Two effects
# set that spread and neither is a defect of the variance chain:
#   - an MC standard deviation over n samples has a relative standard error of
#     1/sqrt(2(n-1)), ~5.8% at this n;
#   - the SEM treats the blocks as independent, but they are adjacent segments of
#     one record.  The modal correlation time is ~1/(zeta*omega) ~ 62 samples
#     against a block length of 8192/20 = 410, so a residual block-to-block
#     correlation of order 15% is expected.
# The band is therefore deliberately loose.  It exists to catch a wrong entry
# covariance model, which the finite-difference tests cannot see at all and which
# fails loudly (the multisine ensemble lands at 0.15-0.39) -- not to re-pin the
# Jacobians, which those tests already hold to 1e-5.  Mutation-checked: it still
# catches any mis-scaling of the predicted std of 15% or more.
MC_BAND = (0.8, 1.3)


def _mc_realization(seed):
    """Identify one independent realization; return point estimates and stds."""
    from tests.system_ambient_ifrf import ambient_gaussian
    _, sig = ambient_gaussian(8192, 6, [5], FS, 10, seed=seed, num_modes=2)
    prep = PreProcessSignals(sig, FS, ref_channels=[4, 5])
    obj = VarPLSCF(prep)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    obj.compute_modal_params(MC_ORDER + 1, modal_contrib=False)
    if int(np.sum(obj.modal_frequencies[MC_ORDER,:] > 0)) != 2:
        return None
    return {
        'f': obj.modal_frequencies[MC_ORDER, :2].copy(),
        'zeta': obj.modal_damping[MC_ORDER, :2].copy(),
        'phi': obj.mode_shapes[:, :2, MC_ORDER].copy(),
        'L': obj.participation_vectors[:, :2, MC_ORDER].copy(),
        'std_f': obj.std_frequencies[MC_ORDER, :2].copy(),
        'std_zeta': obj.std_damping[MC_ORDER, :2].copy(),
        'std_phi': obj.std_mode_shapes[:, :2, MC_ORDER].copy(),
        'std_L': obj.std_participation_vectors[:, :2, MC_ORDER].copy(),
    }


@pytest.fixture(scope='module')
def monte_carlo():
    """Independent realizations of the same system, at ~90 ms each.

    Seeded, so the gate is deterministic rather than merely probable.
    """
    runs = [_mc_realization(5000 + i) for i in range(MC_REALIZATIONS)]
    runs = [r for r in runs if r is not None]
    assert len(runs) > 0.9 * MC_REALIZATIONS, 'identification failed too often'
    return {key: np.array([r[key] for r in runs]) for key in runs[0]}


def _mc_ratio(values, predicted):
    """Sample std across realizations over the mean predicted std."""
    return np.std(values, axis=0, ddof=1) / np.mean(predicted, axis=0)


@pytest.mark.slow
@pytest.mark.parametrize('quantity', ['f', 'zeta'])
def test_predicted_std_matches_monte_carlo_scatter_of_the_poles(monte_carlo, quantity):
    """The delta-method std must reproduce the realization scatter of f and zeta."""
    ratio = _mc_ratio(monte_carlo[quantity], monte_carlo[f'std_{quantity}'])
    assert np.all(ratio > MC_BAND[0]) and np.all(ratio < MC_BAND[1]), \
        f'{quantity}: MC/predicted std ratio {ratio} outside {MC_BAND}'


@pytest.mark.slow
@pytest.mark.parametrize('quantity', ['phi', 'L'])
def test_predicted_std_matches_monte_carlo_scatter_of_the_vectors(monte_carlo, quantity):
    """Same for the mode shapes and participation vectors, component-wise.

    Components whose predicted std is zero by construction -- the normalised one,
    and the grounded channel that carries no signal -- are excluded: they have no
    ratio to form, and their exactly-zero std is asserted elsewhere.
    """
    values = monte_carlo[quantity]
    predicted = monte_carlo[f'std_{quantity}']
    checked = 0
    for part in ('real', 'imag'):
        mc = np.std(getattr(values, part), axis=0, ddof=1)
        pred = np.mean(getattr(predicted, part), axis=0)
        informative = pred > 1e-12 * np.max(pred)
        ratio = mc[informative] / pred[informative]
        assert np.all(ratio > MC_BAND[0]) and np.all(ratio < MC_BAND[1]), \
            f'{quantity}.{part}: MC/predicted std ratio {ratio} outside {MC_BAND}'
        checked += int(np.sum(informative))
    assert checked > 0, 'no informative components, test would be vacuous'


@pytest.mark.slow
def test_the_multisine_ensemble_would_not_validate_anything(monte_carlo):
    """Guard the choice of excitation, which is easy to "simplify" back.

    ``ambient_ifrf`` pins the excitation magnitude and randomises only the phase,
    which makes the record-length spectrum estimate nearly deterministic.  Its
    realisations then scatter far less than any block-based covariance predicts,
    so swapping it back in here would make the gate above silently meaningless
    rather than merely fail.  Asserted at the half-spectra, upstream of every
    Jacobian, to keep the diagnosis unambiguous.
    """
    from tests.system_ambient_ifrf import ambient_ifrf

    estimates, predictions = [], []
    for seed in range(6000, 6012):
        _, sig = ambient_ifrf(8192, 6, [5], FS, 10, seed=seed, num_modes=2)
        obj = VarPLSCF(PreProcessSignals(sig, FS, ref_channels=[4, 5]))
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        estimates.append(obj.pos_half_spectra.copy())
        predictions.append(np.sum(np.abs(obj.spec_block_factor) ** 2, axis=-1))

    estimates = np.array(estimates)
    mean = np.abs(np.mean(estimates, axis=0))
    strong = mean > 0.05 * np.max(mean)
    mc_var = (np.var(estimates.real, axis=0, ddof=1)
              + np.var(estimates.imag, axis=0, ddof=1))
    ratio = np.sqrt(mc_var[strong] / np.mean(predictions, axis=0)[strong])
    assert np.median(ratio) < 0.5, (
        'the multisine ensemble now scatters like the block-based prediction; '
        'if that is genuine, ambient_gaussian is no longer needed')


# ── StabilDiagram integration ─────────────────────────────────────────────────

def test_stabildiagram_picks_up_the_pLSCF_standard_deviations(var_plscf_computed):
    """StabilCalc detects variances by attribute name, so VarPLSCF gets them free.

    This is a read-only check on StabilDiagram: the point is that matching
    VarSSIRef's attribute names and shapes is sufficient, with no changes there.
    """
    from pyOMA.core.StabilDiagram import StabilCalc

    stabil_calc = StabilCalc(var_plscf_computed)
    assert stabil_calc.capabilities['std']
    assert stabil_calc.capabilities['msh']

    stabil_calc.calculate_stabilization_masks()
    assert stabil_calc.masks['mask_pre'] is not None


def test_stabildiagram_exports_the_pLSCF_standard_deviations(var_plscf_computed, tmp_path):
    """The exported std columns must carry the values VarPLSCF identified.

    StabilCalc indexes ``std_frequencies`` as (order, mode) and
    ``std_mode_shapes`` as (channel, mode, order), so comparing the exported
    values pins that layout agreement, not just the plumbing.
    """
    from pyOMA.core.StabilDiagram import StabilCalc

    obj = var_plscf_computed
    stabil_calc = StabilCalc(obj)
    stabil_calc.calculate_stabilization_masks()

    order = MAX_ORDER - 1
    num_modes = int(np.sum(obj.modal_frequencies[order,:] > 0))
    assert num_modes > 0, 'no modes identified, test would be vacuous'
    for i in range(num_modes):
        stabil_calc.add_mode((order, i))

    fname = str(tmp_path / 'results.npz')
    stabil_calc.export_results(fname, binary=True)
    exported = np.load(fname)

    # both the export and compute_modal_params order modes by frequency
    assert np.allclose(exported['selected_stdf'],
                       obj.std_frequencies[order, :num_modes])
    assert np.allclose(exported['selected_stdd'],
                       obj.std_damping[order, :num_modes])
    assert np.allclose(exported['selected_stdmsh'],
                       obj.std_mode_shapes[:, :num_modes, order])

    # the text export formats the std arrays on a separate code path
    fname_txt = str(tmp_path / 'results.txt')
    stabil_calc.export_results(fname_txt, binary=False)
    with open(fname_txt, encoding='utf-8') as export_file:
        text = export_file.read()
    assert 'Standard deviations of the Frequencies' in text
    assert 'Standard Deviations of the Mode shapes' in text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
