"""Block-weight primitives and post-hoc reweighting for the pLSCF method (VarPLSCF).

The weighting matrix is verified against an independent re-derivation of the
reference formula, and then through the properties it is required to have. Those
properties, not the formula, are what the downstream reweighting relies on.

VarPLSCF normalises its block deviations by ``sqrt(n_b * (n_b - 1))`` -- a plain
standard error of the mean -- where VarSSIRef pre-inflates its blocks and
normalises by ``sqrt(n_b**2 * (n_b - 1))``. The weight scalars below therefore
differ from VarSSIRef's while the convention *names* keep their meaning; see
``test_varssiref_scalars_would_not_reduce_at_uniform`` for why the two are not
interchangeable.
"""
import numpy as np
import pytest

from pyOMA.core.PLSCF import PLSCF
from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.VarPLSCF import VarPLSCF

RTOL = 1e-10
ATOL = 1e-13

CONVENTIONS = ('substitution', 'reliability', 'precision')

FS = 128
M_LAGS = 200
NUM_BLOCKS = 20


def reference_scalar(weights, num_blocks, convention):
    """Independent re-derivation of s(w) for VarPLSCF's ``a = 1`` normalisation."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    n_b = float(num_blocks)
    return {
        'substitution': np.sqrt(n_b * (n_b - 1.0) / (n_eff - 1.0)),
        'reliability': np.sqrt(n_eff * (n_b - 1.0) / (n_eff - 1.0)),
        'precision': np.sqrt(n_b),
    }[convention]


def reference_factor(weights, num_blocks, convention):
    """Independent re-derivation of W(w) = s(w) (I - w 1^T) diag(sqrt(w))."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    s_w = reference_scalar(w, num_blocks, convention)
    return s_w * (np.eye(num_blocks) - w[:, np.newaxis]) * np.sqrt(w)[np.newaxis,:]


def block_factor(rng, num_rows, num_blocks):
    """A synthetic centered factor with VarPLSCF's ``a = 1`` normalisation."""
    samples = rng.normal(size=(num_rows, num_blocks))
    deviations = samples - samples.mean(axis=1, keepdims=True)
    return samples, deviations / np.sqrt(num_blocks * (num_blocks - 1))


@pytest.fixture(params=[6, 8, 20])
def num_blocks(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(5)


@pytest.fixture
def weights(rng, num_blocks):
    w = rng.random(num_blocks)
    return w / w.sum()


class TestValidateWeights:

    def test_none_passes_through_with_uniform_n_eff(self, num_blocks):
        weights, n_eff = VarPLSCF._validate_weights(None, num_blocks)
        assert weights is None
        assert n_eff == float(num_blocks)

    def test_renormalised_to_sum_one(self, num_blocks, rng):
        raw = rng.random(num_blocks) * 17.0
        weights, _ = VarPLSCF._validate_weights(raw, num_blocks)
        assert weights.sum() == pytest.approx(1.0, rel=RTOL)
        # the scale of the input carries no meaning
        rescaled, _ = VarPLSCF._validate_weights(raw / 3.0, num_blocks)
        np.testing.assert_allclose(weights, rescaled, rtol=RTOL, atol=ATOL)

    def test_uniform_n_eff_is_num_blocks(self, num_blocks):
        _, n_eff = VarPLSCF._validate_weights(np.ones(num_blocks), num_blocks)
        assert n_eff == pytest.approx(float(num_blocks), rel=RTOL)

    def test_n_eff_is_kish(self, weights, num_blocks):
        _, n_eff = VarPLSCF._validate_weights(weights, num_blocks)
        assert n_eff == pytest.approx(1.0 / np.sum(weights ** 2), rel=RTOL)
        # concentrating the mass can only lose effective blocks
        assert 1.0 <= n_eff <= num_blocks

    def test_single_block_mass_gives_n_eff_one(self, num_blocks):
        w = np.zeros(num_blocks)
        w[2] = 4.0
        _, n_eff = VarPLSCF._validate_weights(w, num_blocks)
        assert n_eff == pytest.approx(1.0, rel=RTOL)

    def test_wrong_length_raises(self, num_blocks):
        with pytest.raises(ValueError, match='length'):
            VarPLSCF._validate_weights(np.ones(num_blocks + 1), num_blocks)

    def test_two_dimensional_raises(self, num_blocks):
        with pytest.raises(ValueError, match='one-dimensional'):
            VarPLSCF._validate_weights(np.ones((num_blocks, 2)), num_blocks)

    def test_negative_raises(self, num_blocks):
        w = np.ones(num_blocks)
        w[1] = -1e-9
        with pytest.raises(ValueError, match='non-negative'):
            VarPLSCF._validate_weights(w, num_blocks)

    def test_all_zero_raises(self, num_blocks):
        with pytest.raises(ValueError, match='all-zero'):
            VarPLSCF._validate_weights(np.zeros(num_blocks), num_blocks)

    def test_non_finite_raises(self, num_blocks):
        w = np.ones(num_blocks)
        w[0] = np.nan
        with pytest.raises(ValueError, match='finite'):
            VarPLSCF._validate_weights(w, num_blocks)

    @pytest.mark.parametrize('imag', [0.0, 1e-9])
    def test_complex_weights_rejected(self, num_blocks, imag):
        # the score chain is conjugate-linear in places, so F @ W is only valid
        # for a real W; numpy would discard the imaginary part silently
        w = np.ones(num_blocks, dtype=complex)
        w[0] += 1j * imag
        with pytest.raises(ValueError, match='real'):
            VarPLSCF._validate_weights(w, num_blocks)


class TestBlockWeightFactor:

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_matches_reference_formula(self, weights, num_blocks, convention):
        W = VarPLSCF._block_weight_factor(weights, num_blocks, convention)
        expected = reference_factor(weights, num_blocks, convention)
        np.testing.assert_allclose(W, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_is_real(self, weights, num_blocks, convention):
        W = VarPLSCF._block_weight_factor(weights, num_blocks, convention)
        assert not np.iscomplexobj(W)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_shape(self, weights, num_blocks, convention):
        W = VarPLSCF._block_weight_factor(weights, num_blocks, convention)
        assert W.shape == (num_blocks, num_blocks)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_none_equals_explicit_uniform(self, num_blocks, convention):
        W_none = VarPLSCF._block_weight_factor(None, num_blocks, convention)
        W_uniform = VarPLSCF._block_weight_factor(
            np.full(num_blocks, 1.0 / num_blocks), num_blocks, convention)
        np.testing.assert_allclose(W_none, W_uniform, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_uniform_scalar_is_sqrt_n_b(self, num_blocks, convention):
        # every convention agrees at uniform weights, for every normalisation
        W = VarPLSCF._block_weight_factor(None, num_blocks, convention)
        uniform = np.full(num_blocks, 1.0 / num_blocks)
        expected = np.sqrt(float(num_blocks)) * (
            np.eye(num_blocks) - uniform[:, np.newaxis]) * np.sqrt(uniform)[np.newaxis,:]
        np.testing.assert_allclose(W, expected, rtol=RTOL, atol=ATOL)

    def test_unknown_convention_raises(self, weights, num_blocks):
        with pytest.raises(ValueError, match='Unknown convention'):
            VarPLSCF._block_weight_factor(weights, num_blocks, 'inverse-variance')

    def test_convention_validated_before_weights(self, num_blocks):
        # a bad convention is reported even when the weights are bad too
        with pytest.raises(ValueError, match='Unknown convention'):
            VarPLSCF._block_weight_factor(np.ones(num_blocks + 1), num_blocks, 'nonsense')

    def test_weight_errors_propagate(self, num_blocks):
        with pytest.raises(ValueError, match='non-negative'):
            VarPLSCF._block_weight_factor(-np.ones(num_blocks), num_blocks)


class TestDegenerateWeights:
    """n_eff <= 1: all weight mass on a single block, no scatter left."""

    @pytest.fixture
    def degenerate(self, num_blocks):
        w = np.zeros(num_blocks)
        w[1] = 1.0
        return w

    def test_substitution_warns_and_returns_zeros(self, degenerate, num_blocks, caplog):
        with caplog.at_level('WARNING'):
            W = VarPLSCF._block_weight_factor(degenerate, num_blocks, 'substitution')
        np.testing.assert_array_equal(W, np.zeros((num_blocks, num_blocks)))
        assert 'n_eff' in caplog.text

    def test_reliability_raises(self, degenerate, num_blocks):
        with pytest.raises(ValueError, match='n_eff'):
            VarPLSCF._block_weight_factor(degenerate, num_blocks, 'reliability')

    def test_precision_degenerates_to_zeros(self, degenerate, num_blocks):
        # no guard needed: sqrt(n_b) is finite and every column of
        # (I - w 1^T) diag(sqrt(w)) vanishes on its own
        W = VarPLSCF._block_weight_factor(degenerate, num_blocks, 'precision')
        np.testing.assert_allclose(W, np.zeros((num_blocks, num_blocks)),
                                   rtol=RTOL, atol=ATOL)

    def test_single_block_is_degenerate(self):
        with pytest.raises(ValueError, match='n_eff'):
            VarPLSCF._block_weight_factor(np.ones(1), 1, 'reliability')


class TestProperties:
    """The verified properties the downstream reweighting relies on."""

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_uniform_acts_as_identity(self, rng, num_blocks, convention):
        """Property 1 -- the binding invariant: uniform weights change nothing.

        W is not the identity matrix; it acts as one on any factor whose columns
        sum to zero, which every centered block factor's do.
        """
        _, F = block_factor(rng, 5, num_blocks)
        W = VarPLSCF._block_weight_factor(None, num_blocks, convention)
        np.testing.assert_allclose(F @ W, F, rtol=RTOL, atol=ATOL)

    def test_leave_one_out(self, rng, num_blocks):
        """Property 4 -- a zero weight deletes a block, giving a free jackknife."""
        k = 2
        samples, F = block_factor(rng, 5, num_blocks)

        w = np.ones(num_blocks)
        w[k] = 0.0
        FW = F @ VarPLSCF._block_weight_factor(w, num_blocks, 'substitution')

        deleted = np.delete(samples, k, axis=1)
        centered = deleted - deleted.mean(axis=1, keepdims=True)
        F_deleted = centered / np.sqrt((num_blocks - 1) * (num_blocks - 2))

        np.testing.assert_allclose(FW @ FW.T, F_deleted @ F_deleted.T,
                                   rtol=RTOL, atol=ATOL)
        # the deleted block contributes nothing
        np.testing.assert_allclose(FW[:, k], 0.0, rtol=RTOL, atol=ATOL)

    def test_convention_ratio(self, rng, weights, num_blocks):
        """Property 5 -- var(substitution) / var(reliability) == n_b / n_eff."""
        _, F = block_factor(rng, 5, num_blocks)
        _, n_eff = VarPLSCF._validate_weights(weights, num_blocks)

        FW_sub = F @ VarPLSCF._block_weight_factor(weights, num_blocks, 'substitution')
        FW_rel = F @ VarPLSCF._block_weight_factor(weights, num_blocks, 'reliability')
        var_sub = np.einsum('ij,ij->i', FW_sub, FW_sub)
        var_rel = np.einsum('ij,ij->i', FW_rel, FW_rel)

        np.testing.assert_allclose(var_sub / var_rel, num_blocks / n_eff,
                                   rtol=RTOL, atol=ATOL)

    def test_precision_matches_inverse_variance_reading(self, rng, weights, num_blocks):
        """Property 6 -- cov_w = sum_j w_j d_j d_j^T / (n_b - 1) for 'precision'."""
        samples, F = block_factor(rng, 5, num_blocks)
        FW = F @ VarPLSCF._block_weight_factor(weights, num_blocks, 'precision')

        centered = samples - (samples @ weights)[:, np.newaxis]
        expected = np.einsum('j,ij,kj->ik', weights, centered, centered) / (num_blocks - 1)

        np.testing.assert_allclose(FW @ FW.T, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_covariance_is_psd(self, rng, weights, num_blocks, convention):
        """Property 7 -- cov_w = (FW)(FW)^T is PSD, so the sqrt of a row variance is safe."""
        _, F = block_factor(rng, 5, num_blocks)
        FW = F @ VarPLSCF._block_weight_factor(weights, num_blocks, convention)

        cov = FW @ FW.T
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() > -ATOL
        assert np.all(np.einsum('ij,ij->i', FW, FW) >= 0.0)

    def test_varssiref_scalars_would_not_reduce_at_uniform(self, rng, num_blocks):
        """The trap: VarSSIRef's weighted rule is wrong by sqrt(n_b) for this class.

        Guards the normalisation, which no other test here would catch: the two
        classes' conventions share their names but not their formulae.
        """
        samples, F = block_factor(rng, 5, num_blocks)
        uniform = np.full(num_blocks, 1.0 / num_blocks)

        # VarSSIRef's build-time rule, sqrt(w_j) (H_j - H_w) / sqrt(n_eff (n_eff - 1)),
        # which at uniform weights has n_eff = n_b
        transplanted = ((samples - (samples @ uniform)[:, np.newaxis])
                        * np.sqrt(uniform)[np.newaxis,:]
                        / np.sqrt(num_blocks * (num_blocks - 1)))

        assert not np.allclose(transplanted, F, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(transplanted * np.sqrt(num_blocks), F,
                                   rtol=RTOL, atol=ATOL)


@pytest.fixture(scope='module')
def prep_signals():
    """Module-scoped signals, hermetic: other test modules mutate shared fixtures.

    A genuinely stochastic excitation, not the random-phase multisine of
    ``ambient_ifrf``: block scatter is what every test here reasons about.
    """
    from tests.system_ambient_ifrf import ambient_gaussian
    _, sig = ambient_gaussian(8192, 6, [5], FS, 10, seed=42, num_modes=2)
    return PreProcessSignals(sig, FS, ref_channels=[4, 5])


@pytest.fixture(scope='module')
def block_weights(prep_signals):
    """Non-uniform weights over all NUM_BLOCKS training blocks."""
    w = np.random.default_rng(11).random(NUM_BLOCKS) + 0.1
    return w / w.sum()


def build(prep, **kwargs):
    obj = VarPLSCF(prep)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS, **kwargs)
    return obj


class TestBuildTimeWeights:

    def test_uniform_reproduces_unweighted_build(self, prep_signals):
        """Property 2 and the binding invariant, at the level of a real build.

        The test that catches the trap: VarSSIRef's weighted rule would land here
        a factor sqrt(n_b) away, and the covariance a factor n_b away.
        """
        plain = build(prep_signals)
        uniform = build(prep_signals, weights=np.full(NUM_BLOCKS, 1.0 / NUM_BLOCKS))

        np.testing.assert_allclose(uniform.pos_half_spectra, plain.pos_half_spectra,
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(uniform.spec_block_factor, plain.spec_block_factor,
                                   rtol=RTOL, atol=ATOL)

    def test_unnormalised_uniform_reproduces_unweighted_build(self, prep_signals):
        # the scale of the weights carries no meaning
        plain = build(prep_signals)
        uniform = build(prep_signals, weights=np.full(NUM_BLOCKS, 7.0))
        np.testing.assert_allclose(uniform.spec_block_factor, plain.spec_block_factor,
                                   rtol=RTOL, atol=ATOL)

    def test_point_estimate_is_weighted_mean_of_correlations(self, prep_signals, block_weights):
        """The half-spectrum is built from sum_j w_j R_j."""
        obj = build(prep_signals, weights=block_weights)

        corr_blocks = obj.prep_signals.corr_matrices_bt[np.arange(NUM_BLOCKS), ..., :M_LAGS]
        expected_corr = np.tensordot(block_weights, corr_blocks, axes=(0, 0))
        _, expected_spectra, _ = obj._windowed_half_spectrum(
            expected_corr, obj.nperseg, obj.window_decay,
            obj.begin_frequency, obj.end_frequency)

        np.testing.assert_allclose(obj.pos_half_spectra, expected_spectra,
                                   rtol=RTOL, atol=ATOL)

    def test_weighted_point_estimate_differs_from_unweighted(self, prep_signals, block_weights):
        # guards the test above against a silently ignored weights argument
        plain = build(prep_signals)
        weighted = build(prep_signals, weights=block_weights)
        assert not np.allclose(weighted.pos_half_spectra, plain.pos_half_spectra)

    def test_factor_is_centered_at_the_weighted_mean(self, prep_signals, block_weights):
        """The deviations are taken about the mean the model was identified from."""
        obj = build(prep_signals, weights=block_weights)
        # column j is sqrt(w_j)(H_j - H_w)/sqrt(n_eff - 1), so the weighted
        # recombination sum_j sqrt(w_j) * column_j vanishes
        recombined = np.tensordot(obj.spec_block_factor, np.sqrt(block_weights), axes=(-1, 0))
        assert np.max(np.abs(recombined)) < 1e-9 * np.max(np.abs(obj.spec_block_factor))

    def test_n_eff_stored(self, prep_signals, block_weights):
        obj = build(prep_signals, weights=block_weights)
        np.testing.assert_allclose(obj.weights, block_weights, rtol=RTOL, atol=ATOL)
        assert obj.n_eff == pytest.approx(1.0 / np.sum(block_weights ** 2), rel=RTOL)

    def test_unweighted_build_reports_uniform_n_eff(self, prep_signals):
        obj = build(prep_signals)
        assert obj.weights is None
        assert obj.n_eff == float(NUM_BLOCKS)

    def test_post_hoc_matches_build_time(self, prep_signals, block_weights):
        """Property 3 -- the reweighting identity reproduces a weighted build.

        Pins the pairing of the two ``a = 1`` rules against each other: the
        build-time sqrt(w_j)(.)/sqrt(n_eff - 1) and the post-hoc s(w) on the
        'substitution' default.
        """
        plain = build(prep_signals)
        weighted = build(prep_signals, weights=block_weights)

        W = VarPLSCF._block_weight_factor(block_weights, NUM_BLOCKS, 'substitution')
        reweighted = plain.spec_block_factor @ W

        np.testing.assert_allclose(reweighted, weighted.spec_block_factor,
                                   rtol=1e-8, atol=1e-12)

    def test_weights_are_indexed_over_training_blocks(self, prep_signals):
        """Weights address the blocks that train, not all num_blocks of them."""
        training = np.arange(0, NUM_BLOCKS, 2)
        w = np.full(training.shape[0], 1.0 / training.shape[0])

        subset = build(prep_signals, training_blocks=training, weights=w)
        plain = build(prep_signals, training_blocks=training)

        assert subset.spec_block_factor.shape[-1] == training.shape[0]
        np.testing.assert_allclose(subset.spec_block_factor, plain.spec_block_factor,
                                   rtol=RTOL, atol=ATOL)

    def test_weights_length_must_match_training_blocks(self, prep_signals):
        training = np.arange(0, NUM_BLOCKS, 2)
        with pytest.raises(ValueError, match='length'):
            build(prep_signals, training_blocks=training, weights=np.ones(NUM_BLOCKS))

    def test_degenerate_weights_raise(self, prep_signals):
        w = np.zeros(NUM_BLOCKS)
        w[3] = 1.0
        with pytest.raises(ValueError, match='n_eff'):
            build(prep_signals, weights=w)

    def test_weights_require_num_blocks(self, prep_signals):
        obj = PLSCF(prep_signals)
        with pytest.raises(ValueError, match='require num_blocks'):
            obj.build_half_spectra(nperseg=M_LAGS, weights=np.ones(NUM_BLOCKS))

    def test_low_n_eff_warns(self, prep_signals, caplog):
        w = np.full(NUM_BLOCKS, 1e-3)
        w[:3] = 1.0
        with caplog.at_level('WARNING'):
            build(prep_signals, weights=w)
        assert 'n_eff' in caplog.text

    def test_no_warning_when_unweighted(self, prep_signals, caplog):
        with caplog.at_level('WARNING'):
            build(prep_signals)
        assert 'n_eff' not in caplog.text


class TestExternalCorrelations:

    def test_round_trip_against_internal_build(self, prep_signals):
        """An external feed of the same correlations reproduces the internal build."""
        internal = build(prep_signals)
        corr = np.array(internal.prep_signals.corr_matrices_bt[..., :M_LAGS], copy=True)

        external = build(prep_signals, corr_matrices=corr)

        np.testing.assert_allclose(external.pos_half_spectra, internal.pos_half_spectra,
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(external.spec_block_factor, internal.spec_block_factor,
                                   rtol=RTOL, atol=ATOL)

    def test_external_feed_is_used(self, prep_signals):
        internal = build(prep_signals)
        corr = np.array(internal.prep_signals.corr_matrices_bt[..., :M_LAGS], copy=True)
        corr[0] *= 2.0

        external = build(prep_signals, corr_matrices=corr)
        assert not np.allclose(external.pos_half_spectra, internal.pos_half_spectra)

    def test_longer_last_axis_is_truncated(self, prep_signals):
        internal = build(prep_signals)
        corr = np.array(internal.prep_signals.corr_matrices_bt, copy=True)
        padded = np.concatenate([corr[..., :M_LAGS], np.ones(corr.shape[:3] + (17,))], axis=-1)

        external = build(prep_signals, corr_matrices=padded)
        np.testing.assert_allclose(external.pos_half_spectra, internal.pos_half_spectra,
                                   rtol=RTOL, atol=ATOL)

    def test_combines_with_weights(self, prep_signals, block_weights):
        internal = build(prep_signals, weights=block_weights)
        corr = np.array(internal.prep_signals.corr_matrices_bt[..., :M_LAGS], copy=True)

        external = build(prep_signals, corr_matrices=corr, weights=block_weights)
        np.testing.assert_allclose(external.spec_block_factor, internal.spec_block_factor,
                                   rtol=RTOL, atol=ATOL)

    def test_wrong_num_blocks_raises(self, prep_signals):
        internal = build(prep_signals)
        corr = np.array(internal.prep_signals.corr_matrices_bt[:NUM_BLOCKS - 1, ..., :M_LAGS])
        with pytest.raises(ValueError, match='num_blocks'):
            build(prep_signals, corr_matrices=corr)

    def test_wrong_channel_count_raises(self, prep_signals):
        internal = build(prep_signals)
        corr = np.array(internal.prep_signals.corr_matrices_bt[:, :3, ..., :M_LAGS])
        with pytest.raises(ValueError, match='channel pairs'):
            build(prep_signals, corr_matrices=corr)

    def test_too_few_lags_raises(self, prep_signals):
        internal = build(prep_signals)
        corr = np.array(internal.prep_signals.corr_matrices_bt[..., :M_LAGS // 2])
        with pytest.raises(ValueError, match='nperseg'):
            build(prep_signals, corr_matrices=corr)

    def test_wrong_ndim_raises(self, prep_signals):
        with pytest.raises(ValueError, match='dimensions'):
            build(prep_signals, corr_matrices=np.ones((NUM_BLOCKS, 6, 2)))

    def test_requires_num_blocks(self, prep_signals):
        obj = PLSCF(prep_signals)
        with pytest.raises(ValueError, match='require num_blocks'):
            obj.build_half_spectra(nperseg=M_LAGS, corr_matrices=np.ones((2, 6, 2, M_LAGS)))


MAX_ORDER = 6

STD_ARRAYS = ('std_frequencies', 'std_damping',
              'std_participation_vectors', 'std_mode_shapes')

# Tolerances for comparing whole std_* arrays across two propagation paths.
#
# Uniform weights reproduce the unweighted *factor* to machine precision (3.8e-16
# relative, asserted directly in TestBuildTimeWeights), but the std arrays are
# reached through np.linalg.solve(M_aa, .) and cannot hold that. The normal-equation
# formulation squares the condition number: cond(M_aa) is ~1e2 at order 1 and
# ~6e10 by order 6, and the relative error of the solve tracks eps * cond almost
# exactly -- measured 1.6e-15 at order 1 rising to 2.7e-10 at order 6, i.e. zero
# error where the problem is well conditioned. That amplification is the
# estimator's, not the weighting's, and it bounds every comparison here:
# uniform-vs-unweighted (worst entrywise 4.3e-10) and Tier A vs Tier B (3.3e-10),
# the latter because only Tier B re-runs the solve. RTOL is set an order of
# magnitude above both. Any real weighting effect is O(1e-2), which these
# tolerances still catch by six orders of magnitude.
#
# ATOL carries the entries that are exactly zero by construction -- the
# normalised mode-shape and participation components, and the unused mode slots
# of each order, together ~75% of std_mode_shapes -- where the two paths land on
# different roundoff around 1e-16 and an entrywise rtol is meaningless.
STD_RTOL = 1e-8
STD_ATOL = 1e-12


@pytest.fixture(scope='module')
def computed(prep_signals):
    """An unweighted identification, the baseline every reweighting starts from."""
    obj = VarPLSCF(prep_signals)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
    return obj


@pytest.fixture(scope='module')
def baseline_stds(computed):
    return {name: np.array(getattr(computed, name), copy=True) for name in STD_ARRAYS}


def assert_stds_close(obj, expected, rtol=STD_RTOL, atol=STD_ATOL):
    for name, ref in expected.items():
        actual = getattr(obj, name)
        np.testing.assert_allclose(actual, ref, rtol=rtol, atol=atol,
                                   err_msg=f'{name} differs')


class TestTierB:
    """compute_modal_params_weighted: reweight the factor, re-run the loop."""

    def test_uniform_reproduces_unweighted_run(self, prep_signals, baseline_stds):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params_weighted(MAX_ORDER, None, modal_contrib=False)
        assert_stds_close(obj, baseline_stds)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_explicit_uniform_reproduces_unweighted_run(self, prep_signals,
                                                        baseline_stds, convention):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params_weighted(
            MAX_ORDER, np.full(NUM_BLOCKS, 1.0 / NUM_BLOCKS), convention,
            modal_contrib=False)
        assert_stds_close(obj, baseline_stds)

    def test_point_estimates_untouched(self, prep_signals, computed, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)

        # the weights enter the covariance only
        np.testing.assert_allclose(obj.modal_frequencies, computed.modal_frequencies,
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(obj.modal_damping, computed.modal_damping,
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(obj.mode_shapes, computed.mode_shapes,
                                   rtol=RTOL, atol=ATOL)

    def test_weights_change_the_stds(self, prep_signals, baseline_stds, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)
        # guards every "reproduces" test above against a silently ignored argument
        assert not np.allclose(obj.std_frequencies, baseline_stds['std_frequencies'])

    def test_stored_factor_not_mutated(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        before = np.array(obj.spec_block_factor, copy=True)

        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)

        np.testing.assert_array_equal(obj.spec_block_factor, before)

    def test_records_the_weighting(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, 'reliability',
                                          modal_contrib=False)
        np.testing.assert_allclose(obj.block_weights, block_weights, rtol=RTOL, atol=ATOL)
        assert obj.block_weight_convention == 'reliability'

    def test_unweighted_run_clears_the_record(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
        assert obj.block_weights is None
        assert obj.block_weight_convention is None

    def test_convention_ratio_on_frequencies(self, prep_signals, block_weights):
        """Property 5 survives the whole chain: var scales by n_b / n_eff."""
        _, n_eff = VarPLSCF._validate_weights(block_weights, NUM_BLOCKS)

        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)

        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, 'substitution',
                                          modal_contrib=False)
        std_sub = np.array(obj.std_frequencies, copy=True)
        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, 'reliability',
                                          modal_contrib=False)
        std_rel = np.array(obj.std_frequencies, copy=True)

        mask = std_rel > 0
        np.testing.assert_allclose((std_sub[mask] / std_rel[mask]) ** 2,
                                   NUM_BLOCKS / n_eff, rtol=1e-8)

    def test_build_time_weights_reject_post_hoc(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS, weights=block_weights)
        with pytest.raises(NotImplementedError, match='unweighted build'):
            obj.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)

    def test_unknown_convention_raises(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        with pytest.raises(ValueError, match='Unknown convention'):
            obj.compute_modal_params_weighted(MAX_ORDER, block_weights, 'nonsense',
                                              modal_contrib=False)

    def test_complex_coefficients_rejected(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        with pytest.raises(NotImplementedError, match='real denominator'):
            obj.compute_modal_params_weighted(MAX_ORDER, block_weights,
                                              complex_coefficients=True)

    def test_missing_factor_raises(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.spec_block_factor = None
        with pytest.raises(RuntimeError, match='covariance factor is missing'):
            obj.compute_modal_params_weighted(MAX_ORDER, block_weights)

    def test_low_n_eff_warns(self, prep_signals, caplog):
        w = np.full(NUM_BLOCKS, 1e-3)
        w[:3] = 1.0
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        with caplog.at_level('WARNING'):
            obj.compute_modal_params_weighted(MAX_ORDER, w, modal_contrib=False)
        assert 'n_eff' in caplog.text

    def test_uniform_does_not_warn(self, prep_signals, caplog):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        with caplog.at_level('WARNING'):
            obj.compute_modal_params_weighted(MAX_ORDER, None, modal_contrib=False)
        assert 'n_eff' not in caplog.text


def build_cached(prep, dtype=np.float64, cache='full'):
    """An identification carrying its per-mode factor caches."""
    obj = VarPLSCF(prep, cache_variance_factors=True, cache=cache, cache_dtype=dtype)
    obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
    obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
    return obj


class TestTierACache:

    def test_off_by_default(self, computed):
        assert computed._U_fixi_cache is None

    def test_cache_shapes(self, prep_signals):
        obj = build_cached(prep_signals)
        n_l = obj.prep_signals.num_analised_channels
        n_r = obj.prep_signals.num_ref_channels

        assert set(obj._U_fixi_cache) == set(range(1, MAX_ORDER))
        for order, U in obj._U_fixi_cache.items():
            n_modes = int((obj.modal_frequencies[order] > 0).sum())
            assert U.shape == (n_modes, 2, NUM_BLOCKS)
            assert obj._U_L_cache[order].shape == (n_modes, 2 * n_r, NUM_BLOCKS)
            assert obj._U_phii_cache[order].shape == (n_modes, 2 * n_l, NUM_BLOCKS)

    def test_cache_dtype_default_is_single_precision(self, prep_signals):
        obj = VarPLSCF(prep_signals, cache_variance_factors=True)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
        assert obj._U_fixi_cache[1].dtype == np.float32

    def test_per_call_kwarg_overrides_constructor(self, prep_signals):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False,
                                 cache_variance_factors=True)
        assert obj._U_fixi_cache is not None

    def test_freqdamp_caches_only_the_frequency_factor(self, prep_signals):
        obj = build_cached(prep_signals, cache='freqdamp')
        assert obj._U_fixi_cache is not None
        assert obj._U_L_cache is None
        assert obj._U_phii_cache is None

    def test_unknown_cache_mode_raises(self, prep_signals):
        with pytest.raises(ValueError, match="'full' or 'freqdamp'"):
            VarPLSCF(prep_signals, cache='everything')

    def test_fresh_run_invalidates_the_cache(self, prep_signals):
        obj = build_cached(prep_signals)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False,
                                 cache_variance_factors=False)
        assert obj._U_fixi_cache is None

    def test_tier_b_leaves_the_cache_intact(self, prep_signals, block_weights):
        """Deliberate divergence from VarSSIRef, which wipes on any fresh loop.

        The cached factors are unweighted and stay valid across a Tier B call.
        """
        obj = build_cached(prep_signals)
        before = {order: np.array(U, copy=True) for order, U in obj._U_fixi_cache.items()}

        obj.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)

        assert obj._U_fixi_cache is not None
        for order, U in before.items():
            np.testing.assert_array_equal(obj._U_fixi_cache[order], U)


class TestApplyBlockWeights:

    def test_matches_tier_b(self, prep_signals, block_weights):
        """The acceptance gate: Tier A reproduces its oracle.

        The two paths are mathematically identical -- Tier A forms L(F) @ W and
        Tier B forms L(F @ W) -- but only Tier B re-runs the ill-conditioned
        solve, so they agree to eps * cond(M_aa) rather than to machine
        precision. Measured worst entrywise error over the entries that carry
        any magnitude: 3.3e-10 (see the STD_RTOL note above).
        """
        tier_a = build_cached(prep_signals)
        tier_a.apply_block_weights(block_weights)

        tier_b = VarPLSCF(prep_signals)
        tier_b.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        tier_b.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)

        for name in STD_ARRAYS:
            np.testing.assert_allclose(getattr(tier_a, name), getattr(tier_b, name),
                                       rtol=STD_RTOL, atol=STD_ATOL, err_msg=name)

    @pytest.mark.parametrize('convention', CONVENTIONS)
    def test_matches_tier_b_per_convention(self, prep_signals, block_weights, convention):
        tier_a = build_cached(prep_signals)
        tier_a.apply_block_weights(block_weights, convention)

        tier_b = VarPLSCF(prep_signals)
        tier_b.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        tier_b.compute_modal_params_weighted(MAX_ORDER, block_weights, convention,
                                             modal_contrib=False)

        np.testing.assert_allclose(tier_a.std_frequencies, tier_b.std_frequencies,
                                   rtol=STD_RTOL, atol=STD_ATOL)

    def test_matches_tier_b_at_float32_default(self, prep_signals, block_weights):
        tier_a = build_cached(prep_signals, dtype=np.float32)
        tier_a.apply_block_weights(block_weights)

        tier_b = VarPLSCF(prep_signals)
        tier_b.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        tier_b.compute_modal_params_weighted(MAX_ORDER, block_weights, modal_contrib=False)

        for name in STD_ARRAYS:
            np.testing.assert_allclose(getattr(tier_a, name), getattr(tier_b, name),
                                       rtol=1e-5, atol=1e-8, err_msg=name)

    def test_uniform_restores_the_stored_stds(self, prep_signals, baseline_stds, block_weights):
        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights)
        obj.apply_block_weights(None)

        assert_stds_close(obj, baseline_stds, rtol=1e-9, atol=1e-12)
        assert obj.block_weights is None

    def test_weights_change_the_stds(self, prep_signals, baseline_stds, block_weights):
        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights)
        # guards the restore test against a no-op implementation
        assert not np.allclose(obj.std_frequencies, baseline_stds['std_frequencies'])

    def test_point_estimates_untouched(self, prep_signals, computed, block_weights):
        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights)
        np.testing.assert_allclose(obj.modal_frequencies, computed.modal_frequencies,
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(obj.mode_shapes, computed.mode_shapes,
                                   rtol=RTOL, atol=ATOL)

    def test_cache_not_mutated(self, prep_signals, block_weights):
        obj = build_cached(prep_signals)
        before = {order: np.array(U, copy=True) for order, U in obj._U_fixi_cache.items()}
        obj.apply_block_weights(block_weights)
        for order, U in before.items():
            np.testing.assert_array_equal(obj._U_fixi_cache[order], U)

    def test_repeated_application_is_not_cumulative(self, prep_signals, block_weights):
        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights)
        once = np.array(obj.std_frequencies, copy=True)
        obj.apply_block_weights(block_weights)
        np.testing.assert_allclose(obj.std_frequencies, once, rtol=RTOL, atol=ATOL)

    def test_records_the_weighting(self, prep_signals, block_weights):
        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights, 'precision')
        np.testing.assert_allclose(obj.block_weights, block_weights, rtol=RTOL, atol=ATOL)
        assert obj.block_weight_convention == 'precision'

    def test_freqdamp_leaves_shape_stds_untouched(self, prep_signals, baseline_stds,
                                                  block_weights):
        obj = build_cached(prep_signals, cache='freqdamp')
        obj.apply_block_weights(block_weights)

        assert not np.allclose(obj.std_frequencies, baseline_stds['std_frequencies'])
        np.testing.assert_allclose(obj.std_mode_shapes, baseline_stds['std_mode_shapes'],
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(obj.std_participation_vectors,
                                   baseline_stds['std_participation_vectors'],
                                   rtol=RTOL, atol=ATOL)

    def test_no_cache_raises(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
        # no automatic Tier B fallback
        with pytest.raises(RuntimeError, match='No cached uncertainty factors'):
            obj.apply_block_weights(block_weights)

    def test_before_compute_raises(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals, cache_variance_factors=True)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        with pytest.raises(RuntimeError, match='compute_modal_params'):
            obj.apply_block_weights(block_weights)

    def test_build_time_weights_reject_post_hoc(self, prep_signals, block_weights):
        obj = VarPLSCF(prep_signals, cache_variance_factors=True)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS, weights=block_weights)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
        with pytest.raises(NotImplementedError, match='unweighted build'):
            obj.apply_block_weights(block_weights)

    def test_unknown_convention_raises(self, prep_signals, block_weights):
        obj = build_cached(prep_signals)
        with pytest.raises(ValueError, match='Unknown convention'):
            obj.apply_block_weights(block_weights, 'nonsense')

    def test_wrong_length_raises(self, prep_signals):
        obj = build_cached(prep_signals)
        with pytest.raises(ValueError, match='length'):
            obj.apply_block_weights(np.ones(NUM_BLOCKS + 1))

    def test_degenerate_weights_zero_the_stds(self, prep_signals):
        w = np.zeros(NUM_BLOCKS)
        w[0] = 1.0
        obj = build_cached(prep_signals)
        obj.apply_block_weights(w)
        np.testing.assert_allclose(obj.std_frequencies, 0.0, atol=ATOL)

    def test_low_n_eff_warns(self, prep_signals, caplog):
        w = np.full(NUM_BLOCKS, 1e-3)
        w[:3] = 1.0
        obj = build_cached(prep_signals)
        with caplog.at_level('WARNING'):
            obj.apply_block_weights(w)
        assert 'n_eff' in caplog.text

    def test_uniform_does_not_warn(self, prep_signals, caplog):
        obj = build_cached(prep_signals)
        with caplog.at_level('WARNING'):
            obj.apply_block_weights(None)
        assert 'n_eff' not in caplog.text

    def test_leave_one_out(self, prep_signals):
        """Property 4 through the whole chain: a zero weight deletes a block."""
        w = np.ones(NUM_BLOCKS)
        w[5] = 0.0

        obj = build_cached(prep_signals)
        obj.apply_block_weights(w)

        deleted = VarPLSCF(prep_signals)
        deleted.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS,
                                   training_blocks=np.delete(np.arange(NUM_BLOCKS), 5))
        deleted.compute_modal_params(MAX_ORDER, modal_contrib=False)

        # the deleted build re-identifies from 19 blocks, so its point estimates
        # differ; only the frequencies' std is compared, and loosely
        mask = deleted.std_frequencies > 0
        ratio = obj.std_frequencies[mask] / deleted.std_frequencies[mask]
        assert 0.5 < np.median(ratio) < 2.0


class TestPersistence:

    def test_round_trip_reweights_after_load(self, prep_signals, block_weights, tmp_path):
        obj = build_cached(prep_signals)
        fname = str(tmp_path / 'varplscf.npz')
        obj.save_state(fname)

        loaded = VarPLSCF.load_state(fname, prep_signals)
        loaded.apply_block_weights(block_weights)

        reference = build_cached(prep_signals)
        reference.apply_block_weights(block_weights)

        for name in STD_ARRAYS:
            np.testing.assert_allclose(getattr(loaded, name), getattr(reference, name),
                                       rtol=RTOL, atol=ATOL, err_msg=name)

    def test_round_trip_without_cache_disables_tier_a(self, prep_signals, block_weights,
                                                      tmp_path):
        obj = VarPLSCF(prep_signals)
        obj.build_half_spectra(nperseg=M_LAGS, num_blocks=NUM_BLOCKS)
        obj.compute_modal_params(MAX_ORDER, modal_contrib=False)
        fname = str(tmp_path / 'nocache.npz')
        obj.save_state(fname)

        loaded = VarPLSCF.load_state(fname, prep_signals)
        assert loaded._U_fixi_cache is None
        with pytest.raises(RuntimeError, match='No cached uncertainty factors'):
            loaded.apply_block_weights(block_weights)

    def test_weight_record_round_trips(self, prep_signals, block_weights, tmp_path):
        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights, 'reliability')
        fname = str(tmp_path / 'weighted.npz')
        obj.save_state(fname)

        loaded = VarPLSCF.load_state(fname, prep_signals)
        np.testing.assert_allclose(loaded.block_weights, block_weights, rtol=RTOL, atol=ATOL)
        assert loaded.block_weight_convention == 'reliability'

    def test_unweighted_record_round_trips_as_none(self, prep_signals, tmp_path):
        obj = build_cached(prep_signals)
        fname = str(tmp_path / 'unweighted.npz')
        obj.save_state(fname)

        loaded = VarPLSCF.load_state(fname, prep_signals)
        assert loaded.block_weights is None
        assert loaded.block_weight_convention is None


class TestStabilDiagramSmoke:
    """StabilDiagram duck-types variance support on std_frequencies, so
    reweighting is free; it must see the refreshed values."""

    def test_sees_refreshed_stds(self, prep_signals, block_weights):
        from pyOMA.core.StabilDiagram import StabilCalc

        obj = build_cached(prep_signals)
        obj.apply_block_weights(block_weights)

        stabil = StabilCalc(obj)
        assert stabil.capabilities['std']
        np.testing.assert_allclose(stabil.modal_data.std_frequencies, obj.std_frequencies,
                                   rtol=RTOL, atol=ATOL)
