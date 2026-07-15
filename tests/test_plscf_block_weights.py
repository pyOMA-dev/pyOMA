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

from pyOMA.core.VarPLSCF import VarPLSCF

RTOL = 1e-10
ATOL = 1e-13

CONVENTIONS = ('substitution', 'reliability', 'precision')


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
