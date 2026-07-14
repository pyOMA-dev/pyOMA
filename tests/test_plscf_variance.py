"""Variance estimation tests for the pLSCF method (VarPLSCF)."""
import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.VarPLSCF import VarPLSCF

FS = 128
M_LAGS = 200
NUM_BLOCKS = 20


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
