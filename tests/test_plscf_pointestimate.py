"""Point-estimate correctness tests for the pLSCF estimator.

These tests exercise properties that must hold for *any* correct pLSCF
implementation, independent of the data:

* the reduced normal equations weight every output channel equally,
* the LSFD design matrix reproduces the modal spectrum model it is fitted to,
* each mode shape is integrated at its own natural frequency,
* the modal spectrum synthesis reproduces the model it was decomposed from.

They are written against noise-free spectra synthesized from a known modal
model, so a failure localizes a defect in the estimator rather than a lack of
identifiability.
"""
import numpy as np
import pytest

from pyOMA.core.PLSCF import PLSCF
from pyOMA.core.PreProcessingTools import PreProcessSignals

FS = 128.0          # sampling rate [Hz]
N_L = 4             # analysed channels
N_R = 2             # reference channels
N_LINES = 64        # retained frequency lines
SEED = 7


def _modal_half_spectra(omega, eigenvalues, part_vecs, mode_shapes,
                        lower_res=None, upper_res=None):
    """Positive half-spectra of the pLSCF modal model.

    ``H[l, r, f] = sum_m phi[l, m] L[r, m] / (i w_f - lambda_m)
                 + conj(phi[l, m] L[r, m]) / (i w_f - conj(lambda_m))``
    plus the optional lower/upper residual terms.
    """
    n_l = mode_shapes.shape[0]
    n_r = part_vecs.shape[0]
    spectra = np.zeros((n_l, n_r, omega.shape[0]), dtype=complex)
    i_omega = 1j * omega[np.newaxis, np.newaxis, :]
    for i_mode in range(eigenvalues.shape[0]):
        lambda_m = eigenvalues[i_mode]
        numerator = np.outer(mode_shapes[:, i_mode], part_vecs[:, i_mode])
        spectra += numerator[:, :, np.newaxis] / (i_omega - lambda_m)
        spectra += np.conj(numerator)[:, :, np.newaxis] / (i_omega - np.conj(lambda_m))
    if lower_res is not None:
        spectra += lower_res[:, :, np.newaxis]
    if upper_res is not None:
        spectra += upper_res[:, :, np.newaxis] * omega[np.newaxis, np.newaxis, :] ** 2
    return spectra


def _modal_model(frequencies=(5.0, 25.0), damping=(0.02, 0.03)):
    """Return (omega, eigenvalues, part_vecs, mode_shapes) of a known system."""
    rng = np.random.default_rng(SEED)
    omega_n = 2 * np.pi * np.asarray(frequencies)
    zeta = np.asarray(damping)
    eigenvalues = -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta ** 2)

    n_modes = eigenvalues.shape[0]
    mode_shapes = rng.normal(size=(N_L, n_modes)) + 1j * rng.normal(size=(N_L, n_modes))
    part_vecs = rng.normal(size=(N_R, n_modes)) + 1j * rng.normal(size=(N_R, n_modes))

    omega = 2 * np.pi * np.linspace(1.0, 40.0, N_LINES)
    return omega, eigenvalues, part_vecs, mode_shapes


def _make_plscf(omega, pos_half_spectra, accel_channels=None,
                velo_channels=None, disp_channels=None):
    """A PLSCF object with half-spectra injected directly (no data pipeline)."""
    rng = np.random.default_rng(SEED)
    signals = rng.normal(size=(512, N_L))
    prep_signals = PreProcessSignals(
        signals, FS, ref_channels=list(range(N_R)),
        accel_channels=accel_channels, velo_channels=velo_channels,
        disp_channels=disp_channels)

    obj = PLSCF(prep_signals)
    obj.selected_omega_vector = omega
    obj.pos_half_spectra = pos_half_spectra
    obj.nperseg = 2 * N_LINES
    obj.begin_frequency = 0.0
    obj.end_frequency = FS / 2
    obj.factor_a = None
    obj.state[0] = True
    return obj


def _poles(obj, alpha, order):
    """Sorted eigenvalues of the companion matrix built from alpha."""
    companion = obj._build_companion_matrix_residuals(alpha, N_R, order)
    return np.sort_complex(np.linalg.eigvals(companion))


def test_poles_invariant_under_output_channel_permutation():
    """The reduced normal equations must not weight one output over another.

    ``M = sum_o (T_o - S_o^T R_o^-1 S_o)`` (Peeters 2004 Eq. 10) is a plain sum
    over output channels, so relabelling the outputs cannot move the poles.
    """
    omega, eigenvalues, part_vecs, mode_shapes = _modal_model()
    spectra = _modal_half_spectra(omega, eigenvalues, part_vecs, mode_shapes)

    # Noise makes the order-4 model inexact, so the per-output normal equations
    # no longer share a common null space and any output weighting shows up.
    rng = np.random.default_rng(SEED + 1)
    scale = 0.05 * np.abs(spectra).mean()
    spectra = spectra + scale * (rng.normal(size=spectra.shape)
                                 + 1j * rng.normal(size=spectra.shape))

    order = 4
    obj = _make_plscf(omega, spectra)
    alpha, _ = obj.estimate_model(order)
    poles = _poles(obj, alpha, order)

    permutation = [3, 1, 0, 2]
    obj.pos_half_spectra = spectra[permutation, :, :]
    alpha_perm, _ = obj.estimate_model(order)
    poles_perm = _poles(obj, alpha_perm, order)

    np.testing.assert_allclose(poles_perm, poles, rtol=1e-9, atol=1e-12)


def test_lsfd_recovers_mode_shapes_from_noise_free_spectra():
    """The LSFD design matrix must reproduce the model that generated the data.

    Given exact eigenvalues and participation vectors and a spectrum built from
    the modal model with zero residuals, the least-squares fit is consistent and
    must return the generating mode shapes.
    """
    omega, eigenvalues, part_vecs, mode_shapes = _modal_model()
    spectra = _modal_half_spectra(omega, eigenvalues, part_vecs, mode_shapes)

    obj = _make_plscf(omega, spectra)
    n_modes = eigenvalues.shape[0]

    _, mode_shapes_raw, lower_res, upper_res = obj._fit_mode_shapes_ls(
        eigenvalues, N_L, N_R, n_modes, part_vecs)

    np.testing.assert_allclose(mode_shapes_raw, mode_shapes, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(lower_res, np.zeros_like(lower_res), atol=1e-8)
    np.testing.assert_allclose(upper_res, np.zeros_like(upper_res), atol=1e-8)


def test_each_mode_shape_is_integrated_at_its_own_frequency():
    """Integration to displacements must use each mode's own frequency.

    Channels 0-1 are accelerations and channels 2-3 displacements, so the
    integration factor does not cancel in the rescaled shape and the frequency
    used is observable.  The check compares against the raw shapes the fit
    actually returned, so it is independent of the fit's own correctness.
    """
    omega, eigenvalues, part_vecs, mode_shapes = _modal_model()
    spectra = _modal_half_spectra(omega, eigenvalues, part_vecs, mode_shapes)

    obj = _make_plscf(omega, spectra, accel_channels=[0, 1],
                      velo_channels=[], disp_channels=[2, 3])
    n_modes = eigenvalues.shape[0]

    mode_shapes_out, mode_shapes_raw, _, _ = obj._fit_mode_shapes_ls(
        eigenvalues, N_L, N_R, n_modes, part_vecs)

    for i_mode in range(n_modes):
        own_omega = np.abs(eigenvalues[i_mode])
        expected = obj.rescale_mode_shape(obj.integrate_quantities(
            mode_shapes_raw[:, i_mode], obj.prep_signals.accel_channels,
            obj.prep_signals.velo_channels, own_omega))
        np.testing.assert_allclose(
            mode_shapes_out[:, i_mode], expected, rtol=1e-8, atol=1e-8,
            err_msg=f'mode {i_mode} not integrated at its own frequency')


def test_modal_context_exposes_untouched_left_and_right_eigenvectors():
    """The retained eigendecomposition must satisfy the relations it claims.

    Uncertainty propagation differentiates through both eigenvector sides, so
    they must be correctly labelled and must not be mutated by the participation
    vector normalization, which indexes into them with a writable view.
    """
    omega, eigenvalues, part_vecs, mode_shapes = _modal_model()
    spectra = _modal_half_spectra(omega, eigenvalues, part_vecs, mode_shapes)

    order = 4
    obj = _make_plscf(omega, spectra)
    alpha, beta_l_i = obj.estimate_model(order)
    obj.modal_analysis_residuals(alpha, beta_l_i)
    ctx = obj._modal_ctx

    for i, z_i in enumerate(ctx.eigvals_z):
        np.testing.assert_allclose(
            ctx.A_c @ ctx.eigvecs_r[:, i], z_i * ctx.eigvecs_r[:, i], atol=1e-10,
            err_msg=f'column {i} of eigvecs_r is not a right eigenvector')
        np.testing.assert_allclose(
            ctx.eigvecs_l[:, i].conj() @ ctx.A_c, z_i * ctx.eigvecs_l[:, i].conj(),
            atol=1e-10, err_msg=f'column {i} of eigvecs_l is not a left eigenvector')

    # mode_indices must reproduce the returned modes, and participation vectors
    # must come from the left side (Steffensen 2025 Eq. 15)
    for i, ind in enumerate(ctx.mode_indices):
        part_vec = ctx.eigvecs_l[-N_R:, ind]
        part_vec = part_vec / part_vec[np.argmax(np.abs(part_vec))]
        np.testing.assert_allclose(
            part_vec, obj._participation_vectors[:, i], rtol=1e-10, atol=1e-12)


def test_modal_synthesis_reproduces_the_model_it_decomposes():
    """Summing the per-mode synthesized spectra must return the input spectrum.

    With the generating modal parameters and zero residuals installed, the modal
    synthesis is exact and every modal contribution is real and sums to one.
    """
    omega, eigenvalues, part_vecs, mode_shapes = _modal_model()
    spectra = _modal_half_spectra(omega, eigenvalues, part_vecs, mode_shapes)

    obj = _make_plscf(omega, spectra)
    obj._eigenvalues = eigenvalues
    obj._participation_vectors = part_vecs
    obj._mode_shapes_raw = mode_shapes
    obj._lower_residuals = np.zeros((N_L, N_R), dtype=complex)
    obj._upper_residuals = np.zeros((N_L, N_R), dtype=complex)

    half_spec_modal, modal_contributions = obj._synthesize_spectrum_modal(
        N_L, N_R, omega.shape[0], omega, spectra)

    np.testing.assert_allclose(
        np.sum(half_spec_modal, axis=-1), spectra, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        np.sum(modal_contributions).real, 1.0, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
