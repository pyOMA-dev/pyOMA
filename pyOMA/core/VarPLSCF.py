# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Variance estimation for the pLSCF method.

Propagates the uncertainty of the positive half-spectra onto the identified modal
parameters by the delta method, following Steffensen, Döhler, Tcherniak & Thomsen
(MSSP 223, 2025), "Variance estimation of modal parameters from the poly-reference
least-squares complex frequency-domain algorithm".

The paper enters the propagation with the covariance of measured FRFs, obtained
from the H1 estimator and the multiple coherence (their Eq. 26-27). pyOMA is
output-only, so no input spectrum and hence no coherence exists, and that entry
point has no counterpart here. Instead the half-spectra are re-estimated block-wise
from the block correlation functions that :meth:`~pyOMA.core.PreProcessingTools.PreProcessSignals.corr_blackman_tukey`
already provides, and their sample covariance over blocks is used. This is the one
deliberate deviation from the paper's derivation; all Jacobians downstream are the
paper's.

The covariance is carried as a *factor* rather than a matrix: the block deviations
are scaled such that ``cov(h) = B B^T`` for the re/im-stacked slice ``B`` of
:attr:`VarPLSCF.spec_block_factor`, at the scale of the covariance *of the estimate*
(the role the paper's ``1/N_avg`` plays). Every downstream perturbation is then
evaluated per block and variances follow as ``einsum('ij,ij->i', U, U)``, mirroring
:mod:`~pyOMA.core.VarSSIRef`. A consequence worth noting: this retains the
correlations across frequency lines and output channels that the paper's Eq. 40
discards for tractability -- with an empirical covariance, imposing that
independence would be extra work rather than less.
"""

import dataclasses
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from .PLSCF import PLSCF
from .Helpers import simplePbar


@dataclasses.dataclass
class Stage1Scores:
    """First-order perturbations of the Stage-1 quantities, one column per block.

    Every array carries the block index as its last axis: column *j* is the
    estimator's linearisation applied to the deviation of block *j*, scaled such
    that contracting the array with itself over that axis yields a variance.

    Attributes
    ----------
    U_theta : np.ndarray
        Free denominator coefficients, shape ``(order * n_r, n_r, n_b)``, real.
    d_lambda : np.ndarray
        Continuous-time eigenvalues, shape ``(n_modes, n_b)``, complex.
    d_frequency, d_damping : np.ndarray
        Modal frequencies and damping ratios, shape ``(n_modes, n_b)``, real.
    d_participation : np.ndarray
        Normalised participation vectors, shape ``(n_r, n_modes, n_b)``, complex.

    Modes are ordered as :meth:`~pyOMA.core.PLSCF.PLSCF.modal_analysis_residuals`
    returns them.
    """
    U_theta: np.ndarray
    d_lambda: np.ndarray
    d_frequency: np.ndarray
    d_damping: np.ndarray
    d_participation: np.ndarray


class VarPLSCF(PLSCF):
    """pLSCF identification with uncertainty quantification.

    Extends :class:`~pyOMA.core.PLSCF.PLSCF` with standard deviations of the
    identified modal parameters.  The workflow matches the base class, except that
    :meth:`build_half_spectra` must be given *num_blocks*:

    1. :meth:`build_half_spectra` — half-spectra plus their block-covariance factor.
    2. :meth:`compute_modal_params` — modal parameters and their standard deviations.

    Attributes
    ----------
    spec_block_factor : np.ndarray or None
        Covariance factor of the positive half-spectra, complex, shape
        ``(n_l, n_r, num_omega, n_b)`` with the block index last.  Column *j* holds
        the deviation of block *j* from the block mean, scaled by
        ``1 / sqrt(n_b * (n_b - 1))``, so that contracting the re/im-stacked slice
        with itself yields the covariance of the mean half-spectrum.

        Not persisted by :meth:`save_state`: it is reconstructible from
        ``prep_signals`` and would dominate the archive size.
    std_frequencies, std_damping : np.ndarray or None
        Standard deviations of the identified modal frequencies and damping
        ratios, shaped like :attr:`~pyOMA.core.PLSCF.PLSCF.modal_frequencies`.
    std_participation_vectors : np.ndarray or None
        Standard deviations of the participation vectors, shaped like
        :attr:`~pyOMA.core.PLSCF.PLSCF.participation_vectors`; the standard
        deviation of the real part is stored in ``.real`` and that of the
        imaginary part in ``.imag``, as in
        :class:`~pyOMA.core.VarSSIRef.VarSSIRef`.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs
            Passed to :class:`~pyOMA.core.PLSCF.PLSCF`.
        """
        super().__init__(*args, **kwargs)
        self.spec_block_factor = None

        self.std_frequencies = None
        self.std_damping = None
        self.std_participation_vectors = None

    def build_half_spectra(self, nperseg=None,
                           begin_frequency=None, end_frequency=None,
                           window_decay=0.001, num_blocks=None, training_blocks=None, **kwargs):
        '''
        Construct the positive half-spectra and their block-covariance factor.

        Behaves as :meth:`~pyOMA.core.PLSCF.PLSCF.build_half_spectra`, but
        *num_blocks* is required: the variance of the half-spectra is estimated
        from the scatter of the individual blocks about their mean, and that mean
        is the half-spectrum the model is identified from.

        Parameters
        ----------
            nperseg: integer, optional
                Number of (positive) frequency lines to consider (rfft)

            begin_frequency, end_frequency: float, optional
                Frequency range to restrict the identified system.

            window_decay: float, (0,1)
                Final value of the exponential window, that is applied to the
                correlation functions.

            num_blocks: integer, required
                The number of blocks to split the signal into. At least two
                blocks are needed for a variance estimate; the estimate becomes
                meaningful only for substantially more.

            training_blocks: list, optional
                The selected blocks to use for system identification
                (=training). Defaults to all blocks. The covariance factor is
                built from these blocks only, i.e. from the blocks that entered
                the point estimate.

        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to prep_signals.correlation
        '''
        if num_blocks is None:
            raise ValueError(
                'Argument num_blocks is required for variance estimation: the '
                'half-spectrum covariance is estimated from the scatter of the '
                'individual blocks. Use PLSCF for an identification without it.'
            )

        super().build_half_spectra(
            nperseg, begin_frequency, end_frequency, window_decay,
            num_blocks, training_blocks, **kwargs)

        self.spec_block_factor = self._build_spec_block_factor()

    def _build_spec_block_factor(self):
        """Build the covariance factor of the positive half-spectra.

        Returns
        -------
            spec_block_factor: (n_l, n_r, num_omega, n_b) numpy.ndarray
                Scaled block deviations; see :attr:`VarPLSCF.spec_block_factor`.
        """
        training_blocks = self.training_blocks
        n_b = training_blocks.shape[0]
        if n_b < 2:
            raise ValueError(
                f'At least two training blocks are needed to estimate a variance, got {n_b}.')
        if n_b < 10:
            logger.warning(
                f'Estimating the half-spectrum covariance from only {n_b} blocks; '
                f'the resulting standard deviations are themselves highly uncertain.')

        corr_blocks = self.prep_signals.corr_matrices_bt[training_blocks, ..., :self.nperseg]

        # the windowed rFFT is linear, so the mean of the block spectra is exactly
        # the half-spectrum the model was identified from
        _, block_spectra, _ = self._windowed_half_spectrum(
            corr_blocks, self.nperseg, self.window_decay,
            self.begin_frequency, self.end_frequency)  # (n_b, n_l, n_r, num_omega)

        deviations = block_spectra - np.mean(block_spectra, axis=0, keepdims=True)
        deviations /= np.sqrt(n_b * (n_b - 1))

        return np.moveaxis(deviations, 0, -1)

    def _theta_scores(self, ctx):
        """Perturbations of the free denominator coefficients, per block.

        pyOMA constrains the *last* denominator block, so the estimating equation
        of the reduced normal equations is ``[M theta]_{:n*n_r} = 0`` with
        ``theta = [theta_b; I_n_r]``, and the free coefficients are already the
        companion-matrix coefficients -- the paper's Appendix B transformation has
        no counterpart here.  Differentiating that equation gives

        .. math:: M_{aa} \\Delta\\theta_b = -[\\Delta M \\theta]_{:n \\cdot n_r}

        Only ``Y_o`` depends on the half-spectra, and linearly so.  Writing the
        assembly out with ``Z_o = X_o Q_o - Y_o theta`` (the negated equation error
        of output *o*) and ``V_o = RS_o^T X_o^H - Y_o^H`` collapses the four terms
        of the product rule into

        .. math::
            -[\\Delta M \\theta] = 2 \\sum_o \\left[
                \\Re(\\Delta Y_o^H Z_o) + \\Re(V_o \\Delta Y_o \\theta) \\right]

        The Kronecker structure of ``Delta Y_o`` is never materialised: it reduces
        every product to a contraction of ``X_o`` with the block deviations.

        Parameters
        ----------
            ctx: NormalEquationsContext
                Assembly to differentiate, from
                :meth:`~pyOMA.core.PLSCF.PLSCF._assemble_normal_equations`.

        Returns
        -------
            U_theta: (order * n_r, n_r, n_b) numpy.ndarray
                Perturbation of ``alpha_b``, one column per block.
        """
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        order = ctx.order
        num_omega = self.num_omega
        n_b = self.spec_block_factor.shape[-1]

        X_o = ctx.X_o
        X_o_conj = np.conj(X_o)
        X_o_H = X_o_conj.T
        # the polynomial evaluated at every frequency line: alpha(Omega_f)
        alpha_omega = np.einsum('fp,pqs->fqs', X_o,
                                ctx.alpha.reshape(order + 1, n_r, n_r), optimize=True)

        rhs = np.zeros(((order + 1) * n_r, n_r, n_b))
        for i_l in range(n_l):
            H_o = self.pos_half_spectra[i_l,:,:]  # (n_r, num_omega)
            D_o = self.spec_block_factor[i_l,:,:,:]  # (n_r, num_omega, n_b)

            # Y_o[f, p * n_r + q] = -X_o[f, p] * H_o[q, f]; the kron of estimate_model
            Y_o = (-X_o[:,:, np.newaxis] * H_o.T[:, np.newaxis,:]).reshape(num_omega, -1)
            RS_o = ctx.RS_solutions[:,:, i_l]
            # R_o^-1 S_o theta, already available as the numerator coefficients
            Q_o = -ctx.beta_l_i[:,:, i_l]

            Z_o = X_o @ Q_o - Y_o @ ctx.alpha  # (num_omega, n_r)
            V_o = RS_o.T @ X_o_H - np.conj(Y_o.T)  # ((order + 1) * n_r, num_omega)

            # (Delta Y_o theta)[f, s] = -sum_q Delta H_o[q, f] alpha_omega[f, q, s]
            dY_theta = -np.einsum('qfj,fqs->fsj', D_o, alpha_omega, optimize=True)

            # (Delta Y_o^H Z_o)[p * n_r + q, s] =
            #     -sum_f conj(X_o[f, p]) conj(Delta H_o[q, f]) Z_o[f, s]
            term_a = -np.einsum('fp,qfj,fs->pqsj', X_o_conj, np.conj(D_o), Z_o,
                                optimize=True)
            term_b = np.einsum('pf,fsj->psj', V_o, dY_theta, optimize=True)

            rhs += np.real(term_a).reshape((order + 1) * n_r, n_r, n_b) + np.real(term_b)

        # the factor 2 that estimate_model applies to the whole sum over outputs;
        # it cancels against M_aa, but keeping both consistent avoids a trap
        rhs *= 2

        U_theta = np.linalg.solve(ctx.M_aa, rhs[:order * n_r,:,:].reshape(order * n_r, -1))
        return U_theta.reshape(order * n_r, n_r, n_b)

    @staticmethod
    def _companion_column_scores(U_theta, order, n_r):
        """Perturbation of the only varying part of the companion matrix, per block.

        ``A_c``'s block superdiagonal is a constant identity, and its first block
        column holds the free denominator blocks reversed and negated: the
        block-solve in
        :meth:`~pyOMA.core.PLSCF.PLSCF._build_companion_matrix_residuals` divides
        by the *constrained* block ``alpha[-n_r:] = I``, which is exact and
        therefore contributes nothing.  Reversing the blocks by slicing is the
        paper's permutation matrix, without building it.

        Returns
        -------
            dA_c1: (order * n_r, n_r, n_b) numpy.ndarray
                Perturbation of ``A_c[:, :n_r]``, one column per block.
        """
        n_b = U_theta.shape[-1]
        blocks = U_theta.reshape(order, n_r, n_r, n_b)[::-1,:,:,:]
        return -blocks.reshape(order * n_r, n_r, n_b)

    def _pole_scores(self, dA_c1, modal_ctx):
        """Perturbations of the continuous-time eigenvalues, per block.

        Uses the standard first-order eigenvalue perturbation
        ``dz_r = y_r^H dA_c x_r / (y_r^H x_r)``; since only the first block column
        of ``dA_c`` is nonzero, only the first ``n_r`` entries of ``x_r``
        contribute.  Both the formula and ``lambda = log(z) f_s`` are invariant to
        the arbitrary scaling of the eigenvectors.

        Returns
        -------
            d_lambda: (n_modes, n_b) numpy.ndarray
                Complex perturbation of the eigenvalues, in the order in which
                :meth:`~pyOMA.core.PLSCF.PLSCF.modal_analysis_residuals` returns
                the modes.
        """
        sampling_rate = self.prep_signals.sampling_rate
        n_r = dA_c1.shape[1]
        inds = modal_ctx.mode_indices

        eigvecs_r = modal_ctx.eigvecs_r[:, inds]
        eigvecs_l = modal_ctx.eigvecs_l[:, inds]
        eigvals_z = modal_ctx.eigvals_z[inds]

        denominator = np.einsum('kr,kr->r', np.conj(eigvecs_l), eigvecs_r, optimize=True)
        numerator = np.einsum('kr,kmj,mr->rj', np.conj(eigvecs_l), dA_c1,
                              eigvecs_r[:n_r,:], optimize=True)
        d_z = numerator / denominator[:, np.newaxis]

        return d_z * sampling_rate / eigvals_z[:, np.newaxis]

    def _frequency_damping_scores(self, eigenvalues, d_lambda):
        """Perturbations of modal frequencies and damping ratios, per block.

        pyOMA defines ``f = |lambda| / 2 pi`` and damping in percent with the
        exponential-window correction (see
        :meth:`~pyOMA.core.PLSCF.PLSCF._compute_damping`), rather than the paper's
        Eq. 14; these Jacobians are derived for pyOMA's convention and replace the
        paper's Eq. 42.  With ``2 pi f = |lambda|`` the damping ratio is
        ``zeta% = -100 (Re lambda - a f_s) / |lambda|``.

        Returns
        -------
            d_frequency, d_damping: (n_modes, n_b) numpy.ndarray
        """
        sampling_rate = self.prep_signals.sampling_rate
        # _compute_damping applies no correction at all if factor_a is None,
        # which is what a vanishing correction term amounts to
        factor_a = self.factor_a if self.factor_a is not None else 0.0

        abs_lambda = np.abs(eigenvalues)[:, np.newaxis]
        lambda_re = np.real(eigenvalues)[:, np.newaxis]
        lambda_im = np.imag(eigenvalues)[:, np.newaxis]

        d_abs = (lambda_re * np.real(d_lambda)
                 + lambda_im * np.imag(d_lambda)) / abs_lambda
        d_frequency = d_abs / (2 * np.pi)
        d_damping = -100 * (np.real(d_lambda) / abs_lambda
                            - (lambda_re - factor_a * sampling_rate)
                            / abs_lambda ** 2 * d_abs)

        return d_frequency, d_damping

    def _participation_scores(self, dA_c1, modal_ctx):
        """Perturbations of the normalised participation vectors, per block.

        The participation vectors are the last ``n_r`` entries of the left
        eigenvectors of ``A_c``.  Perturbing ``A_c^H y_r = conj(z_r) y_r`` gives

        .. math:: (A_c^H - \\bar{z}_r I) \\Delta y_r =
                  \\overline{\\Delta z_r} y_r - \\Delta A_c^H y_r

        whose right-hand side is the projector of the paper's Eq. 46 applied to
        ``-Delta A_c^H y_r``.  The system determines ``Delta y_r`` only up to a
        multiple of ``y_r``; the pseudo-inverse picks the component orthogonal to
        it, which is immaterial because the normalisation below is invariant to a
        scaling of ``y_r``.

        Returns
        -------
            d_participation: (n_r, n_modes, n_b) numpy.ndarray
                Complex perturbation of the normalised participation vectors.
        """
        n_r = dA_c1.shape[1]
        n_b = dA_c1.shape[-1]
        inds = modal_ctx.mode_indices

        A_c_H = np.conj(modal_ctx.A_c.T)
        identity = np.eye(A_c_H.shape[0])

        d_participation = np.zeros((n_r, len(inds), n_b), dtype=complex)
        for i, ind in enumerate(inds):
            z_r = modal_ctx.eigvals_z[ind]
            x_r = modal_ctx.eigvecs_r[:, ind]
            y_r = modal_ctx.eigvecs_l[:, ind]

            projector = identity - np.outer(y_r, np.conj(x_r)) / (np.conj(x_r) @ y_r)
            G_r = -np.linalg.pinv(A_c_H - np.conj(z_r) * identity) @ projector

            # dA_c is real and only its first block column is nonzero, so
            # dA_c^H y_r is nonzero only in its first n_r entries
            rhs = np.einsum('kmj,k->mj', dA_c1, y_r, optimize=True)
            d_L = (G_r[:, :n_r] @ rhs)[-n_r:,:]

            # normalisation to the largest component, c.p. paper Eq. 48/49; that
            # component is exactly one, so its perturbation -- and hence its
            # standard deviation -- is exactly zero
            L_r = y_r[-n_r:]
            k = np.argmax(np.abs(L_r))
            L_tilde = L_r / L_r[k]
            d_participation[:, i,:] = (
                d_L - L_tilde[:, np.newaxis] * d_L[k,:][np.newaxis,:]) / L_r[k]

        return d_participation

    def _stage1_scores(self, ctx, modal_ctx, eigenvalues):
        """Run the Stage-1 propagation for a single model order.

        Parameters
        ----------
            ctx: NormalEquationsContext
                Assembly of the reduced normal equations at this order.
            modal_ctx: ModalContext
                Eigendecomposition and mode selection at this order.
            eigenvalues: (n_modes,) numpy.ndarray
                Continuous-time eigenvalues, as returned by
                :meth:`~pyOMA.core.PLSCF.PLSCF.modal_analysis_residuals`.

        Returns
        -------
            scores: Stage1Scores
        """
        n_r = self.prep_signals.num_ref_channels

        U_theta = self._theta_scores(ctx)
        dA_c1 = self._companion_column_scores(U_theta, ctx.order, n_r)
        d_lambda = self._pole_scores(dA_c1, modal_ctx)
        d_frequency, d_damping = self._frequency_damping_scores(eigenvalues, d_lambda)
        d_participation = self._participation_scores(dA_c1, modal_ctx)

        return Stage1Scores(U_theta=U_theta, d_lambda=d_lambda,
                            d_frequency=d_frequency, d_damping=d_damping,
                            d_participation=d_participation)

    def compute_modal_params(self, max_model_order, complex_coefficients=False,
                             algo='residuals', modal_contrib=None, validation_blocks=None):
        '''
        Perform a multi-order computation of modal parameters and their standard
        deviations.

        Behaves as :meth:`~pyOMA.core.PLSCF.PLSCF.compute_modal_params`, but
        additionally propagates the block-covariance of the half-spectra onto the
        identified modal parameters at every order.

        Parameters
        ----------
            max_model_order: integer
                Maximum model order, where to interrupt the algorithm.
            complex_coefficients: bool, optional
                Must be False: the propagation is derived for real coefficients.
            algo: str, optional
                Must be 'residuals': the propagation differentiates the
                residual-based modal analysis.
            modal_contrib: bool, optional
                Synthesize modal spectra and estimate modal contributions.
            validation_blocks: list, optional
                Forwarded to :meth:`~pyOMA.core.PLSCF.PLSCF.synthesize_spectrum`.
        '''
        if complex_coefficients:
            raise NotImplementedError(
                'Variance estimation is derived for real denominator coefficients; '
                'call PLSCF.compute_modal_params for complex_coefficients=True.')
        if algo != 'residuals':
            raise NotImplementedError(
                "Variance estimation requires algo='residuals': the propagation "
                'differentiates the residual-based modal analysis, which is also '
                'the only one identifying participation vectors.')
        if not self.state[0]:
            raise RuntimeError("Call build_half_spectra() first.")
        if self.spec_block_factor is None:
            # e.g. a state loaded from an archive, which does not carry the factor
            raise RuntimeError(
                'The half-spectrum covariance factor is missing; call '
                'build_half_spectra() on this object to (re)build it.')

        algo, modal_contrib = self._setup_compute_params(
            max_model_order, algo, modal_contrib
        )
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels

        logger.info('Computing modal parameters and their standard deviations...')
        max_modes = max_model_order * n_r // 2
        modal_frequencies = np.zeros((max_model_order, max_modes))
        modal_damping = np.zeros((max_model_order, max_modes))
        mode_shapes = np.zeros((n_l, max_modes, max_model_order), dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_modes), dtype=complex)
        modal_contributions = np.zeros(
            (max_model_order, max_modes,), dtype=complex) if modal_contrib else None
        participation_vectors = np.zeros((n_r, max_modes, max_model_order), dtype=complex)

        std_frequencies = np.zeros((max_model_order, max_modes))
        std_damping = np.zeros((max_model_order, max_modes))
        std_participation_vectors = np.zeros(
            (n_r, max_modes, max_model_order), dtype=complex)

        pbar = simplePbar(max_model_order)
        for order in range(1, max_model_order):
            next(pbar)
            # the assembly is needed in full, so estimate_model is bypassed
            ctx = self._assemble_normal_equations(order, complex_coefficients)
            f, d, phi, lamda = self.modal_analysis_residuals(ctx.alpha, ctx.beta_l_i)
            n_modes = len(f)
            scores = self._stage1_scores(ctx, self._modal_ctx, lamda)

            if modal_contrib:
                _, delta = self.synthesize_spectrum(
                    ctx.alpha, ctx.beta_l_i, True, validation_blocks=validation_blocks)
                modal_contributions[order, :n_modes] = delta
            modal_frequencies[order, :n_modes] = f
            modal_damping[order, :n_modes] = d
            eigenvalues[order, :n_modes] = lamda
            mode_shapes[:, :n_modes, order] = phi
            participation_vectors[:, :n_modes, order] = self._participation_vectors

            std_frequencies[order, :n_modes] = np.sqrt(
                np.einsum('rj,rj->r', scores.d_frequency, scores.d_frequency))
            std_damping[order, :n_modes] = np.sqrt(
                np.einsum('rj,rj->r', scores.d_damping, scores.d_damping))
            # real and imaginary parts are propagated jointly, then split as in
            # VarSSIRef: std of the real part into .real, of the imaginary into .imag
            U_L = np.concatenate([np.real(scores.d_participation),
                                  np.imag(scores.d_participation)], axis=0)
            var_L = np.einsum('crj,crj->cr', U_L, U_L)
            std_participation_vectors.real[:, :n_modes, order] = np.sqrt(var_L[:n_r,:])
            std_participation_vectors.imag[:, :n_modes, order] = np.sqrt(var_L[n_r:,:])

        self.max_model_order = max_model_order
        self.eigenvalues = eigenvalues
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.modal_contributions = modal_contributions
        self.participation_vectors = participation_vectors

        self.std_frequencies = std_frequencies
        self.std_damping = std_damping
        self.std_participation_vectors = std_participation_vectors
        self.state[1] = True
