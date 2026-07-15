# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Poly-reference Least-Squares Complex Frequency (pLSCF) identification method."""

import dataclasses
import numpy as np
import os
import scipy.signal
import scipy.linalg
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .Helpers import validate_array, simplePbar, ConfigFile


@dataclasses.dataclass
class NormalEquationsContext:
    """Assembly of the reduced normal equations at a single model order.

    Holds the intermediate quantities that :meth:`PLSCF.estimate_model` discards,
    so that uncertainty propagation can differentiate the assembly without
    duplicating it.

    Attributes
    ----------
    order : int
        Model order this assembly was built at.
    X_o : np.ndarray
        Polynomial basis, shape ``(num_omega, order + 1)``.
    RS_solutions : np.ndarray
        Per-output-channel ``R_o^-1 S_o``, shape ``(order + 1, (order + 1) * n_r, n_l)``.
    M : np.ndarray
        Reduced normal equations, shape ``((order + 1) * n_r, (order + 1) * n_r)``.
    M_aa : np.ndarray
        The block of *M* that is solved for the free denominator coefficients,
        ``M[:order * n_r, :order * n_r]`` (a view).
    alpha, beta_l_i : np.ndarray
        Denominator and numerator coefficients; see :meth:`PLSCF.estimate_model`.
    """
    order: int
    X_o: np.ndarray
    RS_solutions: np.ndarray
    M: np.ndarray
    M_aa: np.ndarray
    alpha: np.ndarray
    beta_l_i: np.ndarray


@dataclasses.dataclass
class ModalContext:
    """Companion-matrix eigendecomposition and the mode selection applied to it.

    Attributes
    ----------
    A_c : np.ndarray
        Transposed companion matrix, shape ``(order * n_r, order * n_r)``.
    eigvals_z : np.ndarray
        Discrete-time eigenvalues remaining after :meth:`~pyOMA.core.ModalBase.ModalBase.remove_conjugates`.
    eigvecs_l, eigvecs_r : np.ndarray
        Left and right eigenvectors of *A_c*, column-aligned with *eigvals_z*.
    mode_indices : np.ndarray
        Columns of the eigen-arrays corresponding to the returned modes, in the
        returned (in-band, frequency-sorted) order.  Lets derived quantities
        reproduce the identical mode selection and ordering.
    """
    A_c: np.ndarray
    eigvals_z: np.ndarray
    eigvecs_l: np.ndarray
    eigvecs_r: np.ndarray
    mode_indices: np.ndarray


@dataclasses.dataclass
class LSFDContext:
    """Assembly of the least-squares frequency-domain fit for the mode shapes.

    Holds the intermediate quantities that :meth:`PLSCF._fit_mode_shapes_ls`
    discards, so that uncertainty propagation can differentiate the fit without
    duplicating it.

    Attributes
    ----------
    A : np.ndarray
        Real design matrix, shape ``(num_omega * 2 * n_r, 2 * n_modes + 4 * n_r)``.
    X : np.ndarray
        Least-squares solution ``pinv(A) @ h``, shape ``(2 * n_modes + 4 * n_r, n_l)``.
    pinv_A : np.ndarray
        Pseudo-inverse of *A*.  Note ``pinv_A @ pinv_A.T`` is ``(A^T A)^-1`` for a
        full-column-rank *A*, which saves forming and inverting the normal
        equations of this stage a second time.
    residual : np.ndarray
        ``h - A @ X``, shape ``(num_omega * 2 * n_r, n_l)``.  Only the residual is
        retained; *h* itself is a reordering of ``pos_half_spectra``.
    Df1, Df2 : np.ndarray
        Pole factors ``1 / (1j * omega - lambda)`` and its conjugate-pole
        counterpart, for every frequency line, shape ``(num_omega, n_modes)``.
    eigenvalues : np.ndarray
        Continuous-time poles the fit was built on, shape ``(n_modes,)``.
    participation_vectors : np.ndarray
        Normalised participation vectors the fit was built on, shape
        ``(n_r, n_modes)``.  Held together with *eigenvalues* so that the context
        fully describes the fit; both are in mode-column order, not returned order.
    mode_order : np.ndarray
        Permutation relating the mode columns of *A* to the returned modes:
        returned mode ``k`` occupies mode column ``mode_order[k]``.  The fit is
        run in the order the poles were passed in, which is *not* the returned
        frequency-sorted order.  Assigned by :meth:`PLSCF.modal_analysis_residuals`.
    """
    A: np.ndarray
    X: np.ndarray
    pinv_A: np.ndarray
    residual: np.ndarray
    Df1: np.ndarray
    Df2: np.ndarray
    eigenvalues: np.ndarray
    participation_vectors: np.ndarray
    mode_order: np.ndarray = None


class PLSCF(ModalBase):
    """Poly-reference Least-Squares Complex Frequency (pLSCF) method.

    Also known as PolyMAX.  Identifies modal parameters from positive
    half-spectra derived from correlation functions.  The standard workflow is:

    1. :meth:`build_half_spectra` — construct the positive half-spectra.
    2. :meth:`compute_modal_params` — run the multi-order identification.
    3. Pass the result to :class:`~pyOMA.core.StabilDiagram.StabilCalc` for
       stabilisation-diagram analysis.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object providing correlation functions and
        channel metadata.

    .. TODO::
        * Test functions should be added to the test package
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs
            Passed to :class:`~pyOMA.core.ModalBase.ModalBase`.
        """
        super().__init__(*args, **kwargs)
        self.state = [False, False]

        self.begin_frequency = None
        self.end_frequency = None
        self.nperseg = None
        self.factor_a = None
        self.selected_omega_vector = None
        self.pos_half_spectra = None

        self.num_blocks = None
        self.training_blocks = None
        self.window_decay = None
        self.weights = None
        self.n_eff = None
        self.corr_matrices = None

        self.participation_vectors = None

        self._lower_residuals = None
        self._upper_residuals = None
        self._mode_shapes_raw = None
        self._participation_vectors = None
        self._eigenvalues = None
        self._modal_ctx = None
        self._lsfd_ctx = None

        self._half_spec_synth = None
        self.modal_contributions = None

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        cfg = ConfigFile(conf_file)
        begin_frequency = cfg.float('Begin Frequency')
        end_frequency = cfg.float('End Frequency')
        nperseg = cfg.int('Samples per time segment')
        max_model_order = cfg.int('Maximum Model Order')

        pLSCF_object = cls(prep_signals)
        pLSCF_object.build_half_spectra(nperseg, begin_frequency, end_frequency)
        pLSCF_object.compute_modal_params(max_model_order)

        return pLSCF_object

    def write_config(self, conf_file):
        ConfigFile.write(conf_file, {
            'Begin Frequency': self.begin_frequency,
            'End Frequency': self.end_frequency,
            'Samples per time segment': self.nperseg,
            'Maximum Model Order': self.max_model_order,
        })

    @staticmethod
    def _coerce_freq_bound(value, name, lo, hi, none_default):
        """Coerce a frequency bound to float within [lo, hi].

        Returns none_default when value is None; clips at lo or hi when out of range.
        """
        if value is None:
            return none_default
        if value < lo:
            return lo
        if value > hi:
            return hi
        if isinstance(value, int):
            return float(value)
        if not isinstance(value, float):
            raise TypeError(f"{name} must be float, got {type(value).__name__!r}")
        return value

    def _validate_nperseg(self, nperseg):
        """Resolve nperseg from precomputed data or validate the given integer."""
        if nperseg is None:
            if self.prep_signals.m_lags is not None:
                return self.prep_signals.m_lags
            if self.prep_signals.n_lines is not None:
                return self.prep_signals.n_lines // 2 + 1
            raise RuntimeError(
                'Argument nperseg or precomputed spectra/correlations must be provided.'
            )
        if not isinstance(nperseg, int):
            raise TypeError(f"nperseg must be int, got {type(nperseg).__name__!r}")
        return nperseg

    def _validate_frequency_params(self, nperseg, begin_frequency, end_frequency):
        """Validate and normalize frequency parameters for build_half_spectra."""
        nyquist = self.prep_signals.sampling_rate / 2
        begin_frequency = self._coerce_freq_bound(
            begin_frequency, 'begin_frequency', 0.0, nyquist, 0.0
        )
        end_frequency = self._coerce_freq_bound(
            end_frequency, 'end_frequency', 0.0, nyquist, nyquist
        )
        return begin_frequency, end_frequency, self._validate_nperseg(nperseg)

    @staticmethod
    def _coerce_blocks_array(blocks, num_blocks, name):
        """Validate and coerce a blocks argument to a numpy array."""
        if blocks is None:
            return np.arange(num_blocks)
        if isinstance(blocks, (list, tuple)):
            blocks = np.array(blocks)
        elif not isinstance(blocks, np.ndarray):
            raise RuntimeError(f"Argument {name!r} must be an iterable but is type {type(blocks)}")
        if blocks.max() >= num_blocks:
            raise ValueError(f"{name}.max() must be < {num_blocks}, got {blocks.max()}.")
        return blocks

    @staticmethod
    def _validate_weights(weights, num_blocks):
        """Normalise per-block weights and report their effective sample size.

        Parameters
        ----------
            weights: (num_blocks,) array_like or None
                Non-negative per-block weights, in the order of
                :attr:`training_blocks`. Renormalised to sum to one; their scale
                therefore carries no meaning. *None* requests uniform weighting.
            num_blocks: integer
                Number of blocks the weights must address. This is the number of
                *training* blocks, not necessarily :attr:`num_blocks`.

        Returns
        -------
            weights: (num_blocks,) numpy.ndarray or None
                The renormalised weights; *None* is passed through, so that
                callers can distinguish "uniform" from "uniform by request".
            n_eff: float
                Kish's effective sample size ``1 / sum(w^2)``, which equals
                *num_blocks* for uniform weights and drops towards one as the
                weight mass concentrates on fewer blocks.
        """
        if weights is None:
            return None, float(num_blocks)

        weights = np.asarray(weights)
        if np.iscomplexobj(weights):
            # numpy would silently discard the imaginary part on the cast below.
            # Weights scale correlation functions here, and in
            # :class:`~pyOMA.core.VarPLSCF.VarPLSCF` they build a weighting matrix
            # that must stay real for the conjugate-linear parts of its score chain.
            raise ValueError('weights must be real.')
        weights = weights.astype(float)

        if weights.ndim != 1 or weights.shape[0] != num_blocks:
            raise ValueError(
                f'weights must be a one-dimensional array of length {num_blocks} '
                f'(one per training block), got shape {weights.shape}.')
        if not np.all(np.isfinite(weights)):
            raise ValueError('weights must all be finite.')
        if np.any(weights < 0):
            raise ValueError(
                f'weights must be non-negative, got a minimum of {weights.min()}.')

        total = weights.sum()
        if total <= 0:
            raise ValueError('weights must not be all-zero: they are renormalised to sum to one.')

        weights = weights / total
        n_eff = 1.0 / np.sum(weights ** 2)
        return weights, n_eff

    def _validate_corr_matrices(self, corr_matrices, num_blocks, nperseg):
        """Validate an externally supplied array of block correlation functions."""
        corr_matrices = np.asarray(corr_matrices)
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels

        if corr_matrices.ndim != 4:
            raise ValueError(
                f'corr_matrices must have shape (num_blocks, {n_l}, {n_r}, >= nperseg), '
                f'got {corr_matrices.ndim} dimensions.')
        if corr_matrices.shape[0] != num_blocks:
            raise ValueError(
                f'corr_matrices must hold num_blocks={num_blocks} blocks on its first '
                f'axis, got {corr_matrices.shape[0]}.')
        if corr_matrices.shape[1:3] != (n_l, n_r):
            raise ValueError(
                f'corr_matrices must hold ({n_l}, {n_r}) channel pairs on axes 1 and 2, '
                f'got {corr_matrices.shape[1:3]}.')
        if corr_matrices.shape[3] < nperseg:
            raise ValueError(
                f'corr_matrices must span at least nperseg={nperseg} lags on its last '
                f'axis, got {corr_matrices.shape[3]}.')
        return corr_matrices

    def _block_correlations(self, blocks):
        """The block correlation functions that fed the estimate, for *blocks*.

        Reads whichever source :meth:`build_half_spectra` was given: an external
        *corr_matrices* array, or the block-wise Blackman-Tukey estimate cached in
        ``prep_signals``. All consumers of the block correlations go through here,
        so that both sources stay interchangeable.

        Returns
        -------
            corr_blocks: (len(blocks), n_l, n_r, nperseg) numpy.ndarray
        """
        if self.num_blocks is None:
            raise RuntimeError(
                'No block correlations exist: call build_half_spectra() with num_blocks.')
        source = self.corr_matrices
        if source is None:
            source = self.prep_signals.corr_matrices_bt
        return source[blocks, ..., :self.nperseg]

    def _windowed_half_spectrum(self, correlation_matrix, nperseg, window_decay, begin_frequency, end_frequency):
        """Apply the exponential window, rFFT, and frequency-range selection shared by
        build_half_spectra's own construction and the cross-validation reconstruction
        in synthesize_spectrum."""
        tau = -nperseg / np.log(window_decay)
        win = scipy.signal.windows.get_window(('exponential', 0, tau), nperseg, fftbins=True)
        factor_a = -1 / tau
        psd_matrix = np.fft.rfft(correlation_matrix * win)

        sampling_rate = self.prep_signals.sampling_rate
        freqs = np.fft.rfftfreq(nperseg, 1 / sampling_rate)
        freq_inds = (freqs > begin_frequency) & (freqs < end_frequency)
        selected_omega_vector = freqs[freq_inds] * 2 * np.pi
        spectrum_tensor = psd_matrix[..., freq_inds]
        return selected_omega_vector, spectrum_tensor, factor_a

    def build_half_spectra(self, nperseg=None,
                           begin_frequency=None, end_frequency=None,
                           window_decay=0.001, num_blocks=None, training_blocks=None,
                           weights=None, corr_matrices=None, **kwargs):
        '''
        Extracts an array of positive half spectra between begin_frequency
        and end_frequency from a spectrum of nperseg frequency lines. If
        begin_frequency > 0.0 or end_frequency<nyquist freqeuncy, the resulting
        array has less than nperseg lines.

        Positive power spectra are constructed from positive correlation functions,
        that are windowed by an exponential window and transformed to frequency
        domain by and (R)FFT.
        Correlation functions are computed in prep_signals by either
        Welch's or Blackman-Tukey's method, though, Welch's method is not
        recommmended, because the artificial  damping introduced by windowing
        can not be corrected.

        See: Cauberghe-2004-Applied Frequency-Domain System ... : Sections 3.4ff

        Note: The previous implementation contained severe mistakes in the computation
        of positive power spectra, e.g. doubled squaring of spectral values, lazy handling
        of array dimensions and therefore effectively only a quarter of nperseg being used
        as well as numerical inefficiencies.

        .. TODO::
          * Move spectral estimation into prep_signals.pds_blackman_tukey and only
            keep bandwidth selection and argument checking here
          * Allow other windows than exponential


        Parameters
        ----------
            nperseg: integer, optional
                Number of (positive) frequency lines to consider (rfft)

            begin_frequency, end_frequency: float, optional
                Frequency range to restrict the identified system.

            window_decay: float, (0,1)
                Final value of the exponential window, that is applied to the
                correlation functions.

            num_blocks: integer, optional
                The number of blocks to split the signal into for
                cross-validation. If given, correlation functions are
                (re-)estimated block-wise via
                ``prep_signals.corr_blackman_tukey(nperseg, n_segments=num_blocks, refs_only=True)``
                and only *training_blocks* are averaged into the half-spectrum;
                the remaining blocks are then available for
                :meth:`synthesize_spectrum`/:meth:`compute_modal_params` via
                their *validation_blocks* argument. If not given (default),
                behaviour is unchanged: whatever correlation function is
                already cached in ``prep_signals`` (Welch or Blackman-Tukey,
                full signal) is used.

            training_blocks: list, optional
                The selected blocks to use for system identification
                (=training). Only meaningful together with *num_blocks*.
                Defaults to all blocks.

            weights: (len(training_blocks),) array_like, optional
                Non-negative weights, one per *training block*, in the order of
                *training_blocks* -- not per block of *num_blocks*. The two
                coincide in the default case, where all blocks train. The
                half-spectrum is then built from the weighted mean
                ``sum_j w_j R_j`` of the block correlation functions rather than
                their plain mean. Weights are renormalised to sum to one, so
                their scale carries no meaning, and uniform weights reproduce the
                unweighted estimate exactly. Requires *num_blocks*.

            corr_matrices: (num_blocks, n_l, n_r, >= nperseg) array_like, optional
                Externally supplied block correlation functions, bypassing
                ``prep_signals.corr_blackman_tukey``. Lets each block be an
                independent realization rather than a segment of one record.
                Requires *num_blocks*. Note that :meth:`save_state` does not
                persist them.

        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to prep_signals.correlation

        '''

        logger.info('Constructing half-spectrum matrix ... ')
        begin_frequency, end_frequency, nperseg_resolved = self._validate_frequency_params(
            nperseg, begin_frequency, end_frequency
        )

        if num_blocks is not None:
            if not isinstance(num_blocks, int):
                raise TypeError(f"num_blocks must be an int, got {type(num_blocks).__name__!r}.")
            training_blocks = self._coerce_blocks_array(training_blocks, num_blocks, 'training_blocks')

            if corr_matrices is None:
                logger.info(
                    f'Estimating block-wise correlation functions for cross-validation '
                    f'({num_blocks} blocks, {training_blocks.shape[0]} for training).')
                self.prep_signals.corr_blackman_tukey(nperseg_resolved, n_segments=num_blocks, refs_only=True)
            else:
                corr_matrices = self._validate_corr_matrices(
                    corr_matrices, num_blocks, nperseg_resolved)
                logger.info(
                    f'Using externally supplied block-wise correlation functions '
                    f'({num_blocks} blocks, {training_blocks.shape[0]} for training).')

            # set before the accessor below reads them
            self.num_blocks = num_blocks
            self.training_blocks = training_blocks
            self.corr_matrices = corr_matrices
            self.nperseg = nperseg_resolved

            corr_blocks = self._block_correlations(training_blocks)
            weights, n_eff = self._validate_weights(weights, training_blocks.shape[0])
            if weights is None:
                correlation_matrix = np.mean(corr_blocks, axis=0)
            else:
                if n_eff < 10:
                    logger.warning(
                        f'The weights concentrate the estimate on n_eff={n_eff:.1f} effective '
                        f'blocks (of {training_blocks.shape[0]}); the estimate is correspondingly '
                        f'noisy.')
                correlation_matrix = np.tensordot(weights, corr_blocks, axes=(0, 0))

            self.weights = weights
            self.n_eff = n_eff
        else:
            if weights is not None:
                raise ValueError('weights weight the blocks of the estimate and require num_blocks.')
            if corr_matrices is not None:
                raise ValueError('corr_matrices are indexed by block and require num_blocks.')
            if self.prep_signals._last_meth == 'welch':
                logger.info("The selected spectral estimation method (Welch) is not recommended (applied window introduces damping bias).")
            # nperseg=None signals correlation() to reuse precomputed correlations
            correlation_matrix = self.prep_signals.correlation(nperseg, **kwargs)

            self.num_blocks = None
            self.training_blocks = None
            self.corr_matrices = None
            self.weights = None
            self.n_eff = None

        nperseg = nperseg_resolved

        selected_omega_vector, spectrum_tensor, factor_a = self._windowed_half_spectrum(
            correlation_matrix, nperseg, window_decay, begin_frequency, end_frequency)

        self.begin_frequency = begin_frequency
        self.end_frequency = end_frequency
        self.nperseg = nperseg
        self.window_decay = window_decay

        self.selected_omega_vector = selected_omega_vector
        self.pos_half_spectra = spectrum_tensor
        self.factor_a = factor_a

        self.state[0] = True

    @property
    def num_omega(self):
        return self.selected_omega_vector.shape[0]

    @staticmethod
    def _as_real(arr, complex_coefficients):
        """Return arr.real when using real coefficients, else arr unchanged."""
        return arr if complex_coefficients else arr.real

    def estimate_model(self, order, complex_coefficients=False):
        '''
        Estimate a right matrix-fraction model from positive half-spectra, by
        constructing a set of reduced normal equations as shown in Peeters 2004. 
        The polynomial is identified following Cauberghe 2004. Sec. 5.2.1 
        
        Verboven 2002: Sect. 5.3.3 has a discussion on the use of real or complex
        valued coefficients, favoring complex ones. Guillaume 2003, Peeters 2004 
        just assume real coefficients, while later references, e.g.  
        Cauberghe 2004, Reynders 2012 use complex coefficients.
        However, with complex coefficients, stabilization diagrams seem to 
        become corrupted. 
        
        Note: The previous implementation was wrong in the estimation of
        alpha coefficients and led to "bad" stabilization. Additionally there 
        was a wrong sign in the assembly of the C_c matrix, which led to corrupted
        mode shapes.
        
        .. TODO::
            * implement weighting function; c.p. Peeters 2004 Sect. 2.2
            * improve assembly by exploiting the Toeplitz structure of S, R, T; c.p. Cauberghe 2004 Eq. 5.17ff
            * Investigate LS-TLS solution by using a SVD
            * estimate polynomial once at highest order and construct all lower 
              order models from these coefficients; c.p. Peeters 2004 Sect. 2.4
            * Check, if alternative solution for \alpha in Reynders 2012. Sec. 5.2.4 
              leads to clearer stabilization, or it it is actually equivalent to 
              the current implementation
        
        
        Parameters
        ----------
            order: integer, required
                Model order, at which the RMF model should be estimated
            
            complex_coefficients: bool, optional
                Whether to assume real or complex coefficients
                
        Returns
        -------
            alpha: numpy.ndarray
                Denominator coefficients: Array of shape ((order + 1) * n_r, n_r)
                
            beta_l_i: numpy.ndarray
                Numerator coefficients: Array of shape (order + 1, n_r, n_l)
        
        '''
        ctx = self._assemble_normal_equations(order, complex_coefficients)
        return ctx.alpha, ctx.beta_l_i

    def _assemble_normal_equations(self, order, complex_coefficients=False):
        '''
        Assemble and solve the reduced normal equations at a single model order.

        Carries out the estimation described in :meth:`estimate_model` and
        returns the full assembly rather than only the coefficients.

        Parameters
        ----------
            order: integer, required
                Model order, at which the RMF model should be estimated

            complex_coefficients: bool, optional
                Whether to assume real or complex coefficients

        Returns
        -------
            ctx: NormalEquationsContext
                The assembled normal equations and their solution
        '''
        if order > self.nperseg - 1:
            raise RuntimeError(f'Order cannot be higher than nperseg - 1 (={self.nperseg - 1}).')

        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        selected_omega_vector = self.selected_omega_vector
        num_omega = self.num_omega
        pos_half_spectra = self.pos_half_spectra

        sampling_rate = self.prep_signals.sampling_rate
        Delta_t = 1 / sampling_rate

        # whether to assume real or complex coefficients
        if complex_coefficients:
            dtype = complex
        else:
            dtype = float

        RS_solutions = np.zeros((order + 1, (order + 1) * n_r, n_l), dtype=dtype)
        M = np.zeros(((order + 1) * n_r, (order + 1) * n_r), dtype=dtype)

        # Create matrices X_0 and Y_0, Peeters 2004: Sect. 2.2ff
        # for channel-dependent weights, this has to move into the loop below
        X_o = np.exp(1j * selected_omega_vector[:, np.newaxis] *
                     Delta_t * np.arange(order + 1)[np.newaxis,:])  # (num_omega, (order + 1))
        X_o_H = np.conj(X_o.T)  # ((order + 1), num_omega)
        R_o = self._as_real(X_o_H @ X_o, complex_coefficients)  # ((order + 1),(order + 1))

        Y_o = np.empty((num_omega, ((order + 1) * n_r)), dtype=complex)
        for i_l in range(n_l):
            for kk in range(num_omega):
                Y_o[kk,:] = np.kron(-X_o[kk,:], pos_half_spectra[i_l,:, kk].T)

            S_o = self._as_real(X_o_H @ Y_o, complex_coefficients)  # ((order+1),(order+1)*n_r)
            T_o = self._as_real(np.conj(Y_o.T) @ Y_o, complex_coefficients)  # ((order+1)*n_r,…)

            RS_solution = np.linalg.solve(R_o, S_o)

            M = M + (T_o - np.conj(S_o).T @ RS_solution)

            RS_solutions[:,:, i_l] = RS_solution

        # factor 2 stems from the derivative of the quadratic cost function and
        # applies to the whole sum over output channels; c.p. Peeters 2004 Eq. 10
        M *= 2

        # Compute alpha and beta coefficients: Cauberghe 2004. Sec. 5.2.1
        M_aa = M[:order * n_r,:order * n_r]
        M_ab = M[:order * n_r, -n_r:]
        alpha_b = -np.linalg.solve(M_aa, M_ab)
        alpha = np.concatenate((alpha_b, np.eye(n_r)), axis=0)  # ((order + 1) * n_r, n_r)

        beta_l_i = np.zeros(((order + 1), n_r, n_l), dtype=dtype)
        for i_l in range(n_l):
            RS_solution = RS_solutions[:,:, i_l]
            beta_l = -RS_solution @ alpha

            beta_l_i[:,:, i_l] = beta_l

        return NormalEquationsContext(
            order=order, X_o=X_o, RS_solutions=RS_solutions, M=M,
            M_aa=M_aa, alpha=alpha, beta_l_i=beta_l_i)

    def modal_analysis_state_space(self, alpha, beta_l_i):
        '''
        Perform a modal analysis of the identified polyomial by converting it 
        into a state-space model, as outlined in Reynders-2012: Lemma 2.2, followed
        by an eigendecomposition. 
        Mode shapes are scaled to unit modal displacements. Complex conjugate 
        and real modes are removed prior to further processing. Damping values
        are corrected, if half-spectra were constructed with an exponential window.
        
        .. TODO::
            * numerical optimization to increase speed
        
        
        Parameters
        -------
            alpha: numpy.ndarray
                Denominator coefficients: Array of shape ((order + 1) * n_r, n_r)
                
            beta_l_i: numpy.ndarray
                Numerator coefficients: Array of shape (order + 1, n_r, n_l)
         
        Returns
        -------
            modal_frequencies: (order * n_r,) numpy.ndarray 
                Array holding the modal frequencies for each mode
            modal_damping: (order * n_r,) numpy.ndarray 
                Array holding the modal damping ratios (0,100) for each mode
            mode_shapes: (n_l, order * n_r,) numpy.ndarray 
                Complex array holding the mode shapes 
            eigenvalues: (order * n_r,) numpy.ndarray
                Complex array holding the eigenvalues for each mode
        '''
        accel_channels = self.prep_signals.accel_channels
        velo_channels = self.prep_signals.velo_channels

        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        factor_a = self.factor_a
        sampling_rate = self.prep_signals.sampling_rate

        order = alpha.shape[0] // n_r - 1

        # Create matrices A_c and C_c;
        # Reynders-2012-SystemIdentificationMethodsFor(Operational)ModalAnalysisReviewAndComparison: Lemma 2.2
        A_p = alpha[-n_r:,:]
        B_p = beta_l_i[order,:,:].T

        A_c = np.zeros((order * n_r, order * n_r), dtype=alpha.dtype)
        C_c = np.zeros((n_l, order * n_r), dtype=alpha.dtype)

        for p_i in range(order):
            A_p_i = alpha[(order - p_i - 1) * n_r:(order - p_i) * n_r,:]

            this_A_c_block = -np.linalg.solve(A_p, A_p_i)
            A_c[:n_r, p_i * n_r:(p_i + 1) * n_r] = this_A_c_block

            B_p_i = beta_l_i[order - p_i - 1,:,:].T

            this_C_c_block = B_p_i + (B_p @ this_A_c_block)
            C_c[:, p_i * n_r:(p_i + 1) * n_r] = this_C_c_block

        A_c_rest = np.eye((order - 1) * n_r)
        A_c[n_r:,:(order - 1) * n_r] = A_c_rest

        eigvals, eigvecs_r = np.linalg.eig(A_c)

        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        n_modes = len(conj_indices)

        modal_frequencies = np.zeros((n_modes,))
        modal_damping = np.zeros((n_modes,))
        mode_shapes = np.zeros((n_l, n_modes), dtype=complex)
        eigenvalues = np.zeros((n_modes), dtype=complex)

        Phi = C_c @ eigvecs_r

        for i, ind in enumerate(reversed(conj_indices)):

            lambda_i = np.log(eigvals[ind]) * sampling_rate
            # damping with correction if exponential window was applied to spectra
            # if factor_a is not None:
            #     lambda_i -= factor_a * sampling_rate

            freq_i = np.abs(lambda_i) / (2 * np.pi)
            damping_i = self._compute_damping(lambda_i, freq_i, factor_a, sampling_rate)

            mode_shape_i = Phi[:, ind]

            # scale modeshapes to modal displacements
            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)

            # rotate mode shape in complex plane
            mode_shape_i = self.rescale_mode_shape(mode_shape_i)

            modal_frequencies[i] = freq_i
            modal_damping[i] = damping_i
            mode_shapes[:, i] = mode_shape_i
            eigenvalues[i] = lambda_i

        # self._lower_residuals = np.zeros((n_l, n_r))
        # self._upper_residuals = np.zeros((n_l, n_r))
        # self._mode_shapes_raw = Phi[:,np.flip(conj_indices)]
        # self._participation_vectors = eigvecs_r[-n_r:, np.flip(conj_indices)]
        # self._participation_vectors /= self._participation_vectors[:, np.argmax(np.abs(self._participation_vectors), axis=0)]
        # self._eigenvalues = eigenvalues

        argsort = np.argsort(modal_frequencies)

        # remove all frequencies outside the spectral frequency band
        inds = (modal_frequencies[argsort] >= self.begin_frequency) & (modal_frequencies[argsort] <= self.end_frequency)
        argsort = argsort[inds]

        return modal_frequencies[argsort], modal_damping[argsort], mode_shapes[:, argsort], eigenvalues[argsort]

    def _compute_damping(self, lambda_i, freq_i, factor_a, sampling_rate):
        """Compute modal damping ratio with optional exponential-window correction."""
        if factor_a is None:
            return np.real(lambda_i) / np.abs(lambda_i) * (-100)
        return (
            np.real(lambda_i) / np.abs(lambda_i)
            - factor_a * sampling_rate / (freq_i * 2 * np.pi)
        ) * (-100)

    def _build_companion_matrix_residuals(self, alpha, n_r, order):
        """Build the transposed companion matrix for residuals-based modal analysis."""
        A_p = alpha[-n_r:, :]
        A_c = np.zeros((order * n_r, order * n_r), dtype=alpha.dtype)
        for p_i in range(order):
            A_p_i = alpha[(order - p_i - 1) * n_r:(order - p_i) * n_r, :]
            A_c[p_i * n_r:(p_i + 1) * n_r, :n_r] = -np.linalg.solve(A_p, A_p_i)
        A_c[:-n_r, n_r:] = np.eye((order - 1) * n_r)
        return A_c

    def _fit_mode_shapes_ls(self, eigenvalues, n_l, n_r, n_modes,
                            participation_vectors):
        """Fit mode shapes and residuals via least-squares spectral fitting."""
        accel_channels = self.prep_signals.accel_channels
        velo_channels = self.prep_signals.velo_channels
        A = np.zeros((self.num_omega * 2 * n_r, (2 * n_modes + 4 * n_r)))
        h = np.zeros((self.num_omega * 2 * n_r, n_l))
        Df1_lines = np.zeros((self.num_omega, n_modes), dtype=complex)
        Df2_lines = np.zeros((self.num_omega, n_modes), dtype=complex)
        for i_omega, omega in enumerate(self.selected_omega_vector):
            Df1 = 1 / (1j * omega - eigenvalues)
            Df2 = 1 / (1j * omega - np.conj(eigenvalues))
            Df1_lines[i_omega,:] = Df1
            Df2_lines[i_omega,:] = Df2
            LDf1 = participation_vectors * Df1[np.newaxis, :]
            LDf2 = np.conj(participation_vectors) * Df2[np.newaxis, :]
            A_f = np.zeros((2 * n_r, (2 * n_modes + 4 * n_r)))
            A_f[:n_r, :n_modes] = np.real(LDf1) + np.real(LDf2)
            A_f[n_r:, :n_modes] = np.imag(LDf1) + np.imag(LDf2)
            A_f[:n_r, n_modes:2 * n_modes] = -np.imag(LDf1) + np.imag(LDf2)
            A_f[n_r:, n_modes:2 * n_modes] = np.real(LDf1) - np.real(LDf2)
            A_f[:n_r, 2 * n_modes:2 * n_modes + n_r] = np.eye(n_r)
            A_f[n_r:, 2 * n_modes + n_r:2 * n_modes + 2 * n_r] = np.eye(n_r)
            A_f[:n_r, 2 * n_modes + 2 * n_r:2 * n_modes + 3 * n_r] = np.eye(n_r) * omega ** 2
            A_f[n_r:, 2 * n_modes + 3 * n_r:2 * n_modes + 4 * n_r] = np.eye(n_r) * omega ** 2
            A[i_omega * 2 * n_r:(i_omega + 1) * 2 * n_r, :] = A_f
            h[i_omega * 2 * n_r:i_omega * 2 * n_r + n_r, :] = np.real(self.pos_half_spectra[:, :, i_omega]).T
            h[i_omega * 2 * n_r + n_r:i_omega * 2 * n_r + 2 * n_r, :] = np.imag(self.pos_half_spectra[:, :, i_omega]).T
        pinv_A = np.linalg.pinv(A)
        X = pinv_A @ h
        self._lsfd_ctx = LSFDContext(
            A=A, X=X, pinv_A=pinv_A, residual=h - A @ X,
            Df1=Df1_lines, Df2=Df2_lines, eigenvalues=eigenvalues,
            participation_vectors=participation_vectors)
        mode_shapes_raw = X.T[:, :n_modes] + 1j * X.T[:, n_modes:2 * n_modes]
        mode_shapes = np.zeros((n_l, n_modes), dtype=complex)
        for ind in range(n_modes):
            # each mode is integrated at its own circular frequency: 2 pi f_i = |lambda_i|
            mode_shape_i = self.integrate_quantities(
                mode_shapes_raw[:, ind], accel_channels, velo_channels,
                np.abs(eigenvalues[ind])
            )
            mode_shapes[:, ind] = self.rescale_mode_shape(mode_shape_i)
        lower_res = X.T[:, 2 * n_modes:2 * n_modes + n_r] + 1j * X.T[:, 2 * n_modes + n_r:2 * n_modes + 2 * n_r]
        upper_res = X.T[:, 2 * n_modes + 2 * n_r:2 * n_modes + 3 * n_r] + 1j * X.T[:, 2 * n_modes + 3 * n_r:2 * n_modes + 4 * n_r]
        return mode_shapes, mode_shapes_raw, lower_res, upper_res

    def modal_analysis_residuals(self, alpha, *args):
        '''
        Perform a modal analysis of the identified polyomial with the least-squares
        residual-based method as outlined in Steffensen-2025-VarianceEstimation... Sect. 2.1
        Mode shapes are scaled to unit modal displacements. Complex conjugate 
        and real modes are removed prior to further processing. Damping values
        are corrected, if half-spectra were constructed with an exponential window.
        
        .. TODO::
            * numerical optimization to increase speed
            
            
        Parameters
        -------
            alpha: numpy.ndarray
                Denominator coefficients: Array of shape ((order + 1) * n_r, n_r)
                
        Returns
        -------
            modal_frequencies: (order * n_r,) numpy.ndarray 
                Array holding the modal frequencies for each mode
            modal_damping: (order * n_r,) numpy.ndarray 
                Array holding the modal damping ratios (0,100) for each mode
            mode_shapes: (n_l, order * n_r,) numpy.ndarray 
                Complex array holding the mode shapes 
            eigenvalues: (order * n_r,) numpy.ndarray
                Complex array holding the _eigenvalues for each mode
        '''
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        factor_a = self.factor_a
        sampling_rate = self.prep_signals.sampling_rate
        order = alpha.shape[0] // n_r - 1

        if np.issubdtype(alpha.dtype, complex):
            logger.warning('Residual-based modal analysis with complex coefficients has not been verified.')

        A_c = self._build_companion_matrix_residuals(alpha, n_r, order)
        eigvals, eigvecs_l, eigvecs_r = scipy.linalg.eig(A_c, left=True, right=True)
        # remove_conjugates takes (eigval, eigvec_r, eigvec_l) and returns
        # (eigval, eigvec_l, eigvec_r); passing the vectors right-first is
        # required, otherwise left/right eigenvectors are swapped downstream.
        eigvals, eigvecs_l, eigvecs_r = self.remove_conjugates(eigvals, eigvecs_r, eigvecs_l)

        _eigenvalues = np.log(eigvals) * sampling_rate
        _modal_frequencies = np.abs(_eigenvalues) / (2 * np.pi)

        inds = np.where(
            (_modal_frequencies >= self.begin_frequency) & (_modal_frequencies <= self.end_frequency)
        )[0]
        n_modes = len(inds)

        modal_damping = np.zeros((n_modes,))
        participation_vectors = np.zeros((n_r, n_modes), dtype=complex)

        for i, ind in enumerate(inds):
            lambda_i = _eigenvalues[ind]
            freq_i = _modal_frequencies[ind]
            modal_damping[i] = self._compute_damping(lambda_i, freq_i, factor_a, sampling_rate)
            # copy: the in-place normalisation would otherwise write through the
            # basic-indexing view and corrupt eigvecs_l, which _modal_ctx retains
            part_vec = eigvecs_l[-n_r:, ind].copy()
            part_vec /= part_vec[np.argmax(np.abs(part_vec))]
            participation_vectors[:, i] = part_vec

        modal_frequencies = _modal_frequencies[inds]
        eigenvalues = _eigenvalues[inds]
        argsort = np.argsort(modal_frequencies)

        mode_shapes, mode_shapes_raw, lower_res, upper_res = self._fit_mode_shapes_ls(
            eigenvalues, n_l, n_r, n_modes, participation_vectors
        )

        # the fit ran in the unsorted in-band order; record the permutation so a
        # differentiation of its assembly can align with the returned modes
        self._lsfd_ctx.mode_order = argsort

        self._lower_residuals = lower_res
        self._upper_residuals = upper_res
        self._mode_shapes_raw = mode_shapes_raw[:, argsort]
        self._participation_vectors = participation_vectors[:, argsort]
        self._eigenvalues = eigenvalues[argsort]
        self._modal_ctx = ModalContext(
            A_c=A_c, eigvals_z=eigvals, eigvecs_l=eigvecs_l, eigvecs_r=eigvecs_r,
            mode_indices=inds[argsort])

        return modal_frequencies[argsort], modal_damping[argsort], mode_shapes[:, argsort], eigenvalues[argsort]

    def synthesize_spectrum(self, alpha, beta_l_i, modal=True, validation_blocks=None):
        '''
        Spectral synthetization in a modal decoupled form follows
        Steffensen-2025-VarianceEstimation... Sect. 2.1.2
        The spectral synthetization without modal decomposition follows
        Peeters-2004-ThePolyMAX...

        .. TODO::
            * numerical optimization to increase speed


        Parameters
        ----------
            alpha: numpy.ndarray
                Denominator coefficients: Array of shape ((order + 1) * n_r, n_r)

            beta_l_i: numpy.ndarray
                Numerator coefficients: Array of shape (order + 1, n_r, n_l)

            modal: bool, optional
                Synthesize a spectrum for each mode and its modal contribution
                to the full spectrum

            validation_blocks: list, optional
                Only meaningful if :meth:`build_half_spectra` was called with
                *num_blocks* (cross-validation mode) and *modal* is True. The
                selected blocks whose (block-wise, Blackman-Tukey) half-spectrum
                is used as ground truth for computing modal contributions,
                instead of ``self.pos_half_spectra``. Defaults to all blocks
                (matching the default of ``training_blocks`` in
                :meth:`build_half_spectra` -- pass disjoint sets for a held-out
                validation).

        Returns
        -------
            half_spec_modal: (n_l, n_r, num_omega, n_modes) numpy.ndarray
                Array holding the (modally decomposed) synthesized positive half
                spectra for each channel n_l and reference channel n_r and all modes

            modal_contributions: (order,) numpy.ndarray
                Array holding the contributions of each mode to the input
                spectrum
        '''

        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        sampling_rate = self.prep_signals.sampling_rate
        omega = self.selected_omega_vector
        num_omega = self.num_omega

        if modal:
            if self._lower_residuals is None:
                logger.warning('Residuals have not yet been estimated.')
                _, _, _, _ = self.modal_analysis_residuals(alpha)
            if self.num_blocks is not None:
                validation_blocks = self._coerce_blocks_array(
                    validation_blocks, self.num_blocks, 'validation_blocks')
                # validation blocks are held out, so they are never weighted
                corr_matrix = np.mean(self._block_correlations(validation_blocks), axis=0)
                _, comparison_spectrum, _ = self._windowed_half_spectrum(
                    corr_matrix, self.nperseg, self.window_decay,
                    self.begin_frequency, self.end_frequency)
            else:
                comparison_spectrum = self.pos_half_spectra
            return self._synthesize_spectrum_modal(n_l, n_r, num_omega, omega, comparison_spectrum)
        return self._synthesize_spectrum_nonmodal(alpha, beta_l_i, n_l, n_r, omega, sampling_rate)

    def _synthesize_spectrum_modal(self, n_l, n_r, num_omega, omega, comparison_spectrum):
        """Synthesize spectrum using modal decomposition (Steffensen 2025, Sect. 2.1.2)."""
        lower_residuals = self._lower_residuals
        upper_residuals = self._upper_residuals
        participation_vectors = self._participation_vectors
        mode_shapes_raw = self._mode_shapes_raw
        eigenvalues = self._eigenvalues
        n_modes = mode_shapes_raw.shape[1]

        half_spec_modal = np.zeros((n_l, n_r, num_omega, n_modes), dtype=complex)
        for ind in range(n_modes):
            lamda_r = eigenvalues[ind]
            part_vec = participation_vectors[:, ind]
            mode_shape = mode_shapes_raw[:, ind]
            numerator = (part_vec[:, np.newaxis] @ mode_shape[np.newaxis, :]).T
            half_spec_modal[:, :, :, ind] = (
                numerator[:, :, np.newaxis]
                / (1j * omega[np.newaxis, np.newaxis, :] - lamda_r)
                + np.conj(numerator)[:, :, np.newaxis]
                / (1j * omega[np.newaxis, np.newaxis, :] - np.conj(lamda_r))
            )

        half_spec_synth = np.sum(half_spec_modal, axis=-1)
        half_spec_synth += lower_residuals[:, :, np.newaxis]
        half_spec_synth += upper_residuals[:, :, np.newaxis] * omega[np.newaxis, np.newaxis, :] ** 2
        self._half_spec_synth = half_spec_modal

        Sigma_data = np.zeros((n_l * n_r), dtype=complex)
        Sigma_synth = np.zeros((n_l * n_r), dtype=complex)
        Sigma_data_synth = np.zeros((n_l * n_r, n_modes), dtype=complex)
        modal_contributions = np.zeros((n_modes), dtype=complex)

        if logger.isEnabledFor(logging.DEBUG):
            Sigma_data_synthtot = np.zeros((n_l * n_r))

        for i_r in range(n_r):
            for i_l in range(n_l):
                spec_data = comparison_spectrum[i_l, i_r, :]
                spec_synth = np.sum(half_spec_modal, axis=-1)[i_l, i_r, :]
                Sigma_data[i_r * n_l + i_l] = spec_data @ np.conj(spec_data.T)
                Sigma_synth[i_r * n_l + i_l] = spec_synth @ np.conj(spec_synth.T)
                if logger.isEnabledFor(logging.DEBUG):
                    Sigma_data_synthtot[i_r * n_l + i_l] = spec_data @ np.conj(spec_synth.T)
                for i in range(n_modes):
                    Sigma_data_synth[i_r * n_l + i_l, i] = spec_data @ np.conj(half_spec_modal[i_l, i_r, :, i])

        for i in range(n_modes):
            rho = Sigma_data_synth[:, i] / np.sqrt(Sigma_data * Sigma_synth)
            modal_contributions[i] = rho.mean()

        return half_spec_modal, modal_contributions

    def _synthesize_spectrum_nonmodal(self, alpha, beta_l_i, n_l, n_r, omega, sampling_rate):
        """Synthesize spectrum without modal decomposition (Peeters 2004, Eqs. 1, 3, 4)."""
        order = alpha.shape[0] // n_r - 1
        r_vec = np.arange(order + 1)
        half_spec_synth = np.zeros_like(self.pos_half_spectra)  # (n_l, n_r, num_omega)
        for i_omega in range(self.num_omega):
            Omega_r = np.exp(1j * omega[i_omega] / sampling_rate * r_vec)
            A = np.zeros((n_r, n_r), dtype=complex)
            for i_ord in range(order + 1):
                A += alpha[i_ord * n_r:(i_ord + 1) * n_r, :] * Omega_r[i_ord]
            A_inv = np.linalg.inv(A)
            B_o = np.sum(Omega_r[:, np.newaxis, np.newaxis] * beta_l_i[:, :, :], axis=0)
            half_spec_synth[:, :, i_omega] = B_o.T @ A_inv
        self._half_spec_synth = half_spec_synth
        return half_spec_synth, None

    def compute_modal_params(self, max_model_order, complex_coefficients=False,
                             algo='residuals', modal_contrib=None, validation_blocks=None):
        '''
        Perform a multi-order computation of modal parameters. Successively
        calls

         * estimate_model(order, complex_coefficients)
         * modal_analysis_residuals(alpha, beta_l_i) or modal_analysis_state_space(alpha, beta_l_i)
         * synthesize_spectrum(alpha, beta_l_i), if modal_contrib == True

        At ascending model orders, up to max_model_order.
        See the explanations in the the respective methods, for a detailed
        explanation of parameters.

        Parameters
        ----------
            max_model_order: integer
                Maximum model order, where to interrupt the algorithm.
            complex_coefficients: bool, optional
                Whether to estimate a real or complex RMFD model
            algo: str, optional
                Algorithm to use for modal analysis. Either 'state-space' or 'residuals'
                Both algorithms are approximately equally fast. The state space based
                algorithm seems to yield less complex mode shapes.
            modal_contrib: bool, optional
                Synthesize modal spectra and estimate modal contributions. Only
                to be used with residual-based modal analysis algorithm.
            validation_blocks: list, optional
                Only meaningful if :meth:`build_half_spectra` was called with
                *num_blocks* (cross-validation mode). Forwarded to
                :meth:`synthesize_spectrum` at every order when
                *modal_contrib* is True.
        '''
        algo, modal_contrib = self._setup_compute_params(
            max_model_order, algo, modal_contrib
        )
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels

        if not self.state[0]:
            raise RuntimeError("Call build_half_spectra() first.")

        logger.info('Computing modal parameters...')
        max_modes = max_model_order * n_r if complex_coefficients else max_model_order * n_r // 2
        modal_frequencies = np.zeros((max_model_order, max_modes))
        modal_damping = np.zeros((max_model_order, max_modes))
        mode_shapes = np.zeros((n_l, max_modes, max_model_order), dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_modes), dtype=complex)
        modal_contributions = np.zeros((max_model_order, max_modes,), dtype=complex) if modal_contrib else None
        # only the residual-based algorithm identifies participation vectors
        participation_vectors = np.zeros(
            (n_r, max_modes, max_model_order), dtype=complex) if algo == 'residuals' else None

        pbar = simplePbar(max_model_order)
        for order in range(1, max_model_order):
            next(pbar)
            alpha, beta_l_i = self.estimate_model(order, complex_coefficients)
            if algo == 'state-space':
                f, d, phi, lamda = self.modal_analysis_state_space(alpha, beta_l_i)
            else:
                f, d, phi, lamda = self.modal_analysis_residuals(alpha, beta_l_i)
            n_modes = len(f)
            if participation_vectors is not None:
                participation_vectors[:, :n_modes, order] = self._participation_vectors
            if modal_contrib:
                _, delta = self.synthesize_spectrum(alpha, beta_l_i, True, validation_blocks=validation_blocks)
                modal_contributions[order, :n_modes] = delta
            modal_frequencies[order, :n_modes] = f
            modal_damping[order, :n_modes] = d
            eigenvalues[order, :n_modes] = lamda
            mode_shapes[:, :n_modes, order] = phi

        self.max_model_order = max_model_order
        self.eigenvalues = eigenvalues
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.modal_contributions = modal_contributions
        self.participation_vectors = participation_vectors
        self.state[1] = True

    def _setup_compute_params(self, max_model_order, algo, modal_contrib):
        """Validate and normalise compute_modal_params arguments."""
        if max_model_order > self.nperseg - 1:
            raise ValueError(
                f"max_model_order ({max_model_order}) exceeds limit"
                f" self.nperseg - 1 ({self.nperseg - 1})"
            )
        if algo not in ['state-space', 'residuals']:
            raise ValueError(f"algo must be 'state-space' or 'residuals', got {algo!r}")
        if modal_contrib is None:
            modal_contrib = algo != 'state-space'
        if modal_contrib and algo == 'state-space':
            logger.warning('State space algorithm can not be used with spectral synthetization.')
            algo = 'residuals'
        return algo, modal_contrib

    def _collect_modal_state(self):
        """Return dict of modal parameter entries for save_state.

        Subclasses that identify additional modal quantities extend this.
        """
        return {
            'self.modal_frequencies': self.modal_frequencies,
            'self.modal_damping': self.modal_damping,
            'self.mode_shapes': self.mode_shapes,
            'self.eigenvalues': self.eigenvalues,
            'self.modal_contributions': self.modal_contributions,
            'self.participation_vectors': self.participation_vectors,
            'self.max_model_order': self.max_model_order,
        }

    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0             1
        # self.state= [Half_spectra, Modal Par.
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time
        # out_dict['self.prep_signals']=self.prep_signals
        if self.state[0]:  # half spectra
            out_dict['self.begin_frequency'] = self.begin_frequency
            out_dict['self.end_frequency'] = self.end_frequency
            out_dict['self.nperseg'] = self.nperseg
            out_dict['self.selected_omega_vector'] = self.selected_omega_vector
            out_dict['self.pos_half_spectra'] = self.pos_half_spectra
            out_dict['self.factor_a'] = self.factor_a
        if self.state[1]:  # modal params
            out_dict.update(self._collect_modal_state())

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):

        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            # bool(...): entries loaded straight out of the .npz archive are
            # numpy.bool_, not plain Python bool.
            state = [bool(s) for s in in_dict['self.state']]
        else:
            return

        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be PreProcessSignals, got {type(prep_signals).__name__!r}"
            )
        setup_name = str(in_dict['self.setup_name'].item())
        if setup_name != prep_signals.setup_name:
            raise ValueError(
                f"setup_name mismatch: expected {setup_name!r},"
                f" got {prep_signals.setup_name!r}"
            )
        start_time = prep_signals.start_time

        if start_time != prep_signals.start_time:
            raise ValueError(
                f"start_time mismatch: expected {start_time!r},"
                f" got {prep_signals.start_time!r}"
            )

        pLSCF_object = cls(prep_signals)
        pLSCF_object.state = state
        if state[0]:  # positive half spectra
            pLSCF_object.begin_frequency = validate_array(in_dict['self.begin_frequency'])
            pLSCF_object.end_frequency = validate_array(in_dict['self.end_frequency'])
            pLSCF_object.nperseg = validate_array(in_dict['self.nperseg'])
            pLSCF_object.selected_omega_vector = validate_array(in_dict['self.selected_omega_vector'])
            pLSCF_object.pos_half_spectra = validate_array(in_dict['self.pos_half_spectra'])
            pLSCF_object.factor_a = validate_array(in_dict['self.factor_a'])
        if state[1]:  # modal params
            cls._restore_modal_state(pLSCF_object, in_dict)

        return pLSCF_object

    @classmethod
    def _restore_modal_state(cls, pLSCF_object, in_dict):
        """Restore modal parameter attributes from a loaded archive dict.

        Subclasses that identify additional modal quantities extend this.
        """
        pLSCF_object.modal_frequencies = in_dict['self.modal_frequencies']
        pLSCF_object.modal_damping = in_dict['self.modal_damping']
        pLSCF_object.mode_shapes = in_dict['self.mode_shapes']
        pLSCF_object.eigenvalues = in_dict['self.eigenvalues']
        pLSCF_object.modal_contributions = in_dict['self.modal_contributions']
        # absent from archives written before participation vectors were stored
        pLSCF_object.participation_vectors = in_dict.get('self.participation_vectors', None)
        pLSCF_object.max_model_order = int(in_dict['self.max_model_order'])


def _build_channel_pairs(channel_inds, ref_channel_inds, ref_channels):
    """Build non-repeating (i_l, i_r) index pairs for channel combinations."""
    num_channels = len(channel_inds)
    num_ref_channels = len(ref_channel_inds)
    i_l_i_r = np.full((num_channels * num_ref_channels, 2), np.nan)
    j = 0
    for index_l in channel_inds:
        index_l_in_ref = ref_channels.index(index_l) if index_l in ref_channels else None
        for index_r in ref_channel_inds:
            if index_l_in_ref is None:
                i_l_i_r[j, 0] = index_l
                i_l_i_r[j, 1] = index_r
                j += 1
            else:
                index_r_in_all = ref_channels[index_r]
                inds_inv = np.array([[index_r_in_all, index_l_in_ref]])
                if not np.any(np.all(i_l_i_r == inds_inv, axis=1)):
                    i_l_i_r[j, 0] = index_l
                    i_l_i_r[j, 1] = index_r
                    j += 1
    return i_l_i_r[~np.all(np.isnan(i_l_i_r), axis=1), :].astype(int)


def plot_spec_synth(modal_data, modelist=None, channel_inds=None, ref_channel_inds=None, axes=None):
    import matplotlib.pyplot as plt

    half_spec_synth = modal_data._half_spec_synth
    pos_half_spectra = modal_data.pos_half_spectra
    ref_channels = modal_data.prep_signals.ref_channels
    sampling_rate = modal_data.prep_signals.sampling_rate
    channel_headers = modal_data.prep_signals.channel_headers

    if channel_inds is None:
        channel_inds = np.arange(modal_data.prep_signals.num_analised_channels)
    if ref_channel_inds is None:
        ref_channel_inds = np.arange(modal_data.prep_signals.num_ref_channels)

    i_l_i_r = _build_channel_pairs(channel_inds, ref_channel_inds, ref_channels)
    num_plots = len(i_l_i_r)

    fig2, axes = plt.subplots(num_plots, 1, sharex='col', sharey='col', squeeze=False)
    ft_freq = modal_data.selected_omega_vector / 2 / np.pi

    for j in range(num_plots):
        i_l, i_r = i_l_i_r[j, :]
        ft_meas = pos_half_spectra[i_l, i_r, :]
        label = 'Inp.' if j == 0 else None
        axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_meas)), ls='solid', color='k', label=label)
        for ip, i in enumerate(modelist):
            ft_synth = half_spec_synth[i_l, i_r, :, i]
            color = str(np.linspace(0, 1, len(modelist) + 2)[ip + 1])
            ls = ['-', '--', ':', '-.'][i % 4]
            label = f'm={i+1}' if j == 0 else None
            axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_synth)), color=color, ls=ls, label=label)
            axes[j, 0].set_ylabel(
                f'{channel_headers[i_l]}\n $\\leftrightarrow$ \n{channel_headers[ref_channels[i_r]]}',
                rotation=0, labelpad=20, va='center', ha='center'
            )

    axes[-1, 0].set_xlabel(r'$f$ [\si{\hertz}]')
    for ax in axes.flat:
        ax.set_yticks([])
        ax.set_xlim(0, 1 / 2 * sampling_rate)
        ax.set_ylim(ymin=-50)
    fig2.legend(title='Mode')
    fig2.subplots_adjust(left=None, bottom=None, right=0.97, top=0.97, wspace=None, hspace=0.1)
    return fig2


def main():
    pass


if __name__ == '__main__':
    main()
