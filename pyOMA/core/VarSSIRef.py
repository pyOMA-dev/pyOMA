# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Covariance-driven SSI with propagated parameter variances (VarSSIRef)."""
import scipy.sparse as sparse
import numpy as np
import scipy.linalg
import os
from collections import namedtuple

from .Helpers import lq_decomp, simplePbar, ConfigFile
from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Container for per-eigenvalue geometric data passed to Jacobian helpers.
_EigvalData = namedtuple(
    '_EigvalData',
    ['lambda_i', 'Phi_i', 'Chi_i', 'J_fixiili', 'order',
     'state_matrix', 'output_matrix', 'alpha_ik', 't_ik', 's_ik', 'e_k'])

# Container for per-order variance inputs to _compute_per_eigval.
_VarParams = namedtuple(
    '_VarParams', ['sigma_AC', 'J_AHT', 'Q4n', 'On_up2i', 'PQ1', 'PQ23'])

# Container for per-order modal-loop context passed to _compute_per_eigval.
_OrderCtx = namedtuple(
    '_OrderCtx', ['eigvec_l', 'eigvec_r', 'output_matrix', 'order', 'sampling_rate',
                  'state_matrix'])


def vectorize(matrix):
    '''
    .. math::

        A=\\begin{bmatrix}
             1 & 2 & 3 \\
             4 & 5 & 6 \\
             7 & 8 & 9 \\
        \\end{bmatrix}

    returns vertically stacked columns of matrix A

    ..math::

        \\begin{bmatrix}
         1 \\
         4 \\
         7 \\
         2 \\
         5 \\
         8 \\
         3 \\
         6 \\
         9 \\
        \\end{bmatrix}

    '''
    return np.reshape(matrix, (np.prod(matrix.shape), 1), 'F')


def dot(a, b):
    if sparse.issparse(b):
        return b.T.dot(a.T).T
    else:
        return a.dot(b)

# import scipy.sparse.linalg


def permutation(a, b):
    P = sparse.lil_matrix((a * b, a * b))  # zeros((a*b,a*b))
    ind1 = np.array(range(a * b))  # range(a*b)
    with np.errstate(divide='ignore'):
        ind2 = np.mod(ind1 * a, a * b - 1)  # mod(ind1*a,a*b-1)
    ind2[-1] = a * b - 1  # a*b-1
    P[ind1, ind2] = 1

    return P


class VarSSIRef(ModalBase):
    """Covariance-driven SSI with first-order perturbation variance estimation.

    Extends :class:`~pyOMA.core.SSICovRef.BRSSICovRef` with analytical
    uncertainty propagation from measurement noise through the correlation
    functions, Toeplitz matrix, SVD, and eigendecomposition to the final modal
    parameters.  Both covariance-based and projection-based subspace estimation
    are supported.

    The standard workflow is:

    1. :meth:`build_subspace_mat` — build the subspace matrix and its
       statistical properties.
    2. :meth:`compute_state_matrices` — estimate state and output matrices.
    3. :meth:`prepare_sensitivities` — pre-compute sensitivity matrices for
       variance propagation.
    4. :meth:`compute_modal_params` — identify modal parameters with variances.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object providing correlation functions and
        channel metadata.

    .. TODO::
        * define unit tests to check functionality after changes
        * optimize multi-order QR-based estimation routine
        * add mode-shape integration with variances
        * use Monte-Carlo sampling in the last step of variance propagation
    """

    def __init__(self, prep_signals, cache_variance_factors=False):
        """
        Parameters
        ----------
        prep_signals : PreProcessSignals
            Pre-processed signal object.
        cache_variance_factors : bool, optional
            Default for whether :meth:`compute_modal_params` caches the per-mode
            variance factors ``U_fixi``/``U_phii`` (Tier A), enabling
            millisecond post-hoc reweighting via :meth:`apply_block_weights`.
            Can be overridden per call.
        """
        super().__init__(prep_signals)

        #             0         1           2
        # self.state= [Hankel, State Mat., Modal Par.
        self.state = [False, False, False]
        # Tracked separately from self.state: state[1] only reflects
        # compute_state_matrices() (there is no dedicated slot for
        # prepare_sensitivities()). compute_modal_params() actually depends
        # on prepare_sensitivities() having run since the most recent
        # compute_state_matrices()/build_subspace_mat() call.
        self.sensitivities_prepared = False

        self.num_block_columns = None
        self.num_block_rows = None
        self.subspace_matrix = None

        # weights is None for the classical unweighted estimator; a
        # normalized (num_blocks,) array switches all subspace/covariance
        # estimates to their importance-weighted counterparts.
        self.weights = None
        self.n_eff = None
        self.external_corr = False

        # Post-hoc block reweighting (Tier B / Tier A): applied on top of an
        # unweighted build to the cached variance factors only, leaving the
        # point estimates untouched.  None means uniform (unweighted).
        self.block_weights = None
        self.block_weight_convention = None

        # Tier A per-mode variance-factor caches (populated by
        # compute_modal_params when caching is on): dict order -> stacked array
        # of shape (n_modes, 2, n_b) for U_fixi and (n_modes, 2*n_l, n_b) for
        # U_phii.  None disables Tier A (apply_block_weights unavailable).
        self.cache_variance_factors = cache_variance_factors
        self.U_fixi_cache = None
        self.U_phii_cache = None

        self.max_model_order = None

        self.lsq_method = 'pinv'  # 'qr'
        self.variance_algo = 'fast'  # 'slow'
        self.state_matrix = None
        self.output_matrix = None

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        cfg = ConfigFile(conf_file)
        num_block_columns = cfg.int('Number of Block-Columns')
        max_model_order = cfg.int('Maximum Model Order')
        num_blocks = cfg.int('Number of Blocks')
        subspace_method = cfg.str('Subspace Method (projection/covariance)')
        lsq_method = cfg.str('LSQ Method for A (pinv/qr)')
        variance_algo = cfg.str('Variance Algorithm (fast/slow)')

        ssi_object = cls(prep_signals)

        ssi_object.build_subspace_mat(
            num_block_columns,
            num_blocks=num_blocks,
            subspace_method=subspace_method)
        ssi_object.compute_state_matrices(
            max_model_order, lsq_method=lsq_method)
        ssi_object.prepare_sensitivities(variance_algo=variance_algo)
        ssi_object.compute_modal_params()

        return ssi_object

    def write_config(self, conf_file):
        ConfigFile.write(conf_file, {
            'Number of Block-Columns': self.num_block_columns,
            'Maximum Model Order': self.max_model_order,
            'Number of Blocks': self.num_blocks,
            'Subspace Method (projection/covariance)': self.subspace_method,
            'LSQ Method for A (pinv/qr)': self.lsq_method,
            'Variance Algorithm (fast/slow)': self.variance_algo,
        })

    @staticmethod
    def _validate_weights(weights, num_blocks):
        """Validate and renormalize block weights; return (weights, n_eff).

        ``weights=None`` keeps the classical unweighted estimator and returns
        ``(None, float(num_blocks))``.  Otherwise the weights are checked to
        be non-negative with a positive sum, renormalized to sum to one, and
        Kish's effective sample size ``n_eff = 1 / sum(w**2)`` is derived.
        """
        if weights is None:
            return None, float(num_blocks)
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape[0] != num_blocks:
            raise ValueError(
                f"Expected {num_blocks} weights, got {weights.shape[0]}.")
        if np.any(weights < 0):
            raise ValueError("'weights' must be non-negative.")
        total = weights.sum()
        if not total > 0:
            raise ValueError("'weights' must contain at least one positive entry.")
        weights = weights / total
        n_eff = 1.0 / float(np.sum(weights ** 2))
        return weights, n_eff

    @staticmethod
    def _block_weight_factor(weights, num_blocks, convention='substitution'):
        r"""Dense ``(num_blocks, num_blocks)`` post-hoc block-reweighting matrix ``W(w)``.

        Right-multiplying *any* cached, already-centered block factor ``F`` (its
        block index on the last axis: ``hankel_cov_matrix``, ``Q1..Q4``,
        ``J_OHT``, the per-mode ``U_fixi``/``U_phii``) by this matrix reweights
        it to the block weights ``weights`` *without* redoing the SVD or
        sensitivity preparation::

            F_w = F @ W(w),      cov_w = F_w @ F_w.T

        with (Eq. I of the task brief)

        .. math::

            W(w) = s(w)\,\bigl(I_{n_b} - w\,\mathbf{1}^{\mathsf T}\bigr)\,
                   \operatorname{diag}(\sqrt{w}).

        The ``(I - w 1^T)`` factor re-centers every column on the *new* weighted
        mean -- the cached columns are already centered on the uniform mean, so
        their columns sum to zero and uniform weights leave them untouched;
        ``diag(sqrt(w))`` applies the importance weights; and ``s(w)`` is a
        convention-dependent scalar (below) fixing the covariance normalization.
        All three conventions coincide at uniform weights, where ``F @ W``
        reproduces ``F`` to machine precision -- the binding invariant.

        .. warning::
            This is a *frozen-linearization* (delta-method) reweighting.  The
            point estimates (mean Hankel matrix, SVD, state/output matrices,
            eigenstructure) and every Jacobian stay at their original weighting;
            the result is the covariance of the reweighted estimator around the
            original linearization.  It is first-order consistent for moderate
            weight changes but does NOT relocate a point estimate contaminated
            by a bad block -- for that use the build-time weighted path.  Trust
            it only while the effective sample size ``n_eff`` stays well above
            ~10.

        Parameters
        ----------
        weights : (num_blocks,) array or None
            Non-negative block weights, renormalized to sum to one.  ``None``
            selects uniform weights, for which ``W`` is the centering projector
            that acts as the identity on already-centered factors.
        num_blocks : int
            Number of blocks ``n_b`` (the last-axis length of the cached factors).
        convention : {'substitution', 'reliability', 'precision'}
            Statistical convention for the scalar ``s(w)`` with Kish's effective
            sample size ``n_eff = 1 / sum(w**2)``:

            ``'substitution'`` (default)
                ``s = sqrt(n_b**2 (n_b - 1) / (n_eff (n_eff - 1)))``; pretends
                ``n_eff`` uniform blocks, reproduces the build-time weighted
                covariance factor of ``feature/weighted-subspace`` exactly, and
                is the most conservative (largest sigma).
            ``'reliability'``
                ``s = sqrt(n_b (n_b - 1) / (n_eff - 1))``; unbiased covariance
                of the weighted mean of homoscedastic blocks.  Then
                ``variance(substitution) / variance(reliability) = n_b / n_eff``.
            ``'precision'``
                ``s = sqrt(n_b**2 / n_eff)``; for blocks whose covariance scales
                as ``1 / w_j`` (e.g. length-proportional weights).

            Which one is "correct" is a statistics-convention decision, not a
            bug; ``'substitution'`` is the default so post-hoc reweighting is
            numerically identical to a fresh build-time-weighted run.

        Returns
        -------
        W : (num_blocks, num_blocks) ndarray
            The dense reweighting matrix.  It is all-zero when the weight mass
            collapses onto a single block (``n_eff <= 1``): ``'substitution'``
            warns and returns zeros (mirroring
            :meth:`_compute_hankel_cov_matrix`), ``'reliability'`` raises
            ``ValueError``, and ``'precision'`` degenerates to zeros on its own.
        """
        if convention not in ('substitution', 'reliability', 'precision'):
            raise ValueError(
                "'convention' must be one of ('substitution', 'reliability', "
                f"'precision'), got {convention!r}.")
        weights, n_eff = VarSSIRef._validate_weights(weights, num_blocks)
        if weights is None:
            weights = np.full(num_blocks, 1.0 / num_blocks)
        n_b = float(num_blocks)

        if convention == 'reliability':
            if not n_eff > 1:
                raise ValueError(
                    "'reliability' convention is undefined for n_eff <= 1 "
                    "(all weight mass concentrated on a single block).")
            s_w = np.sqrt(n_b * (n_b - 1.0) / (n_eff - 1.0))
        elif convention == 'precision':
            s_w = np.sqrt(n_b ** 2 / n_eff)
        else:  # 'substitution'
            if not n_eff > 1:
                logger.warning(
                    'Effective sample size n_eff=%.3f <= 1 (all weight concentrated '
                    'on a single block): no covariance information is available, the '
                    'block-weight factor is set to zero.', n_eff)
                return np.zeros((num_blocks, num_blocks))
            s_w = np.sqrt(n_b ** 2 * (n_b - 1.0) / (n_eff * (n_eff - 1.0)))

        # column j of W is s_w * sqrt(w_j) * (e_j - w):
        W = np.eye(num_blocks) - weights[:, np.newaxis]  # (I - w 1^T)
        W = s_w * W * np.sqrt(weights)[np.newaxis, :]  # ... @ diag(sqrt(w))
        return W

    def build_subspace_mat(
            self,
            num_block_columns,
            num_block_rows=None,
            num_blocks=None,
            subspace_method='covariance',
            weights=None,
            corr_matrices=None):
        '''
        Builds a Block-Hankel Matrix of Covariances with varying time lags

            |    R_1    R_2      ...    R_q    |
            |    R_2    R_3      ...    R_q+1  |
            |    ...    ...      ...    ...    |
            |    R_p+1  ...      ...    R_p+q  |

        Parameters ``weights`` and ``corr_matrices`` (covariance method only)
        turn the estimator into a weighted statistic over independent
        correlation estimates:

        weights : (num_blocks,) array, optional
            Non-negative probability weights, one per block; renormalized to
            sum to one.  ``None`` (default) keeps the classical unweighted
            (uniform) averaging.
        corr_matrices : (num_blocks, n_l, n_r, >=m_lags) array, optional
            Externally computed correlation estimates, one per block (e.g.
            one per aleatory sample), replacing the correlation functions of
            the attached ``prep_signals``.  Unlike the internal path, no
            block-count inflation is applied, since each entry is a
            standalone estimate rather than a fragment of one shared signal.
        '''
        if not isinstance(num_block_columns, int):
            raise TypeError(
                f"Expected int for 'num_block_columns', got {type(num_block_columns).__name__!r}.")
        if num_block_rows is None:
            num_block_rows = num_block_columns  # -10
        if not isinstance(num_block_rows, int):
            raise TypeError(
                f"Expected int for 'num_block_rows', got {type(num_block_rows).__name__!r}.")
        if subspace_method not in ['covariance', 'projection']:
            raise ValueError(
                f"'subspace_method' must be one of {['covariance', 'projection']}, got {subspace_method!r}.")
        if subspace_method == 'projection' and (
                weights is not None or corr_matrices is not None):
            raise NotImplementedError(
                "'weights' and 'corr_matrices' are only supported with "
                "subspace_method='covariance'.")

        logger.info('Building subspace matrices with {}-based method...'.format(subspace_method))

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows
        self.subspace_method = subspace_method

        n_l = self.num_analised_channels
        n_r = self.num_ref_channels

        if subspace_method == 'covariance':
            num_blocks = self._build_subspace_covariance(
                num_block_columns, num_block_rows, num_blocks, n_l, n_r,
                weights=weights, corr_matrices=corr_matrices)
        else:
            num_blocks = self._build_subspace_projection(num_blocks)
            self.weights = None
            self.n_eff = float(num_blocks)
            self.external_corr = False

        self.num_blocks = num_blocks
        self.state[0] = True
        # Rebuilding the subspace matrix invalidates any state matrices,
        # sensitivities and modal params computed against the previous one.
        self.state[1] = False
        self.state[2] = False
        self.sensitivities_prepared = False

    def _build_subspace_covariance(
            self, num_block_columns, num_block_rows, num_blocks, n_l, n_r,
            weights=None, corr_matrices=None):
        """Build subspace matrix using the covariance-based method.

        With ``corr_matrices`` provided, each of its entries is treated as an
        independent, externally computed correlation estimate instead of a
        fragment of the attached signals; the block-count inflation of the
        internal path does not apply then (``scale_factor=1``).  ``weights``
        switches the block averaging to a weighted mean.
        """
        m_lags = num_block_rows + 1 + num_block_columns
        if corr_matrices is not None:
            corr_matrices = np.asarray(corr_matrices)
            if (corr_matrices.ndim != 4 or corr_matrices.shape[1] != n_l
                    or corr_matrices.shape[2] != n_r):
                raise ValueError(
                    f"Expected 'corr_matrices' of shape (num_blocks, {n_l}, {n_r},"
                    f" >={m_lags}), got {corr_matrices.shape}.")
            if corr_matrices.shape[3] < m_lags:
                raise ValueError(
                    f"'corr_matrices' provides {corr_matrices.shape[3]} lags, but the"
                    f" requested matrix dimensions need {m_lags}.")
            if num_blocks is not None and num_blocks != corr_matrices.shape[0]:
                raise ValueError(
                    f"'num_blocks' ({num_blocks}) contradicts the number of provided"
                    f" correlation estimates ({corr_matrices.shape[0]}).")
            num_blocks = corr_matrices.shape[0]
            scale_factor = 1
            external_corr = True
            logger.info(
                f'Assembling {num_blocks} Hankel matrices from externally provided'
                f' correlation estimates {num_block_columns} block-columns and'
                f' {num_block_rows + 1} block rows ')
        else:
            if num_blocks is None:
                if self.prep_signals.n_segments is not None:
                    num_blocks = self.prep_signals.n_segments
                else:
                    raise RuntimeError(
                        'Either num_blocks, or pre-computed correlation functions must be provided.')

            logger.info(
                f'Assembling {num_blocks} Hankel matrices using pre-computed correlation functions'
                f' {num_block_columns} block-columns and {num_block_rows + 1} block rows ')

            self._validate_covariance_dims(m_lags, num_blocks)
            self.prep_signals.correlation(m_lags, n_segments=num_blocks)
            corr_matrices = self.prep_signals.corr_matrices
            scale_factor = num_blocks
            external_corr = False

        weights, n_eff = self._validate_weights(weights, num_blocks)

        subspace_matrices = [
            self._corr_to_subspace_block(
                corr_matrices[n_block, ...], num_block_columns, num_block_rows, n_l, n_r,
                scale_factor)
            for n_block in range(num_blocks)]

        if weights is None:
            self.subspace_matrix = np.mean(subspace_matrices, axis=0)
        else:
            self.subspace_matrix = np.tensordot(weights, subspace_matrices, axes=(0, 0))
        self.subspace_matrices = subspace_matrices
        self.weights = weights
        self.n_eff = n_eff
        self.external_corr = external_corr
        return num_blocks

    def _validate_covariance_dims(self, m_lags, num_blocks):
        """Warn if precomputed correlation data is mismatched for covariance build."""
        max_lags = self.prep_signals.m_lags
        if max_lags is not None and max_lags < m_lags:
            logger.warning(
                'The pre-computed correlation function is too short for the requested matrix dimensions.')
        if self.prep_signals.n_segments is not None and num_blocks < self.prep_signals.n_segments:
            logger.warning(
                'The pre-computed correlation function does not have the requested number of blocks.')

    @staticmethod
    def _corr_to_subspace_block(corr_matrix, num_block_columns, num_block_rows, n_l, n_r, scale_factor):
        """Assemble one Hankel block from a correlation matrix slice.

        ``scale_factor`` compensates block-wise correlation estimates that
        stem from splitting one signal into ``num_blocks`` fragments; for
        standalone (external) correlation estimates it must be 1.
        """
        this_subspace_matrix = np.zeros(
            ((num_block_rows + 1) * n_l, num_block_columns * n_r))
        for ii in range(num_block_columns):
            this_block_column = corr_matrix[:, :, ii + 1:num_block_rows + 1 + ii + 1] * scale_factor
            for i in range(num_block_rows + 1):
                this_subspace_matrix[i * n_l:(i + 1) * n_l, ii * n_r:(ii + 1) * n_r] = \
                    this_block_column[:, :, i]
        return this_subspace_matrix

    def _build_subspace_projection(self, num_blocks):
        """Build subspace matrix using the projection-based method."""
        num_block_columns = self.num_block_columns
        num_block_rows = self.num_block_rows
        total_time_steps = self.prep_signals.total_time_steps
        measurement = self.prep_signals.signals
        ref_channels = sorted(self.prep_signals.ref_channels)
        n_l = self.num_analised_channels
        n_r = self.num_ref_channels

        if num_blocks is None:
            logger.info('Argument num_blocks was no provided, default num_blocks = 50')
            num_blocks = 50

        # q == p == num_block_rows for the projection method
        block_length = int(np.floor((total_time_steps - 2 * num_block_rows) / num_blocks))
        if block_length < n_r * num_block_rows:
            raise RuntimeError(
                'Block-length (={}) may not be smaller than the number of reference channels * '
                'number of block rows (={})! \n Lower the number of blocks (={}), lower the number '
                'of reference channels (={}) or lower the number of block rows(={})!'.format(
                    block_length, n_r * num_block_rows, num_blocks, n_r, num_block_rows))

        N = block_length * num_blocks
        Y_minus = np.zeros((num_block_rows * n_r, N))
        Y_plus = np.zeros(((num_block_rows + 1) * n_l, N))
        for ii in range(num_block_rows):
            Y_minus[(num_block_rows - ii - 1) * n_r:(num_block_rows - ii) * n_r, :] = \
                measurement[(ii):(ii + N), ref_channels].T
        for ii in range(num_block_rows + 1):
            Y_plus[ii * n_l:(ii + 1) * n_l, :] = \
                measurement[(num_block_rows + ii):(num_block_rows + ii + N)].T

        Hankel_matrix = np.vstack((Y_minus, Y_plus))
        hankel_matrices = np.hsplit(
            Hankel_matrix,
            np.arange(block_length, block_length * num_blocks, block_length))

        for n_block in range(num_blocks):
            hankel_matrices[n_block] /= np.sqrt(block_length) * num_blocks

        H_dat_matrices, R_11_matrices = self._projection_qr_step(
            hankel_matrices, num_blocks, num_block_columns, n_l, n_r, num_block_rows)

        _L_breve, Q_breve = lq_decomp(
            np.hstack(R_11_matrices), mode='reduced', unique=True)
        Q_11_matrices = np.hsplit(
            Q_breve,
            np.arange(
                n_r * num_block_columns,
                num_blocks * n_r * num_block_columns,
                n_r * num_block_columns))

        pbar = simplePbar(num_blocks)
        for n_block in range(num_blocks):
            next(pbar)
            H_dat_matrices[n_block] = H_dat_matrices[n_block].dot(Q_11_matrices[n_block].T)

        self.subspace_matrices = H_dat_matrices
        self.subspace_matrix = np.mean(H_dat_matrices, axis=0)
        return num_blocks

    def _projection_qr_step(
            self, hankel_matrices, num_blocks, num_block_columns, n_l, n_r, p):
        """Perform the first QR-decomposition pass for the projection method."""
        H_dat_matrices = []
        R_11_matrices = []
        pbar = simplePbar(num_blocks)
        for n_block in range(num_blocks):
            next(pbar)
            L = lq_decomp(hankel_matrices[n_block], mode='r', unique=True)
            R11 = L[0:n_r * num_block_columns, 0:n_r * num_block_columns]
            R_11_matrices.append(R11)
            R21 = L[
                n_r * num_block_columns:n_r * num_block_columns + n_l * (p + 1),
                0:n_r * num_block_columns]
            H_dat_matrices.append(R21)
        return H_dat_matrices, R_11_matrices

    def plot_covariances(self):
        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns
        num_ref_channels = self.prep_signals.num_ref_channels
        num_analised_channels = self.prep_signals.num_analised_channels

#         subspace_matrices = []
#         for n_block in range(self.num_blocks):
#             corr_matrix = self.corr_matrices[n_block]
#             this_subspace_matrix= np.zeros(((num_block_rows+1)*num_analised_channels, num_block_columns*num_ref_channels))
#             for block_column in range(num_block_columns):
#                 this_block_column = corr_matrix[block_column*num_analised_channels:(num_block_rows+1+block_column)*num_analised_channels,:]
#                 this_subspace_matrix[:,block_column*num_ref_channels:(block_column+1)*num_ref_channels]=this_block_column
#             subspace_matrices.append(this_subspace_matrix)
        # self.subspace_matrices = subspace_matrices
        # subspace_matrices = self.subspace_matrices

        import matplotlib.pyplot as plot
        matrices = self.subspace_matrices + [self.subspace_matrix]
        # matrices = [self.subspace_matrix]
        for subspace_matrix in matrices[0:]:
            plot.figure()
            for num_channel, ref_channel in enumerate(
                    self.prep_signals.ref_channels):
                inds = ([], [])
                for i in range(num_block_columns):
                    row = ref_channel
                    col = i * num_ref_channels + num_channel
                    inds[0].append(row)
                    inds[1].append(col)
                for ii in range(1, num_block_rows):
                    row = (ii) * num_analised_channels + ref_channel
                    col = (num_block_columns - 1) * \
                        num_ref_channels + num_channel
                    inds[0].append(row)
                    inds[1].append(col)
                means = subspace_matrix[inds]
                # print(means.shape, sigma_r[inds,inds].shape, len(inds))
                # plot.errorbar(range(num_block_rows+num_block_rows-1), means, yerr=np.sqrt(sigma_r[inds,inds]))
                # print(np.sqrt(sigma_r[inds,inds]))

                # plot.plot(vec_R[inds,0])
                # plot.plot(vec_R[inds,1])
                plot.plot(range(1, num_block_columns + num_block_rows), means)
            break

        plot.show()

    def compute_state_matrices(self, max_model_order=None, lsq_method='pinv'):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-hankel-matrix of covariances
        the state space model matrices are obtained by appropriate truncation
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        '''

        if max_model_order is not None:
            if not isinstance(max_model_order, int):
                raise TypeError(
                    f"Expected int for 'max_model_order', got {type(max_model_order).__name__!r}.")
        if not self.state[0]:
            raise RuntimeError("Call build_subspace_mat() first.")

        subspace_matrix = self.subspace_matrix
        num_channels = self.prep_signals.num_analised_channels
        num_block_rows = self.num_block_rows  # p
        logger.info('Computing state matrices with {}-based method...'.format(lsq_method))

        # [U,S,V_T] = np.linalg.svd(subspace_matrix,1)
        [U, S, V_T] = scipy.linalg.svd(subspace_matrix, 1)
        # [U,S,V_T] = scipy.sparse.linalg.svds(subspace_matrix,k=max_model_order)

        # print(S.shape)
        # choose highest possible model order
        if max_model_order is None:
            max_model_order = len(S)
        else:
            max_model_order = min(max_model_order, len(S))

        # print(S.shape)
        S_2 = np.diag(np.power(np.copy(S)[:max_model_order], 0.5))
        # print(U.shape)
        U = U[:,:max_model_order]
        # print(U.shape)
        V_T = V_T[:max_model_order,:]
        # import matplotlib.pyplot as plot
        # plot.plot(S_2)

        O = np.dot(U, S_2)
        # plot.matshow(O)
        # plot.show()

        self.O = O

        self.U = U
        self.S = S
        self.V_T = V_T

        C = O[:num_channels,:]

        O_up = O[:num_channels * num_block_rows,:]

        O_down = O[num_channels:num_channels * (num_block_rows + 1),:]

        if lsq_method == 'pinv':
            A = np.dot(np.linalg.pinv(O_up), O_down)

        elif lsq_method == 'qr':
            Q_nmax, R_nmax = np.linalg.qr(O_up)
            S_nmax = np.dot(Q_nmax.T, O_down)
            self.Q_nmax = Q_nmax
            self.R_nmax = R_nmax
            self.S_nmax = S_nmax
            A = np.linalg.solve(R_nmax, S_nmax)

        self.state_matrix = A
        self.output_matrix = C
        self.max_model_order = max_model_order
        self.lsq_method = lsq_method

        self.state[1] = True
        # Recomputing state matrices invalidates any sensitivities/modal
        # params computed against the previous ones.
        self.state[2] = False
        self.sensitivities_prepared = False

    def _compute_hankel_cov_matrix(
            self, num_block_rows, num_block_columns, num_channels, num_ref_channels, num_blocks):
        """Precompute the T (Hankel covariance) matrix for fast/projection algorithms.

        Unweighted: ``Cov = sum_n (x_n - mean)(x_n - mean)^T / (N^2 (N-1))``.
        Weighted (``self.weights`` set): the T columns are ``sqrt(w_n) (x_n -
        x_bar_w) / sqrt(n_eff (n_eff - 1))`` with Kish's ``n_eff``, which
        reduces to the unweighted form at uniform weights.
        """
        subspace_matrix = self.subspace_matrix
        subspace_matrices = self.subspace_matrices
        weights = self.weights
        T = np.zeros(
            ((num_block_rows + 1) * num_block_columns * num_channels * num_ref_channels,
             num_blocks))
        for n_block in range(num_blocks):
            T[:, n_block:n_block + 1] = vectorize(subspace_matrices[n_block] - subspace_matrix)
        if weights is None:
            if num_blocks > 1:
                T /= np.sqrt(num_blocks ** 2 * (num_blocks - 1))
        else:
            n_eff = self.n_eff
            if n_eff > 1:
                T *= np.sqrt(weights)[np.newaxis, :]
                T /= np.sqrt(n_eff * (n_eff - 1))
            else:
                logger.warning(
                    'Effective sample size n_eff=%.3f <= 1 (all weight concentrated on a '
                    'single block): no covariance information is available, the Hankel '
                    'covariance is set to zero.', n_eff)
                T[:] = 0
        self.hankel_cov_matrix = T
        return T

    def _compute_slow_sigma_r_s3(
            self, num_block_columns, num_block_rows, num_channels, num_ref_channels, num_blocks):
        """Precompute sigma_R and S3 for the slow covariance method."""
        corr_matrices = self.prep_signals.corr_matrices
        corr_mats_mean = self.prep_signals.corr_matrix
        dim = (num_block_columns + num_block_rows) * num_channels * num_ref_channels
        sigma_R = np.zeros((dim, dim))
        for n_block in range(num_blocks):
            this_corr = vectorize(corr_matrices[n_block]) - vectorize(corr_mats_mean)
            sigma_R += np.dot(this_corr, this_corr.T)
        sigma_R /= (num_blocks * (num_blocks - 1))
        self.sigma_R = sigma_R
        S3 = []
        for k in range(num_block_columns):
            S3.append(sparse.kron(
                sparse.identity(num_ref_channels),
                sparse.hstack([
                    sparse.csr_matrix(((num_block_rows + 1) * num_channels, k * num_channels)),
                    sparse.identity((num_block_rows + 1) * num_channels, format='csr'),
                    sparse.csr_matrix(((num_block_rows + 1) * num_channels,
                                       (num_block_columns - k - 1) * num_channels))])).T)
        self.S3 = sparse.hstack(S3).T

    def _slow_joh_per_mode(self, j, U, S, V_T, P_p1rqr0, subspace_matrix):
        """Compute per-SVD-mode B/C matrices for the slow J_OH loop."""
        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns
        num_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels
        v_j_T = V_T[j:j + 1, :]
        u_j = U[:, j:j + 1]
        s_j = S[j]
        B_j = sparse.vstack([
            sparse.hstack([
                sparse.identity((num_block_rows + 1) * num_channels),
                -1 / s_j * subspace_matrix]),
            sparse.hstack([
                -1 / s_j * subspace_matrix.T,
                sparse.identity(num_block_columns * num_ref_channels)])])
        C_j = 1 / s_j * sparse.vstack([
            sparse.kron(v_j_T,
                        sparse.identity((num_block_rows + 1) * num_channels) -
                        np.dot(u_j, u_j.T)),
            P_p1rqr0.T.dot(sparse.kron(
                u_j.T,
                sparse.identity(num_block_columns * num_ref_channels) -
                np.dot(v_j_T.T, v_j_T)).T).T])
        Bi_pinv = np.linalg.pinv(B_j.toarray())
        S3 = getattr(self, 'S3', None)
        # Always compute bc/vu for projection path; compute bcs3/vus3 for covariance path.
        bc = C_j.T.dot(Bi_pinv.T).T
        vu = np.kron(v_j_T.T, u_j).T
        bcs3 = C_j.dot(S3).T.dot(Bi_pinv.T).T if S3 is not None else None
        vus3 = S3.T.dot(np.kron(v_j_T.T, u_j)).T if S3 is not None else None
        return bcs3, vus3, bc, vu

    def _assemble_slow_joh(self, BCS3, vuS3, BC, vu, U, S, debug):
        """Assemble J_OHS3 / J_OH from per-mode accumulations."""
        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns
        num_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels
        max_model_order = self.max_model_order
        subspace_method = self.subspace_method
        S_half_diag = np.diag(np.power(np.copy(S)[:max_model_order], 0.5))
        S_mhalf_mat = np.dot(U[:, :max_model_order],
                             np.diag(np.power(np.copy(S)[:max_model_order], -0.5)))
        left_sel = sparse.hstack([
            sparse.identity((num_block_rows + 1) * num_channels, format='csr'),
            sparse.csr_matrix(((num_block_rows + 1) * num_channels,
                                num_block_columns * num_ref_channels))])
        S4 = np.zeros((max_model_order ** 2, max_model_order))
        for k in range(1, max_model_order + 1):
            S4[(k - 1) * max_model_order + k - 1, k - 1] += 1
        if subspace_method == 'covariance':
            self.J_OHS3 = (
                0.5 * sparse.kron(sparse.identity(max_model_order), S_mhalf_mat).dot(
                    S4).dot(np.vstack(vuS3)) +
                sparse.kron(S_half_diag, left_sel).dot(np.vstack(BCS3)))
        if subspace_method == 'projection' or debug:
            self.J_OH = (
                0.5 * sparse.kron(sparse.identity(max_model_order), S_mhalf_mat).dot(
                    S4).dot(np.vstack(vu)) +
                sparse.kron(S_half_diag, left_sel).dot(np.vstack(BC)))
        if debug:
            print('J_OH', np.allclose(
                self.J_OH, self.J_OH[:max_model_order * num_block_rows * num_channels, :]))

    def _compute_slow_joh_loop(self, U, S, V_T, debug):
        """Run the slow-algorithm per-SVD-mode loop to compute J_OH/J_OHS3."""
        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns
        num_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels
        max_model_order = self.max_model_order
        P_p1rqr0 = permutation(
            (num_block_rows + 1) * num_channels, num_block_columns * num_ref_channels)
        subspace_matrix = self.subspace_matrix
        # Accumulate per-mode arrays unconditionally; unused lists are discarded after.
        BCS3, vuS3, BC, vu = [], [], [], []
        pbar = simplePbar(max_model_order)
        for j in range(max_model_order):
            next(pbar)
            bcs3, vus3, bc, v = self._slow_joh_per_mode(
                j, U, S, V_T, P_p1rqr0, subspace_matrix)
            BCS3.append(bcs3)
            vuS3.append(vus3)
            BC.append(bc)
            vu.append(v)
        self._assemble_slow_joh(BCS3, vuS3, BC, vu, U, S, debug)

    def _fast_joht_per_order(self, order, U, S, V_T, T, subspace_matrix):
        """Compute J_OHT_j for one SVD order in the fast algorithm."""
        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns
        num_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels
        v_j_T = V_T[order:order + 1, :]
        u_j = U[:, order:order + 1]
        s_j = S[order]
        K_j = (np.identity(num_block_columns * num_ref_channels) +
               np.vstack([np.zeros((num_block_columns * num_ref_channels - 1,
                                    num_block_columns * num_ref_channels)), (2 * v_j_T)]) -
               np.dot(subspace_matrix.T, subspace_matrix) / (s_j ** 2))
        K_ji = np.linalg.inv(K_j)
        HK_j = np.dot(subspace_matrix, K_ji) / s_j
        B_j1 = np.hstack([
            np.identity((num_block_rows + 1) * num_channels),
            np.dot(HK_j, subspace_matrix.T / s_j -
                   np.vstack([np.zeros((num_block_columns * num_ref_channels - 1,
                                        (num_block_rows + 1) * num_channels)),
                              u_j.T])).dot(HK_j)])
        T_j1 = sparse.kron(sparse.identity(num_block_columns * num_ref_channels), u_j.T).dot(T)
        T_j2 = sparse.kron(v_j_T, sparse.identity((num_block_rows + 1) * num_channels)).dot(T)
        J_OHT_j = (
            0.5 * s_j ** (-0.5) * np.dot(u_j, T_j1.T.dot(v_j_T.T).T) +
            s_j ** (-0.5) * np.dot(B_j1, np.vstack([
                T_j2 - np.dot(u_j, T_j2.T.dot(u_j).T),
                T_j1 - np.dot(v_j_T.T, T_j1.T.dot(v_j_T.T).T)])))
        return J_OHT_j

    def _fast_jacobian_accumulate(self, order, J_OHT_j, Q1, Q2, Q3, J_OHT, Q4):
        """Accumulate Q1-Q4 and J_OHT for one order in the fast Jacobian loop."""
        num_block_rows = self.num_block_rows
        num_channels = self.prep_signals.num_analised_channels
        max_model_order = self.max_model_order
        lsq_method = self.lsq_method
        O = self.O
        O_up = O[:num_channels * num_block_rows, :]
        O_down = O[num_channels:num_channels * (num_block_rows + 1), :]
        beg, end = order, order + 1
        if lsq_method == 'pinv':
            Q1[beg * max_model_order:end * max_model_order, :] = \
                O_up.T.dot(J_OHT_j[:num_channels * num_block_rows, :])
            Q2[beg * max_model_order:end * max_model_order, :] = \
                O_down.T.dot(J_OHT_j[:num_channels * num_block_rows, :])
            Q3[beg * max_model_order:end * max_model_order, :] = \
                O_up.T.dot(J_OHT_j[num_channels:num_channels * (num_block_rows + 1), :])
        if J_OHT is not None:
            J_OHT[beg * (num_block_rows + 1) * num_channels:
                  end * (num_block_rows + 1) * num_channels, :] = J_OHT_j
        Q4[beg * num_channels:end * num_channels, :] = sparse.hstack([
            sparse.identity(num_channels, format='csr'),
            sparse.csr_matrix((num_channels, num_block_rows * num_channels))]).dot(J_OHT_j)

    def _compute_fast_qr_jacobians(self, U, S, V_T, T, num_blocks, debug):
        """Precompute J_OHT, Q1-Q4 for the fast algorithm."""
        num_block_rows = self.num_block_rows
        num_channels = self.prep_signals.num_analised_channels
        max_model_order = self.max_model_order
        lsq_method = self.lsq_method
        subspace_matrix = self.subspace_matrix
        Q1 = Q2 = Q3 = None
        if lsq_method == 'pinv':
            Q1 = np.zeros((max_model_order ** 2, num_blocks))
            Q2 = np.zeros((max_model_order ** 2, num_blocks))
            Q3 = np.zeros((max_model_order ** 2, num_blocks))
        J_OHT = np.zeros((max_model_order * (num_block_rows + 1) * num_channels, num_blocks))
        Q4 = np.zeros((max_model_order * num_channels, num_blocks))
        pbar = simplePbar(max_model_order)
        for order in range(max_model_order):
            next(pbar)
            J_OHT_j = self._fast_joht_per_order(order, U, S, V_T, T, subspace_matrix)
            self._fast_jacobian_accumulate(order, J_OHT_j, Q1, Q2, Q3, J_OHT, Q4)
        if lsq_method == 'qr':
            self.J_OHT = J_OHT
        if lsq_method == 'pinv':
            self.Q1 = Q1
            self.Q2 = Q2
            self.Q3 = Q3
        self.Q4 = Q4

    def _compute_qr_lsq_jacobians(
            self, O_up, O_down, S1, S2, num_block_rows, num_channels, max_model_order):
        """Precompute J_Rnmax / J_Snmax for the qr-based state matrix estimation."""
        R_nmax = self.R_nmax
        Q_nmax = self.Q_nmax
        print('J_Rnmax')
        S_3 = sparse.lil_matrix((max_model_order ** 2, max_model_order ** 2))
        for k in range(1, max_model_order + 1):
            S_3[(k - 1) * max_model_order + k - 1, (k - 1) * max_model_order + k - 1] += 1
        S_4 = sparse.lil_matrix((max_model_order ** 2, max_model_order ** 2))
        for k1 in range(1, max_model_order):
            for k2 in range(1, k1 + 1):
                S_4[k1 * max_model_order + k2 - 1, k1 * max_model_order + k2 - 1] += 1
        R_nmaxi = np.linalg.inv(R_nmax)
        P_nn = permutation(max_model_order, max_model_order)
        U_ = sparse.bsr_matrix(S_3 + S_4 + P_nn.T.dot(S_4.T).T).dot(
            sparse.kron(R_nmaxi.T, sparse.hstack([
                Q_nmax.T,
                sparse.bsr_matrix((max_model_order, num_channels))])))
        J_Rnmax = sparse.kron(R_nmax.T, sparse.identity(max_model_order)).dot(U_)
        P_rn = permutation(num_block_rows * num_channels, max_model_order)
        J_Snmax = (
            sparse.kron(O_down.T, sparse.identity(max_model_order)).dot(
                P_rn.dot(
                    sparse.kron(R_nmaxi.T, S1) -
                    sparse.kron(sparse.identity(max_model_order), Q_nmax).dot(U_))) +
            sparse.kron(sparse.identity(max_model_order), S2.T.dot(Q_nmax).T))
        self.J_Rnmax = J_Rnmax
        self.J_Snmax = J_Snmax

    def _prepare_sigma_and_T(
            self, variance_algo, subspace_method,
            num_block_rows, num_block_columns, num_channels, num_ref_channels, num_blocks):
        """Precompute T matrix and slow-algorithm sigma quantities."""
        T = None
        if variance_algo == 'fast' or subspace_method == 'projection':
            T = self._compute_hankel_cov_matrix(
                num_block_rows, num_block_columns, num_channels, num_ref_channels, num_blocks)
        if variance_algo == 'slow' and subspace_method == 'covariance':
            self._compute_slow_sigma_r_s3(
                num_block_columns, num_block_rows, num_channels, num_ref_channels, num_blocks)
        elif variance_algo == 'slow' and subspace_method == 'projection':
            self.sigma_H = T.dot(T.T)
        return T

    def prepare_sensitivities(self, variance_algo='fast', debug=False):
        """Prepare Jacobians and covariance matrices for variance propagation."""
        if variance_algo not in ['fast', 'slow']:
            raise ValueError(
                f"'variance_algo' must be one of {['fast', 'slow']}, got {variance_algo!r}.")
        if variance_algo == 'slow' and (self.weights is not None or self.external_corr):
            raise NotImplementedError(
                "Weighted or externally provided correlation estimates are only "
                "implemented for variance_algo='fast'.")

        logger.info('Preparing sensitivities for use with {} (co)variance algorithm...'.format(
            variance_algo))

        num_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels
        num_block_columns = self.num_block_columns
        num_block_rows = self.num_block_rows
        num_blocks = self.num_blocks
        subspace_method = self.subspace_method
        lsq_method = self.lsq_method
        max_model_order = self.max_model_order

        T = self._prepare_sigma_and_T(
            variance_algo, subspace_method,
            num_block_rows, num_block_columns, num_channels, num_ref_channels, num_blocks)

        U, S, V_T = self.U, self.S, self.V_T
        O = self.O
        O_up = O[:num_channels * num_block_rows, :]
        O_down = O[num_channels:num_channels * (num_block_rows + 1), :]
        S1 = sparse.hstack([
            sparse.identity(num_block_rows * num_channels, format='csr'),
            sparse.csr_matrix((num_block_rows * num_channels, num_channels))])
        S2 = sparse.hstack([
            sparse.csr_matrix((num_block_rows * num_channels, num_channels)),
            sparse.identity(num_block_rows * num_channels, format='csr')])

        if lsq_method == 'qr':
            self._compute_qr_lsq_jacobians(
                O_up, O_down, S1, S2, num_block_rows, num_channels, max_model_order)
        if variance_algo == 'slow':
            self._compute_slow_joh_loop(U, S, V_T, debug)
        if variance_algo == 'fast':
            self._compute_fast_qr_jacobians(U, S, V_T, T, num_blocks, debug)

        self.variance_algo = variance_algo
        self.state[1] = True
        self.state[2] = False
        self.sensitivities_prepared = True

    @staticmethod
    def _compute_freq_damp_from_eigval(lambda_i, sampling_rate, debug=False):
        """Convert a discrete-time eigenvalue to frequency and damping ratio."""
        a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
        b_i = np.log(np.abs(lambda_i))
        freq_i = np.sqrt(a_i ** 2 + b_i ** 2) * sampling_rate / 2 / np.pi
        damping_i = 100 * np.abs(b_i) / np.sqrt(a_i ** 2 + b_i ** 2)
        if debug:
            lambda_ci = np.log(complex(lambda_i)) * sampling_rate
            freq_i = np.abs(lambda_ci) / 2 / np.pi
            damping_i = -100 * np.real(lambda_ci) / np.abs(lambda_ci)
        return a_i, b_i, freq_i, damping_i

    def _compute_jacobian_fast_pinv(self, ed, On_up2i, PQ23, PQ1, Q4n, debug=False):
        """Fast-pinv per-eigenvalue Jacobian and variance computation.

        Also returns the block-column sensitivity factors ``U_fixi`` (2 x n_b:
        frequency, damping) and ``U_phii`` (2*n_l x n_b: real/imag mode-shape)
        whose row 2-norms are the standard deviations; caching them enables
        Tier A post-hoc reweighting (``U @ W``).
        """
        num_channels = self.prep_signals.num_analised_channels
        Q_i = sparse.kron(ed.Phi_i.T, sparse.identity(ed.order)).dot(
            PQ23 - ed.lambda_i * PQ1)
        J_liHT = (1 / np.dot(ed.Chi_i.T.conj(), ed.Phi_i) *
                  np.dot(ed.Chi_i.conj().T, np.dot(On_up2i, Q_i)))
        U_fixi = np.dot(ed.J_fixiili, np.vstack([np.real(J_liHT), np.imag(J_liHT)]))
        if debug:
            J_liHT = 1 / np.dot(ed.Chi_i.T.conj(), ed.Phi_i) * np.dot(
                ed.Chi_i.conj().T, np.linalg.solve(On_up2i, Q_i))
        var_fixi = np.einsum('ij,ij->i', U_fixi, U_fixi)
        J_PhiiHT = np.dot(
            np.linalg.pinv(ed.lambda_i * np.identity(ed.order) - ed.state_matrix),
            np.dot(
                np.identity(ed.order) - np.dot(ed.Phi_i, ed.Chi_i.T.conj()) /
                np.dot(ed.Chi_i.T.conj(), ed.Phi_i),
                np.dot(On_up2i, Q_i)))
        if debug:
            J_PhiiHT = np.dot(
                np.linalg.pinv(ed.lambda_i * np.identity(ed.order) - ed.state_matrix),
                np.dot(
                    np.identity(ed.order) - np.dot(ed.Phi_i, ed.Chi_i.T.conj()) /
                    np.dot(ed.Chi_i.T.conj(), ed.Phi_i),
                    np.linalg.solve(On_up2i, Q_i)))
        J_phiiHT = np.exp(-1j * ed.alpha_ik) * np.dot(
            -1j * np.power(ed.t_ik, -2) * np.dot(
                np.dot(ed.output_matrix[:, :ed.order], ed.Phi_i),
                np.hstack([-np.imag(ed.s_ik) * ed.e_k.T, np.real(ed.s_ik) * ed.e_k.T])) +
            np.hstack([np.identity(num_channels), 1j * np.identity(num_channels)]),
            np.vstack([
                np.dot(ed.output_matrix[:, :ed.order], np.real(J_PhiiHT)) +
                np.dot(np.kron(np.real(ed.Phi_i).T, np.identity(num_channels)), Q4n),
                np.dot(ed.output_matrix[:, :ed.order], np.imag(J_PhiiHT)) +
                np.dot(np.kron(np.imag(ed.Phi_i).T, np.identity(num_channels)), Q4n)]))
        U_phii = np.vstack([np.real(J_phiiHT), np.imag(J_phiiHT)])
        var_phii = np.einsum('ij,ij->i', U_phii, U_phii)
        return var_fixi, var_phii, U_fixi, U_phii

    def _compute_jacobian_fast_qr(self, ed, J_AHT, Q4n):
        """Fast-qr per-eigenvalue Jacobian and variance computation.

        Also returns the block-column sensitivity factors ``U_fixi`` (2 x n_b)
        and ``U_phii`` (2*n_l x n_b), as in :meth:`_compute_jacobian_fast_pinv`,
        for Tier A caching.
        """
        num_channels = self.prep_signals.num_analised_channels
        J_liA = 1 / np.dot(ed.Chi_i.T.conj(), ed.Phi_i) * np.kron(ed.Phi_i.T, ed.Chi_i.T.conj())
        J_liHT = np.dot(J_liA, J_AHT)
        U_fixi = np.dot(ed.J_fixiili, np.vstack([np.real(J_liHT), np.imag(J_liHT)]))
        var_fixi = np.einsum('ij,ij->i', U_fixi, U_fixi)
        J_PhiA = np.dot(
            np.linalg.pinv(ed.lambda_i * np.identity(ed.order) - ed.state_matrix),
            np.kron(ed.Phi_i.T, np.identity(ed.order) - np.dot(
                ed.Phi_i, ed.Chi_i.T.conj()) / np.dot(ed.Chi_i.T.conj(), ed.Phi_i)))
        J_PhiiHT = np.dot(J_PhiA, J_AHT)
        J_phiiHT = np.exp(-1j * ed.alpha_ik) * np.dot(
            -1j * np.power(ed.t_ik, -2) * np.dot(
                np.dot(ed.output_matrix[:, :ed.order], ed.Phi_i),
                np.hstack([-np.imag(ed.s_ik) * ed.e_k.T, np.real(ed.s_ik) * ed.e_k.T])) +
            np.hstack([np.identity(num_channels), 1j * np.identity(num_channels)]),
            np.vstack([
                np.dot(ed.output_matrix[:, :ed.order], np.real(J_PhiiHT)) +
                np.dot(np.kron(np.real(ed.Phi_i).T, np.identity(num_channels)), Q4n),
                np.dot(ed.output_matrix[:, :ed.order], np.imag(J_PhiiHT)) +
                np.dot(np.kron(np.imag(ed.Phi_i).T, np.identity(num_channels)), Q4n)]))
        U_phii = np.vstack([np.real(J_phiiHT), np.imag(J_phiiHT)])
        var_phii = np.einsum('ij,ij->i', U_phii, U_phii)
        return var_fixi, var_phii, U_fixi, U_phii

    def _compute_jacobian_slow(self, ed, sigma_AC):
        """Slow per-eigenvalue Jacobian and variance computation."""
        num_channels = self.prep_signals.num_analised_channels
        J_liA = 1 / np.dot(ed.Chi_i.T.conj(), ed.Phi_i) * np.kron(ed.Phi_i.T, ed.Chi_i.T.conj())
        J_fixiA = np.dot(ed.J_fixiili, np.vstack([np.real(J_liA), np.imag(J_liA)]))
        J_full = np.hstack([J_fixiA, np.zeros((2, num_channels * ed.order))])
        var_fixi = np.diag(J_full.dot(sigma_AC.dot(J_full.T)))
        J_PhiA = np.dot(
            np.linalg.pinv(ed.lambda_i * np.identity(ed.order) - ed.state_matrix),
            np.kron(ed.Phi_i.T, np.identity(ed.order) - np.dot(
                ed.Phi_i, ed.Chi_i.T.conj()) / np.dot(ed.Chi_i.T.conj(), ed.Phi_i)))
        J_phiiAC = np.exp(-1j * ed.alpha_ik) * np.dot(
            -1j * np.power(ed.t_ik, -2) * np.dot(
                np.dot(ed.output_matrix[:, 0:ed.order], ed.Phi_i),
                np.hstack([-np.imag(ed.s_ik) * ed.e_k.T, np.real(ed.s_ik) * ed.e_k.T])) +
            np.hstack([np.identity(num_channels), 1j * np.identity(num_channels)]),
            np.vstack([
                np.hstack([
                    np.dot(ed.output_matrix[:, 0:ed.order], np.real(J_PhiA)),
                    np.kron(np.real(ed.Phi_i).T, np.identity(num_channels))]),
                np.hstack([
                    np.dot(ed.output_matrix[:, 0:ed.order], np.imag(J_PhiA)),
                    np.kron(np.imag(ed.Phi_i).T, np.identity(num_channels))])]))
        J_phi_stacked = np.vstack([np.real(J_phiiAC), np.imag(J_phiiAC)])
        var_phii = np.diag(J_phi_stacked.dot(sigma_AC.dot(J_phi_stacked.T)))
        return var_fixi, var_phii

    def _compute_state_matrix_per_order(self, order, O, S1, S2, block_weight_factor=None):
        """Compute state matrix and Jacobians for a given model order.

        ``block_weight_factor`` (the dense ``W(w)`` of
        :meth:`_block_weight_factor`), when given, post-hoc reweights the fast
        ``J_OHT`` factor by ``J_OHT @ W`` before it enters ``J_AHT`` (qr path);
        ``None`` keeps the original (uniform) weighting.  Only the variance
        Jacobians are affected -- ``state_matrix`` is unchanged.
        """
        lsq_method = self.lsq_method
        variance_algo = self.variance_algo
        num_block_rows = self.num_block_rows
        num_channels = self.prep_signals.num_analised_channels
        On_up = O[:num_channels * num_block_rows, :order]
        J_AO = None
        J_AHT = None

        if lsq_method == 'pinv':
            On_down = O[num_channels:num_channels * (num_block_rows + 1), :order]
            state_matrix = np.dot(np.linalg.pinv(On_up), On_down)
            if variance_algo == 'slow':
                P_p1rn = permutation((num_block_rows + 1) * num_channels, order)
                J_AO = (
                    sparse.kron(sparse.identity(order), S2.T.dot(np.linalg.pinv(On_up).T).T) -
                    sparse.kron(state_matrix.T, S1.T.dot(np.linalg.pinv(On_up).T).T) +
                    P_p1rn.T.dot(np.kron(
                        S1.T.dot(On_down).T - S1.T.dot(np.dot(state_matrix.T, On_up.T).T).T,
                        np.linalg.inv(np.dot(On_up[:, :order].T, On_up[:, :order]))).T).T)
        else:  # qr
            R_nmax = self.R_nmax
            S_nmax = self.S_nmax
            J_Snmax = self.J_Snmax
            J_Rnmax = self.J_Rnmax
            S_n = S_nmax[:order, :order]
            R_ni = np.linalg.inv(R_nmax[:order, :order])
            state_matrix = np.dot(R_ni, S_n)
            rows = np.hstack(
                [np.arange(order) + i * self.max_model_order for i in range(order)])
            J_Rn = J_Rnmax[rows, :order * (num_block_rows + 1) * num_channels]
            J_Sn = J_Snmax[rows, :order * (num_block_rows + 1) * num_channels]
            J_AO = -dot(np.kron(state_matrix.T, R_ni), J_Rn) + \
                dot(sparse.kron(sparse.identity(order), R_ni), J_Sn)
            if variance_algo == 'slow':
                J_AO = J_AO[:order ** 2, :order * (num_block_rows + 1) * num_channels]
            elif variance_algo == 'fast':
                J_OHT = self.J_OHT
                J_OHT_n = J_OHT[:order * (num_block_rows + 1) * num_channels, :]
                if block_weight_factor is not None:
                    J_OHT_n = J_OHT_n @ block_weight_factor
                J_AHT = J_AO.dot(J_OHT_n)

        return state_matrix, J_AO, J_AHT, On_up

    def _compute_sigma_ac_slow(
            self, order, J_AO, num_block_rows, num_channels, subspace_method):
        """Compute sigma_AC for the slow variance algorithm."""
        J_CO = sparse.kron(
            sparse.identity(order),
            sparse.hstack([
                sparse.identity(num_channels, format='csr'),
                sparse.csr_matrix((num_channels, num_block_rows * num_channels))]))
        if subspace_method == 'covariance':
            AS3 = sparse.vstack([J_AO, J_CO]).dot(
                self.J_OHS3[:(num_block_rows + 1) * num_channels * order, :])
            return AS3.dot(self.sigma_R).dot(AS3.T)
        AS3 = sparse.vstack([J_AO, J_CO]).dot(
            self.J_OH[:(num_block_rows + 1) * num_channels * order, :])
        return AS3.dot(self.sigma_H).dot(AS3.T)

    def _setup_fast_variance_per_order(self, order, max_model_order, On_up, lsq_method,
                                       block_weight_factor=None):
        """Pre-compute fast-algorithm quantities for one model order.

        ``block_weight_factor`` (the dense ``W(w)`` of
        :meth:`_block_weight_factor`), when given, post-hoc reweights the cached
        block factors by right-multiplying the sliced ``Q1..Q4`` columns with
        it (``Qk @ W``); ``None`` leaves them at their original weighting.  The
        row-slicing and the reweighting commute (``W`` acts on the block axis),
        so the sliced factors are reweighted here for efficiency.
        """
        Q4n = self.Q4[:self.prep_signals.num_analised_channels * order, :]
        if block_weight_factor is not None:
            Q4n = Q4n @ block_weight_factor
        On_up2i = None
        PQ1 = None
        PQ23 = None
        if lsq_method == 'pinv':
            rows = np.hstack(
                [np.arange(order) + i * max_model_order for i in range(order)])
            Q1n = self.Q1[rows, :]
            Q2n = self.Q2[rows, :]
            Q3n = self.Q3[rows, :]
            if block_weight_factor is not None:
                Q1n = Q1n @ block_weight_factor
                Q2n = Q2n @ block_weight_factor
                Q3n = Q3n @ block_weight_factor
            On_up2 = np.dot(On_up.T, On_up)
            On_up2i = np.linalg.pinv(On_up2)
            P_nn = permutation(order, order)
            PQ1 = (P_nn + sparse.identity(order ** 2)).dot(Q1n)
            PQ23 = P_nn.dot(Q2n) + Q3n
        return Q4n, On_up2i, PQ1, PQ23

    def _compute_per_eigval(self, i, lambda_i, oc, vp, debug):
        """Compute modal param and variance for one eigenvalue.

        Parameters
        ----------
        oc : _OrderCtx
            Per-order context (eigenvectors, output_matrix, order, sampling_rate, state_matrix).
        vp : _VarParams
            Variance-algorithm-specific pre-computed inputs.
        """
        num_channels = self.prep_signals.num_analised_channels
        variance_algo = self.variance_algo
        lsq_method = self.lsq_method
        output_matrix = oc.output_matrix
        order = oc.order
        a_i, b_i, freq_i, damping_i = self._compute_freq_damp_from_eigval(
            lambda_i, oc.sampling_rate, debug)

        mode_shape_i = np.array(
            np.dot(output_matrix[:, 0:order], oc.eigvec_r[:, i]), dtype=complex)
        k = np.argmax(np.abs(mode_shape_i))
        s_ik = mode_shape_i[k]
        t_ik = np.abs(s_ik)
        alpha_ik = np.angle(s_ik)
        e_k = np.zeros((num_channels, 1))
        e_k[k, 0] = 1
        mode_shape_i *= np.exp(-1j * alpha_ik)

        Phi_i = oc.eigvec_r[:, i:i + 1]
        Chi_i = oc.eigvec_l[:, i:i + 1]

        tlambda_i = (b_i + 1j * a_i) * oc.sampling_rate
        J_fixiili = (
            oc.sampling_rate / ((np.abs(lambda_i) ** 2) * np.abs(tlambda_i)) *
            np.dot(
                np.dot(
                    np.array([[1 / (2 * np.pi), 0],
                              [0, 100 / (np.abs(tlambda_i) ** 2)]]),
                    np.array([[np.real(tlambda_i), np.imag(tlambda_i)],
                              [-(np.imag(tlambda_i) ** 2),
                               np.real(tlambda_i) * np.imag(tlambda_i)]])),
                np.array([[np.real(lambda_i), np.imag(lambda_i)],
                          [-np.imag(lambda_i), np.real(lambda_i)]])))

        ed = _EigvalData(lambda_i, Phi_i, Chi_i, J_fixiili, order,
                         oc.state_matrix, output_matrix, alpha_ik, t_ik, s_ik, e_k)
        # U_fixi / U_phii are the block-column sensitivity factors (only the
        # fast algorithm exposes them); they are cached for Tier A reweighting.
        U_fixi = U_phii = None
        if variance_algo == 'fast' and lsq_method == 'pinv':
            var_fixi, var_phii, U_fixi, U_phii = self._compute_jacobian_fast_pinv(
                ed, vp.On_up2i, vp.PQ23, vp.PQ1, vp.Q4n, debug)
        elif variance_algo == 'fast' and lsq_method == 'qr':
            var_fixi, var_phii, U_fixi, U_phii = self._compute_jacobian_fast_qr(
                ed, vp.J_AHT, vp.Q4n)
        else:
            var_fixi, var_phii = self._compute_jacobian_slow(ed, vp.sigma_AC)

        return freq_i, damping_i, mode_shape_i, var_fixi, var_phii, U_fixi, U_phii

    def _init_variance_cache(self, cache_mode, block_weight_factor):
        """Reset the Tier A caches for a fresh modal loop; return whether to cache.

        Only the unweighted factors are cacheable, since they are the baseline
        for later :meth:`apply_block_weights` calls.
        """
        do_cache = cache_mode is not None and block_weight_factor is None
        self.U_fixi_cache = {} if do_cache else None
        self.U_phii_cache = {} if (do_cache and cache_mode == 'full') else None
        return do_cache

    def _store_order_cache(self, order, cache_mode, fixi_stack, phii_stack):
        """Stack one order's per-mode variance factors into the Tier A caches."""
        if not fixi_stack:
            return
        self.U_fixi_cache[order] = np.stack(fixi_stack)
        if cache_mode == 'full':
            self.U_phii_cache[order] = np.stack(phii_stack)

    def _run_modal_order_loop(
            self, O, S1, S2, output_matrix, max_model_order, sampling_rate, debug,
            orders=None, block_weight_factor=None, cache_mode=None,
            cache_dtype=np.float32):
        """Run the per-order loop for compute_modal_params; return result arrays.

        ``orders`` restricts the loop to the given model orders; result rows
        for all other orders stay zero.  ``None`` evaluates all orders
        ``1..max_model_order-1`` as before.

        ``block_weight_factor`` (the dense ``W(w)`` of
        :meth:`_block_weight_factor`), when given, post-hoc reweights the fast
        variance factors (``Q1..Q4`` for pinv, ``J_OHT``/``Q4`` for qr) so the
        returned ``std_*`` arrays reflect the new block weights; the point
        estimates (eigenvalues, frequencies, damping, mode shapes) are
        identical to the unweighted loop.  Only meaningful for
        ``variance_algo='fast'``.

        ``cache_mode`` (``'full'`` / ``'freqdamp'`` / ``None``) populates the
        Tier A per-mode caches ``self.U_fixi_cache`` (and ``self.U_phii_cache``
        for ``'full'``) as ``cache_dtype`` arrays.  Caching is only done for the
        unweighted factors (``block_weight_factor is None``), so the cache is
        always the baseline for later :meth:`apply_block_weights` calls.
        """
        num_channels = self.prep_signals.num_analised_channels
        num_block_rows = self.num_block_rows
        variance_algo = self.variance_algo
        subspace_method = self.subspace_method
        eigenvalues = np.zeros((max_model_order, max_model_order), dtype=np.complex128)
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        std_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        std_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((num_channels, max_model_order, max_model_order), dtype=complex)
        std_mode_shapes = np.zeros((num_channels, max_model_order, max_model_order), dtype=complex)

        # Tier A caches: a fresh loop invalidates any stale cache (see helper).
        do_cache = self._init_variance_cache(cache_mode, block_weight_factor)

        if orders is None:
            orders = range(1, max_model_order)
        pbar = simplePbar(len(orders))
        for order in orders:
            next(pbar)
            state_matrix, J_AO, J_AHT, On_up = self._compute_state_matrix_per_order(
                order, O, S1, S2, block_weight_factor=block_weight_factor)
            eigval, eigvec_l, eigvec_r = scipy.linalg.eig(
                a=state_matrix, b=None, left=True, right=True)
            eigval, eigvec_l, eigvec_r = self.remove_conjugates(eigval, eigvec_l, eigvec_r)
            sigma_AC = None
            if variance_algo == 'slow':
                sigma_AC = self._compute_sigma_ac_slow(
                    order, J_AO, num_block_rows, num_channels, subspace_method)
            Q4n = On_up2i = PQ1 = PQ23 = None
            if variance_algo == 'fast':
                Q4n, On_up2i, PQ1, PQ23 = self._setup_fast_variance_per_order(
                    order, max_model_order, On_up, self.lsq_method,
                    block_weight_factor=block_weight_factor)
            vp = _VarParams(sigma_AC, J_AHT, Q4n, On_up2i, PQ1, PQ23)
            oc = _OrderCtx(eigvec_l, eigvec_r, output_matrix, order, sampling_rate, state_matrix)

            fixi_stack = []
            phii_stack = []
            for i, lambda_i in enumerate(eigval):
                (freq_i, damping_i, mode_shape_i, var_fixi, var_phii,
                 U_fixi, U_phii) = self._compute_per_eigval(i, lambda_i, oc, vp, debug)
                eigenvalues[order, i] = lambda_i
                modal_frequencies[order, i] = freq_i
                modal_damping[order, i] = damping_i
                mode_shapes[:, i, order] = mode_shape_i
                std_frequencies[order, i] = np.sqrt(var_fixi[0])
                std_damping[order, i] = np.sqrt(var_fixi[1])
                std_mode_shapes.real[:, i, order] = np.sqrt(var_phii[:num_channels])
                std_mode_shapes.imag[:, i, order] = np.sqrt(
                    var_phii[num_channels:2 * num_channels])
                if do_cache:
                    # np.asarray strips the np.matrix subclass the qr path
                    # produces (via scipy sparse .dot), so np.stack can lift the
                    # per-mode 2-D factors into a 3-D per-order array.
                    fixi_stack.append(np.asarray(U_fixi, dtype=cache_dtype))
                    if cache_mode == 'full':
                        phii_stack.append(np.asarray(U_phii, dtype=cache_dtype))
                if debug:
                    print('Frequency: {}, Std_Frequency: {}'.format(freq_i, std_frequencies[order, i]))
                    print('Damping: {}, Std_damping: {}'.format(damping_i, std_damping[order, i]))
                    print('Mode_Shape: {}, Std_Mode_Shape: {}'.format(
                        mode_shape_i, std_mode_shapes[:, i, order]))
            if do_cache:
                self._store_order_cache(order, cache_mode, fixi_stack, phii_stack)

        return (eigenvalues, modal_frequencies, std_frequencies,
                modal_damping, std_damping, mode_shapes, std_mode_shapes)

    def _compute_modal_params_impl(self, max_model_order, debug, orders,
                                   block_weight_factor, cache_mode=None,
                                   cache_dtype=np.float32):
        """Shared body of compute_modal_params / compute_modal_params_weighted.

        ``block_weight_factor`` is threaded into the per-order loop to post-hoc
        reweight the fast variance factors; ``None`` gives the classical
        (unweighted) result.  ``cache_mode``/``cache_dtype`` control the Tier A
        per-mode caches (see :meth:`_run_modal_order_loop`).
        """
        if max_model_order is not None:
            if max_model_order > self.max_model_order:
                raise ValueError(
                    f"max_model_order ({max_model_order}) must be <= self.max_model_order ({self.max_model_order}).")
            self.max_model_order = max_model_order
        if not self.sensitivities_prepared:
            raise RuntimeError("Call prepare_sensitivities() first.")
        if orders is not None:
            orders = [int(order) for order in orders]
            for order in orders:
                if not 1 <= order < self.max_model_order:
                    raise ValueError(
                        f"Model order {order} outside the valid range "
                        f"1..{self.max_model_order - 1}.")

        logger.info(
            'Computing modal parameters with {} (co)variance computation...'.format(
                self.variance_algo))

        num_channels = self.prep_signals.num_analised_channels
        num_block_rows = self.num_block_rows
        max_model_order = self.max_model_order

        S1 = sparse.hstack([
            sparse.identity(num_block_rows * num_channels, format='csr'),
            sparse.csr_matrix((num_block_rows * num_channels, num_channels))])
        S2 = sparse.hstack([
            sparse.csr_matrix((num_block_rows * num_channels, num_channels)),
            sparse.identity(num_block_rows * num_channels, format='csr')])

        results = self._run_modal_order_loop(
            self.O, S1, S2, self.output_matrix, max_model_order,
            self.prep_signals.sampling_rate, debug, orders=orders,
            block_weight_factor=block_weight_factor, cache_mode=cache_mode,
            cache_dtype=cache_dtype)

        (self.eigenvalues, self.modal_frequencies, self.std_frequencies,
         self.modal_damping, self.std_damping, self.mode_shapes, self.std_mode_shapes) = results
        self.state[2] = True

    def compute_modal_params(self, max_model_order=None, debug=False, qr=True,
                             orders=None, cache_variance_factors=None,
                             cache='full', cache_dtype=np.float32):
        """Compute modal parameters with variance estimation.

        Parameters
        ----------
        orders : list of int, optional
            Restrict the evaluation to these model orders (each in
            ``1..max_model_order-1``); result rows for all other orders stay
            zero.  Saves most of the per-order eigendecomposition/Jacobian
            cost when only a single sampled order is of interest.
        cache_variance_factors : bool, optional
            Whether to cache the per-mode variance factors ``U_fixi``/``U_phii``
            (Tier A) for millisecond post-hoc reweighting via
            :meth:`apply_block_weights`.  ``None`` (default) uses the object
            default set in the constructor.
        cache : {'full', 'freqdamp'}, optional
            What to cache when caching is on: ``'full'`` stores both the
            frequency/damping factor ``U_fixi`` and the mode-shape factor
            ``U_phii``; ``'freqdamp'`` stores only ``U_fixi`` (so
            ``apply_block_weights`` reweights frequency/damping stds but leaves
            the mode-shape stds untouched), saving the bulk of the memory.
        cache_dtype : numpy dtype, optional
            Storage dtype for the caches (default ``np.float32``; use
            ``np.float64`` for exact Tier A == Tier B agreement).
        """
        do_cache = (self.cache_variance_factors if cache_variance_factors is None
                    else cache_variance_factors)
        cache_mode = None
        if do_cache:
            if cache not in ('full', 'freqdamp'):
                raise ValueError(
                    f"'cache' must be 'full' or 'freqdamp', got {cache!r}.")
            cache_mode = cache
        self._compute_modal_params_impl(
            max_model_order, debug, orders, block_weight_factor=None,
            cache_mode=cache_mode, cache_dtype=cache_dtype)

    def compute_modal_params_weighted(self, weights, convention='substitution',
                                      max_model_order=None, orders=None, debug=False):
        """Recompute modal-parameter variances for post-hoc block weights (Tier B).

        Reruns the modal-order loop with the cached fast variance factors
        (``Q1..Q4`` / ``J_OHT``) right-multiplied by the block-weight matrix
        ``W(w)`` (see :meth:`_block_weight_factor`), so the ``std_*`` arrays are
        those of the reweighted estimator.  No SVD, sensitivity preparation, or
        extra memory is needed -- the cost is a single modal loop (seconds).
        The point estimates (frequencies, damping, mode shapes) are recomputed
        identically to :meth:`compute_modal_params`; only the variances change.

        Works for both ``subspace_method='covariance'`` and ``'projection'``
        (the fast Q-pipeline is identical).  Requires an *unweighted* build:
        the frozen-linearization identity is defined relative to the
        uniform-mean cached factors, so ``self.weights`` must be ``None``.

        .. note::
            For ``subspace_method='projection'`` the per-block samples are
            defined *after* the joint LQ normalization (the ``lq_decomp`` of the
            stacked ``R11`` matrices couples the blocks).  Holding that
            normalization fixed is consistent with the frozen-linearization
            semantics below, but the block independence the reweighting assumes
            is weaker than in the covariance method, so projection reweighting is
            the more approximate of the two.

        .. warning::
            This is a frozen-linearization (delta-method) reweighting: the point
            estimates and Jacobians stay at their original weighting.  It is
            first-order consistent for moderate weight changes but does NOT
            relocate a point estimate contaminated by a bad block -- for that
            use the build-time weighted path.  A warning is emitted once the
            effective sample size ``n_eff`` drops below ~10.

        Parameters
        ----------
        weights : (num_blocks,) array or None
            Non-negative block weights (renormalized to sum to one).  ``None``
            restores the uniform (unweighted) result.
        convention : {'substitution', 'reliability', 'precision'}
            Covariance-normalization convention for the reweighting scalar; see
            :meth:`_block_weight_factor`.  Default ``'substitution'`` matches a
            fresh build-time-weighted run.
        max_model_order, orders, debug
            As in :meth:`compute_modal_params`.
        """
        if not self.sensitivities_prepared:
            raise RuntimeError("Call prepare_sensitivities() first.")
        if self.variance_algo != 'fast':
            raise NotImplementedError(
                "compute_modal_params_weighted supports variance_algo='fast' "
                "only; slow-algorithm reweighting is not implemented here.")
        if self.weights is not None:
            raise NotImplementedError(
                "Post-hoc block reweighting requires an unweighted build "
                "(self.weights is None): the frozen-linearization identity is "
                "defined relative to the uniform-mean cached factors.  Rebuild "
                "without build-time 'weights=' to use this method.")

        weights_norm, n_eff = self._validate_weights(weights, self.num_blocks)
        if weights is not None and n_eff < 10:
            logger.warning(
                'Effective sample size n_eff=%.2f < 10: the block covariance '
                'estimate is itself noisy, so the reweighted variances should be '
                'treated with caution.', n_eff)

        block_weight_factor = self._block_weight_factor(
            weights, self.num_blocks, convention)
        self._compute_modal_params_impl(
            max_model_order, debug, orders, block_weight_factor=block_weight_factor)
        self.block_weights = weights_norm
        self.block_weight_convention = convention

    def apply_block_weights(self, weights=None, convention='substitution'):
        """Millisecond post-hoc block reweighting from the Tier A cache.

        Reweights the cached per-mode variance factors
        (``self.U_fixi_cache``/``self.U_phii_cache``, populated by
        :meth:`compute_modal_params` with ``cache_variance_factors=True``) by
        ``U @ W(w)`` and overwrites the ``std_frequencies``/``std_damping`` (and,
        for the ``'full'`` cache, ``std_mode_shapes``) entries in place.  This is
        the algebraic equivalent of :meth:`compute_modal_params_weighted` (Tier
        B) -- ``U @ W`` equals the recomputed reweighted factor because ``W`` is
        real and the factors are linear in the block columns -- but costs a
        handful of small matrix products rather than a full modal loop, at the
        price of the cache memory (and the ``cache_dtype`` precision, ``float32``
        by default).  The point estimates are untouched.

        ``weights=None`` restores the uniform (unweighted) stds.  Requires an
        unweighted build (``self.weights is None``); a warning is emitted once
        the effective sample size ``n_eff`` drops below ~10.  See
        :meth:`_block_weight_factor` for the ``convention`` parameter and the
        frozen-linearization caveat, and :meth:`compute_modal_params_weighted`
        for the weaker block-independence caveat under
        ``subspace_method='projection'``.

        There is no automatic fallback: if no cache is present (Tier A disabled,
        or an archive saved without it) a ``RuntimeError`` is raised; use
        :meth:`compute_modal_params_weighted` (Tier B) for the recompute path.
        """
        if self.U_fixi_cache is None:
            raise RuntimeError(
                "No Tier A variance-factor cache present: run "
                "compute_modal_params(cache_variance_factors=True) first, or use "
                "compute_modal_params_weighted() for the recompute (Tier B) path.")
        if self.weights is not None:
            raise NotImplementedError(
                "Post-hoc block reweighting requires an unweighted build "
                "(self.weights is None): the frozen-linearization identity is "
                "defined relative to the uniform-mean cached factors.")

        weights_norm, n_eff = self._validate_weights(weights, self.num_blocks)
        if weights is not None and n_eff < 10:
            logger.warning(
                'Effective sample size n_eff=%.2f < 10: the block covariance '
                'estimate is itself noisy, so the reweighted variances should be '
                'treated with caution.', n_eff)

        W = self._block_weight_factor(weights, self.num_blocks, convention)
        num_channels = self.prep_signals.num_analised_channels
        for order, U_fixi in self.U_fixi_cache.items():
            # U_fixi: (n_modes, 2, n_b); float32 @ float64 -> float64 (accurate).
            UW = U_fixi @ W
            var_fixi = np.einsum('mij,mij->mi', UW, UW)
            n_modes = U_fixi.shape[0]
            self.std_frequencies[order, :n_modes] = np.sqrt(var_fixi[:, 0])
            self.std_damping[order, :n_modes] = np.sqrt(var_fixi[:, 1])
            if self.U_phii_cache is not None:
                U_phii = self.U_phii_cache[order]  # (n_modes, 2*n_l, n_b)
                UWp = U_phii @ W
                var_phii = np.einsum('mij,mij->mi', UWp, UWp)
                self.std_mode_shapes.real[:, :n_modes, order] = \
                    np.sqrt(var_phii[:, :num_channels]).T
                self.std_mode_shapes.imag[:, :n_modes, order] = \
                    np.sqrt(var_phii[:, num_channels:2 * num_channels]).T
        self.block_weights = weights_norm
        self.block_weight_convention = convention

    def _collect_subspace_state(self):
        """Return dict of subspace-matrix entries for save_state."""
        d = {}
        d['self.subspace_method'] = self.subspace_method
        d['self.num_block_columns'] = self.num_block_columns
        d['self.num_block_rows'] = self.num_block_rows
        d['self.num_blocks'] = self.num_blocks
        d['self.subspace_matrix'] = self.subspace_matrix
        d['self.subspace_matrices'] = self.subspace_matrices
        if self.weights is not None:
            d['self.weights'] = self.weights
        d['self.external_corr'] = self.external_corr
        return d

    def _collect_variance_algo_state(self):
        """Return dict of variance-algorithm-specific entries for save_state."""
        d = {'self.variance_algo': self.variance_algo}
        if self.variance_algo == 'slow' and self.subspace_method == 'covariance':
            d['self.sigma_R'] = self.sigma_R
            d['self.S3'] = self.S3
            d['self.J_OHS3'] = self.J_OHS3
        if self.variance_algo == 'slow' and self.subspace_method == 'projection':
            d['self.sigma_H'] = self.sigma_H
            d['self.J_OH'] = self.J_OH
        if self.variance_algo == 'fast' or self.subspace_method == 'projection':
            d['self.hankel_cov_matrix'] = self.hankel_cov_matrix
        return d

    def _collect_lsq_state(self):
        """Return dict of LSQ-method-specific entries for save_state."""
        d = {'self.lsq_method': self.lsq_method}
        if self.lsq_method == 'qr':
            d['self.Q_nmax'] = self.Q_nmax
            d['self.R_nmax'] = self.R_nmax
            d['self.S_nmax'] = self.S_nmax
            d['self.J_Rnmax'] = self.J_Rnmax
            d['self.J_Snmax'] = self.J_Snmax
        if self.variance_algo == 'fast' and self.lsq_method == 'pinv':
            d['self.Q1'] = self.Q1
            d['self.Q2'] = self.Q2
            d['self.Q3'] = self.Q3
        if self.variance_algo == 'fast' and self.lsq_method == 'qr':
            d['self.J_OHT'] = self.J_OHT
        if self.variance_algo == 'fast':
            d['self.Q4'] = self.Q4
        return d

    def _collect_state_model_state(self):
        """Return dict of state-model and sensitivity entries for save_state."""
        d = {
            'self.max_model_order': self.max_model_order,
            'self.state_matrix': self.state_matrix,
            'self.output_matrix': self.output_matrix,
            'self.O': self.O,
            'self.U': self.U,
            'self.S': self.S,
            'self.V_T': self.V_T,
        }
        d.update(self._collect_variance_algo_state())
        d.update(self._collect_lsq_state())
        return d

    def _collect_modal_state(self):
        """Return dict of modal parameter entries for save_state."""
        d = {
            'self.eigenvalues': self.eigenvalues,
            'self.modal_frequencies': self.modal_frequencies,
            'self.modal_damping': self.modal_damping,
            'self.mode_shapes': self.mode_shapes,
            'self.std_frequencies': self.std_frequencies,
            'self.std_damping': self.std_damping,
            'self.std_mode_shapes': self.std_mode_shapes,
        }
        # Tier A per-mode caches (dicts order -> array) are stored as pickled
        # object arrays; absence on load simply disables Tier A.
        if self.U_fixi_cache is not None:
            d['self.U_fixi_cache'] = np.array(self.U_fixi_cache, dtype=object)
            if self.U_phii_cache is not None:
                d['self.U_phii_cache'] = np.array(self.U_phii_cache, dtype=object)
        if self.block_weights is not None:
            d['self.block_weights'] = self.block_weights
            d['self.block_weight_convention'] = self.block_weight_convention
        return d

    def save_state(self, fname):
        """Save the current object state to a compressed NumPy archive."""
        dirname, _ = os.path.split(fname)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {
            'self.state': self.state,
            'self.sensitivities_prepared': self.sensitivities_prepared,
            'self.setup_name': self.setup_name,
            'self.start_time': self.start_time,
        }
        if self.state[0]:
            out_dict.update(self._collect_subspace_state())
        if self.state[1]:
            out_dict.update(self._collect_state_model_state())
        if self.state[2]:
            out_dict.update(self._collect_modal_state())

        np.savez_compressed(fname, **out_dict)
        logger.info('Modal results saved to {}'.format(fname))

    @classmethod
    def _restore_subspace_state(cls, ssi_object, in_dict):
        """Restore subspace-matrix attributes from a loaded archive dict."""
        ssi_object.subspace_method = str(in_dict['self.subspace_method'])
        ssi_object.num_block_columns = int(in_dict['self.num_block_columns'])
        ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        ssi_object.num_blocks = int(in_dict['self.num_blocks'])
        if ssi_object.subspace_method == 'covariance':
            ssi_object.corr_mats_mean = in_dict.get('self.corr_mats_mean', None)
            ssi_object.corr_matrices = in_dict.get('self.corr_matrices', None)
        ssi_object.subspace_matrix = in_dict['self.subspace_matrix']
        ssi_object.subspace_matrices = in_dict['self.subspace_matrices']
        # Archives predating the weighted estimator have neither key; they
        # were always unweighted with internally computed correlations.
        weights = in_dict.get('self.weights', None)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
        weights, n_eff = cls._validate_weights(weights, ssi_object.num_blocks)
        ssi_object.weights = weights
        ssi_object.n_eff = n_eff
        ssi_object.external_corr = bool(in_dict.get('self.external_corr', False))
        logger.debug('Subspace Matrices Built: {}, {} block_rows'.format(
            ssi_object.subspace_method, ssi_object.num_block_rows))

    @classmethod
    def _restore_variance_algo_state(cls, ssi_object, in_dict):
        """Restore variance-algorithm-specific attributes from a loaded archive dict."""
        ssi_object.variance_algo = str(in_dict['self.variance_algo'])
        if ssi_object.variance_algo == 'slow' and ssi_object.subspace_method == 'covariance':
            ssi_object.sigma_R = in_dict['self.sigma_R']
            ssi_object.S3 = in_dict['self.S3']
            ssi_object.J_OHS3 = in_dict['self.J_OHS3']
        if ssi_object.variance_algo == 'slow' and ssi_object.subspace_method == 'projection':
            ssi_object.sigma_H = in_dict['self.sigma_H']
            ssi_object.J_OH = in_dict['self.J_OH']
        if ssi_object.variance_algo == 'fast' or ssi_object.subspace_method == 'projection':
            ssi_object.hankel_cov_matrix = in_dict['self.hankel_cov_matrix']

    @classmethod
    def _restore_lsq_state(cls, ssi_object, in_dict):
        """Restore LSQ-method-specific attributes from a loaded archive dict."""
        ssi_object.lsq_method = str(in_dict['self.lsq_method'])
        if ssi_object.lsq_method == 'qr':
            ssi_object.Q_nmax = in_dict['self.Q_nmax']
            ssi_object.R_nmax = in_dict['self.R_nmax']
            ssi_object.S_nmax = in_dict['self.S_nmax']
            ssi_object.J_Rnmax = in_dict['self.J_Rnmax']
            ssi_object.J_Snmax = in_dict['self.J_Snmax']
        if ssi_object.variance_algo == 'fast' and ssi_object.lsq_method == 'pinv':
            ssi_object.Q1 = in_dict['self.Q1']
            ssi_object.Q2 = in_dict['self.Q2']
            ssi_object.Q3 = in_dict['self.Q3']
        if ssi_object.variance_algo == 'fast' and ssi_object.lsq_method == 'qr':
            ssi_object.J_OHT = in_dict['self.J_OHT']
        if ssi_object.variance_algo == 'fast':
            ssi_object.Q4 = in_dict['self.Q4']

    @classmethod
    def _restore_state_model_state(cls, ssi_object, in_dict):
        """Restore state-model and sensitivity attributes from a loaded archive dict."""
        ssi_object.max_model_order = int(in_dict['self.max_model_order'])
        ssi_object.state_matrix = in_dict['self.state_matrix']
        ssi_object.output_matrix = in_dict['self.output_matrix']
        ssi_object.O = in_dict['self.O']
        ssi_object.U = in_dict['self.U']
        ssi_object.S = in_dict['self.S']
        ssi_object.V_T = in_dict['self.V_T']
        cls._restore_variance_algo_state(ssi_object, in_dict)
        cls._restore_lsq_state(ssi_object, in_dict)
        logger.debug('State Matrices and Sensitivities Computed: {} up to order {}'.format(
            ssi_object.lsq_method, ssi_object.max_model_order))

    @classmethod
    def _restore_modal_state(cls, ssi_object, in_dict):
        """Restore modal parameter attributes from a loaded archive dict."""
        ssi_object.eigenvalues = in_dict['self.eigenvalues']
        ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
        ssi_object.modal_damping = in_dict['self.modal_damping']
        ssi_object.mode_shapes = in_dict['self.mode_shapes']
        ssi_object.std_frequencies = in_dict['self.std_frequencies']
        ssi_object.std_damping = in_dict['self.std_damping']
        ssi_object.std_mode_shapes = in_dict['self.std_mode_shapes']
        # Tier A caches / post-hoc weights: absent in older archives, in which
        # case Tier A stays disabled (apply_block_weights would raise) but the
        # Tier B path via compute_modal_params_weighted still works.
        fixi_cache = in_dict.get('self.U_fixi_cache', None)
        if fixi_cache is not None:
            ssi_object.U_fixi_cache = fixi_cache.item()
            phii_cache = in_dict.get('self.U_phii_cache', None)
            ssi_object.U_phii_cache = phii_cache.item() if phii_cache is not None else None
        block_weights = in_dict.get('self.block_weights', None)
        if block_weights is not None:
            ssi_object.block_weights = np.asarray(block_weights, dtype=float)
            ssi_object.block_weight_convention = str(
                in_dict['self.block_weight_convention'])
        logger.debug('Modal Parameters Computed')

    @classmethod
    def load_state(cls, fname, prep_signals):
        """Load a previously saved state from a compressed NumPy archive."""
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' not in in_dict:
            return
        # bool(...): entries loaded straight out of the .npz archive are
        # numpy.bool_, not plain Python bool.
        state = [bool(s) for s in in_dict['self.state']]

        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"Expected PreProcessSignals for 'prep_signals', got {type(prep_signals).__name__!r}.")
        setup_name = str(in_dict['self.setup_name'].item())
        if setup_name != prep_signals.setup_name:
            raise ValueError(
                f"setup_name mismatch: file has {setup_name!r}, prep_signals has {prep_signals.setup_name!r}.")
        start_time = prep_signals.start_time
        if start_time != prep_signals.start_time:
            raise ValueError(
                f"start_time mismatch: got {start_time!r} vs {prep_signals.start_time!r}.")

        ssi_object = cls(prep_signals)
        ssi_object.state = state
        # Older archives (saved before sensitivities_prepared was tracked
        # separately) don't have this key - state[1] was the best available
        # signal at the time, since prepare_sensitivities() used to just
        # re-assert it.
        if 'self.sensitivities_prepared' in in_dict:
            ssi_object.sensitivities_prepared = bool(in_dict['self.sensitivities_prepared'])
        else:
            ssi_object.sensitivities_prepared = state[1]
        if state[0]:
            cls._restore_subspace_state(ssi_object, in_dict)
        if state[1]:
            cls._restore_state_model_state(ssi_object, in_dict)
        if state[2]:
            cls._restore_modal_state(ssi_object, in_dict)
        return ssi_object

#     @staticmethod
#     def rescale_mode_shape(modeshape, doehler_style=False):
#         #scaling of mode shape
#         if doehler_style:
#             k = np.argmax(np.abs(modeshape))
#             alpha = np.angle(modeshape[k])
#             return modeshape * np.exp(-1j*alpha)
#         else:
#             modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]


def main():
    pass


if __name__ == '__main__':
    main()
