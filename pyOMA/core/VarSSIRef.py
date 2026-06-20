# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Covariance-driven SSI with propagated parameter variances (VarSSIRef)."""
import scipy.sparse as sparse
import numpy as np
import scipy.linalg
import os
from collections import namedtuple

from .Helpers import rq_decomp, ql_decomp, lq_decomp, simplePbar, ConfigFile
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

    def __init__(self, prep_signals):
        """
        Parameters
        ----------
        prep_signals : PreProcessSignals
            Pre-processed signal object.
        """
        super().__init__(prep_signals)

        #             0         1           2
        # self.state= [Hankel, State Mat., Modal Par.
        self.state = [False, False, False]

        self.num_block_columns = None
        self.num_block_rows = None
        self.subspace_matrix = None

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

    def build_subspace_mat(
            self,
            num_block_columns,
            num_block_rows=None,
            num_blocks=None,
            subspace_method='covariance'):
        '''
        Builds a Block-Hankel Matrix of Covariances with varying time lags

            |    R_1    R_2      ...    R_q    |
            |    R_2    R_3      ...    R_q+1  |
            |    ...    ...      ...    ...    |
            |    R_p+1  ...      ...    R_p+q  |

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

        logger.info('Building subspace matrices with {}-based method...'.format(subspace_method))

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows
        self.subspace_method = subspace_method

        n_l = self.num_analised_channels
        n_r = self.num_ref_channels

        if subspace_method == 'covariance':
            num_blocks = self._build_subspace_covariance(
                num_block_columns, num_block_rows, num_blocks, n_l, n_r)
        else:
            num_blocks = self._build_subspace_projection(num_blocks)

        self.num_blocks = num_blocks
        self.state[0] = True

    def _build_subspace_covariance(
            self, num_block_columns, num_block_rows, num_blocks, n_l, n_r):
        """Build subspace matrix using the covariance-based method."""
        if num_blocks is None:
            if self.prep_signals.n_segments is not None:
                num_blocks = self.prep_signals.n_segments
            else:
                raise RuntimeError(
                    'Either num_blocks, or pre-computed correlation functions must be provided.')

        logger.info(
            f'Assembling {num_blocks} Hankel matrices using pre-computed correlation functions'
            f' {num_block_columns} block-columns and {num_block_rows + 1} block rows ')

        m_lags = num_block_rows + 1 + num_block_columns
        self._validate_covariance_dims(m_lags, num_blocks)
        self.prep_signals.correlation(m_lags, n_segments=num_blocks)
        corr_matrices = self.prep_signals.corr_matrices

        subspace_matrices = [
            self._corr_to_subspace_block(
                corr_matrices[n_block, ...], num_block_columns, num_block_rows, n_l, n_r, num_blocks)
            for n_block in range(num_blocks)]

        self.subspace_matrix = np.mean(subspace_matrices, axis=0)
        self.subspace_matrices = subspace_matrices
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
    def _corr_to_subspace_block(corr_matrix, num_block_columns, num_block_rows, n_l, n_r, num_blocks):
        """Assemble one Hankel block from a correlation matrix slice."""
        this_subspace_matrix = np.zeros(
            ((num_block_rows + 1) * n_l, num_block_columns * n_r))
        for ii in range(num_block_columns):
            this_block_column = corr_matrix[:, :, ii + 1:num_block_rows + 1 + ii + 1] * num_blocks
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

    def _compute_hankel_cov_matrix(
            self, num_block_rows, num_block_columns, num_channels, num_ref_channels, num_blocks):
        """Precompute the T (Hankel covariance) matrix for fast/projection algorithms."""
        subspace_matrix = self.subspace_matrix
        subspace_matrices = self.subspace_matrices
        T = np.zeros(
            ((num_block_rows + 1) * num_block_columns * num_channels * num_ref_channels,
             num_blocks))
        for n_block in range(num_blocks):
            T[:, n_block:n_block + 1] = vectorize(subspace_matrices[n_block] - subspace_matrix)
        if num_blocks > 1:
            T /= np.sqrt(num_blocks ** 2 * (num_blocks - 1))
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
        """Fast-pinv per-eigenvalue Jacobian and variance computation."""
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
        return var_fixi, var_phii

    def _compute_jacobian_fast_qr(self, ed, J_AHT, Q4n):
        """Fast-qr per-eigenvalue Jacobian and variance computation."""
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
        return var_fixi, var_phii

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

    def _compute_state_matrix_per_order(self, order, O, S1, S2):
        """Compute state matrix and Jacobians for a given model order."""
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
                J_AHT = J_AO.dot(J_OHT[:order * (num_block_rows + 1) * num_channels, :])

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

    def _setup_fast_variance_per_order(self, order, max_model_order, On_up, lsq_method):
        """Pre-compute fast-algorithm quantities for one model order."""
        Q4n = self.Q4[:self.prep_signals.num_analised_channels * order, :]
        On_up2i = None
        PQ1 = None
        PQ23 = None
        if lsq_method == 'pinv':
            rows = np.hstack(
                [np.arange(order) + i * max_model_order for i in range(order)])
            Q1n = self.Q1[rows, :]
            Q2n = self.Q2[rows, :]
            Q3n = self.Q3[rows, :]
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
        if variance_algo == 'fast' and lsq_method == 'pinv':
            var_fixi, var_phii = self._compute_jacobian_fast_pinv(
                ed, vp.On_up2i, vp.PQ23, vp.PQ1, vp.Q4n, debug)
        elif variance_algo == 'fast' and lsq_method == 'qr':
            var_fixi, var_phii = self._compute_jacobian_fast_qr(ed, vp.J_AHT, vp.Q4n)
        else:
            var_fixi, var_phii = self._compute_jacobian_slow(ed, vp.sigma_AC)

        return freq_i, damping_i, mode_shape_i, var_fixi, var_phii

    def _run_modal_order_loop(
            self, O, S1, S2, output_matrix, max_model_order, sampling_rate, debug):
        """Run the per-order loop for compute_modal_params; return result arrays."""
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

        pbar = simplePbar(max_model_order)
        for order in range(1, max_model_order):
            next(pbar)
            state_matrix, J_AO, J_AHT, On_up = self._compute_state_matrix_per_order(
                order, O, S1, S2)
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
                    order, max_model_order, On_up, self.lsq_method)
            vp = _VarParams(sigma_AC, J_AHT, Q4n, On_up2i, PQ1, PQ23)
            oc = _OrderCtx(eigvec_l, eigvec_r, output_matrix, order, sampling_rate, state_matrix)

            for i, lambda_i in enumerate(eigval):
                freq_i, damping_i, mode_shape_i, var_fixi, var_phii = self._compute_per_eigval(
                    i, lambda_i, oc, vp, debug)
                eigenvalues[order, i] = lambda_i
                modal_frequencies[order, i] = freq_i
                modal_damping[order, i] = damping_i
                mode_shapes[:, i, order] = mode_shape_i
                std_frequencies[order, i] = np.sqrt(var_fixi[0])
                std_damping[order, i] = np.sqrt(var_fixi[1])
                std_mode_shapes.real[:, i, order] = np.sqrt(var_phii[:num_channels])
                std_mode_shapes.imag[:, i, order] = np.sqrt(
                    var_phii[num_channels:2 * num_channels])
                if debug:
                    print('Frequency: {}, Std_Frequency: {}'.format(freq_i, std_frequencies[order, i]))
                    print('Damping: {}, Std_damping: {}'.format(damping_i, std_damping[order, i]))
                    print('Mode_Shape: {}, Std_Mode_Shape: {}'.format(
                        mode_shape_i, std_mode_shapes[:, i, order]))

        return (eigenvalues, modal_frequencies, std_frequencies,
                modal_damping, std_damping, mode_shapes, std_mode_shapes)

    def compute_modal_params(self, max_model_order=None, debug=False, qr=True):
        """Compute modal parameters with variance estimation."""
        if max_model_order is not None:
            if max_model_order > self.max_model_order:
                raise ValueError(
                    f"max_model_order ({max_model_order}) must be <= self.max_model_order ({self.max_model_order}).")
            self.max_model_order = max_model_order
        if not self.state[1]:
            raise RuntimeError("Call compute_modal_params() first.")

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
            self.prep_signals.sampling_rate, debug)

        (self.eigenvalues, self.modal_frequencies, self.std_frequencies,
         self.modal_damping, self.std_damping, self.mode_shapes, self.std_mode_shapes) = results
        self.state[2] = True


    def _collect_subspace_state(self):
        """Return dict of subspace-matrix entries for save_state."""
        d = {}
        d['self.subspace_method'] = self.subspace_method
        d['self.num_block_columns'] = self.num_block_columns
        d['self.num_block_rows'] = self.num_block_rows
        d['self.num_blocks'] = self.num_blocks
        d['self.subspace_matrix'] = self.subspace_matrix
        d['self.subspace_matrices'] = self.subspace_matrices
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
        return {
            'self.eigenvalues': self.eigenvalues,
            'self.modal_frequencies': self.modal_frequencies,
            'self.modal_damping': self.modal_damping,
            'self.mode_shapes': self.mode_shapes,
            'self.std_frequencies': self.std_frequencies,
            'self.std_damping': self.std_damping,
            'self.std_mode_shapes': self.std_mode_shapes,
        }

    def save_state(self, fname):
        """Save the current object state to a compressed NumPy archive."""
        dirname, _ = os.path.split(fname)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {
            'self.state': self.state,
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
        logger.debug('Modal Parameters Computed')

    @classmethod
    def load_state(cls, fname, prep_signals):
        """Load a previously saved state from a compressed NumPy archive."""
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' not in in_dict:
            return
        state = list(in_dict['self.state'])

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
