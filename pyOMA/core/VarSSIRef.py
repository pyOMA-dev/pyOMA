# -*- coding: utf-8 -*-
'''
pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2021  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import scipy.sparse as sparse
import numpy as np
import scipy.linalg
import os

import multiprocessing as mp
import ctypes as c

from .Helpers import rq_decomp, ql_decomp, lq_decomp, simplePbar
from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase

'''
..TODO ::
     * define unit tests to check functionality after changes
     * use covariance estimation from PreProcessSignals
     * optimize multi order qr-based estimation routine
     * iterate over conjugate indices instead of removing them --> SSI_Data MC
     * add mode-shape integration with variances
     * use monte-carlo sampling in the last step of variance propagation (see: `https://doi.org/10.1007/978-3-7091-0399-9_3` -> 5.3)

'''
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
    return np.reshape(matrix, (np.product(matrix.shape), 1), 'F')


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

    def __init__(self, prep_signals):
        '''
        channel definition: channels start at 0
        '''
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
        assert os.path.exists(conf_file)
        assert isinstance(prep_signals, PreProcessSignals)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_columns = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Use Multiprocessing:'
            multiprocessing = f.__next__().strip('\n').strip(' ') == 'yes'
            assert f.__next__().strip('\n').strip(' ') == 'Number of Blocks:'
            num_blocks = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(
                ' ') == 'Subspace Method (projection/covariance):'
            subspace_method = f.__next__().strip('\n').strip(' ')
            assert f.__next__().strip('\n').strip(' ') == 'LSQ Method for A (pinv/qr):'
            lsq_method = f.__next__().strip('\n').strip(' ')
            assert f.__next__().strip('\n').strip(' ') == 'Variance Algorithm (fast/slow):'
            variance_algo = f.__next__().strip('\n').strip(' ')

        ssi_object = cls(prep_signals)

        ssi_object.build_subspace_mat(
            num_block_columns,
            multiprocess=multiprocessing,
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
            multiprocess=True,
            num_blocks=50,
            subspace_method='covariance'):

        assert multiprocess
        '''
        Builds a Block-Hankel Matrix of Covariances with varying time lags

            |    R_i    R_i-1    ...    R_1    |
            |    R_i+1  R_i      ...    R_2    |
            |    ...    ...      ...    ...    |
            |    R_2i-1 ...      ...    R_i    |

        '''
        # print(multiprocess)
        assert isinstance(num_block_columns, int)
        if num_block_rows is None:
            num_block_rows = num_block_columns  # -10
        assert isinstance(num_block_rows, int)
        assert subspace_method in ['covariance', 'projection']

        logger.info('Building subspace matrices with {}-based method...'.format(subspace_method))

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows
        self.num_blocks = num_blocks
        self.subspace_method = subspace_method

        total_time_steps = self.prep_signals.total_time_steps
        ref_channels = sorted(self.prep_signals.ref_channels)
        # roving_channels = self.prep_signals.roving_channels
        measurement = self.prep_signals.signals
        num_analised_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels

        # ref_channels + roving_channels
        all_channels = list(range(num_analised_channels))
        # all_channels.sort()

        if subspace_method == 'covariance':
            block_length = int(np.floor(total_time_steps / num_blocks))
            m_lags = num_block_columns + num_block_rows
            if block_length <= m_lags:
                raise RuntimeError(
                    'Block length (={}) must be greater or equal to max time lag (={})'.format(
                        block_length, m_lags))
            # extract_length = block_length - m_lags

            corr_matrices_mem = []

            corr_mats_shape = (
                m_lags * num_analised_channels,
                num_ref_channels)
            for n_block in range(num_blocks):
                # shared memory, can be used by multiple processes
                # @UndefinedVariable
                corr_memory = mp.Array(
                    c.c_double, np.zeros(
                        (np.product(corr_mats_shape))))
                corr_matrices_mem.append(corr_memory)

            # signals*=float(np.sqrt(block_length))
            measurement_shape = measurement.shape
            measurement_memory = mp.Array(
                c.c_double, measurement.reshape(
                    measurement.size, 1))  # @UndefinedVariable

            # each process should have at least 10 blocks to compute, to reduce
            # overhead associated with spawning new processes
            n_proc = min(int(m_lags * num_blocks / 10), os.cpu_count())
            pool = mp.Pool(
                processes=n_proc,
                initializer=self.init_child_process,
                initargs=(
                    measurement_memory,
                    corr_matrices_mem))  # @UndefinedVariable

            iterators = []
            it_len = int(np.ceil(m_lags * num_blocks / n_proc))

            printsteps = np.linspace(0, m_lags * num_blocks, 100, dtype=int)

            curr_it = []
            i = 0
            for n_block in range(num_blocks):
                for tau in range(1, m_lags + 1):
                    i += 1
                    if i in printsteps:
                        curr_it.append([n_block, tau, True])
                    else:
                        curr_it.append((n_block, tau))
                    if len(curr_it) > it_len:
                        iterators.append(curr_it)
                        curr_it = []
            iterators.append(curr_it)

            for curr_it in iterators:
                pool.apply_async(
                    self.compute_covariance,
                    args=(
                        curr_it,
                        m_lags,
                        block_length,
                        ref_channels,
                        all_channels,
                        measurement_shape,
                        corr_mats_shape))

            pool.close()
            pool.join()

            corr_matrices = []
            for corr_mats_mem in corr_matrices_mem:
                corr_mats = np.frombuffer(
                    corr_mats_mem.get_obj()).reshape(corr_mats_shape)
                corr_matrices.append(corr_mats * num_blocks)

            self.corr_matrices = corr_matrices

            corr_mats_mean = np.mean(corr_matrices, axis=0)
            # corr_mats_mean = np.sum(corr_matrices, axis=0)
            # corr_mats_mean /= num_blocks - 1
            self.corr_mats_mean = corr_mats_mean
            # self.corr_mats_std = np.std(corr_matrices, axis=0)

            subspace_matrix = np.zeros(
                ((num_block_rows + 1) * num_analised_channels,
                 num_block_columns * num_ref_channels))
            for block_column in range(num_block_columns):
                this_block_column = corr_mats_mean[block_column * num_analised_channels:(
                    num_block_rows + 1 + block_column) * num_analised_channels,:]
                subspace_matrix[:, block_column *
                                num_ref_channels:(block_column +
                                                  1) *
                                num_ref_channels] = this_block_column
            self.subspace_matrix = subspace_matrix

            subspace_matrices = []
            for n_block in range(num_blocks):
                corr_matrix = corr_matrices[n_block]
                this_subspace_matrix = np.zeros(
                    ((num_block_rows + 1) * num_analised_channels,
                     num_block_columns * num_ref_channels))
                for block_column in range(num_block_columns):
                    this_block_column = corr_matrix[block_column * num_analised_channels:(
                        num_block_rows + 1 + block_column) * num_analised_channels,:]
                    this_subspace_matrix[:, block_column *
                                         num_ref_channels:(block_column +
                                                           1) *
                                         num_ref_channels] = this_block_column
                subspace_matrices.append(this_subspace_matrix)
            self.subspace_matrices = subspace_matrices

        if subspace_method == 'projection':

            q = num_block_rows
            p = num_block_rows
            block_length = int(
                np.floor(
                    (total_time_steps - q - p) / num_blocks))
            if block_length < num_ref_channels * q:
                raise RuntimeError(
                    'Block-length (={}) may not be smaller than the number of reference channels * number of block rows (={})! \n Lower the number of blocks (={}), lower the number of reference channels (={}) or lower the number of block rows(={})!'.format(
                        block_length,
                        num_ref_channels *
                        q,
                        num_blocks,
                        num_ref_channels,
                        q))
            # might omit some timesteps in favor of equally sized blocks
            N = block_length * num_blocks

            Y_minus = np.zeros((q * num_ref_channels, N))
            Y_plus = np.zeros(((p + 1) * num_analised_channels, N))

            for ii in range(q):
                Y_minus[(q - ii - 1) * num_ref_channels:(q - ii) *
                        num_ref_channels,:] = measurement[(ii):(ii + N), ref_channels].T

            for ii in range(p + 1):
                Y_plus[ii *
                       num_analised_channels:(ii +
                                              1) *
                       num_analised_channels,:] = measurement[(q +
                                                                ii):(q +
                                                                     ii +
                                                                     N)].T

            Hankel_matrix = np.vstack((Y_minus, Y_plus))
            # Hankel_matrix /=np.sqrt(N)
##################################
            # self.Hankel_matrix = Hankel_matrix/np.sqrt(N)

            hankel_matrices = np.hsplit(
                Hankel_matrix,
                np.arange(
                    block_length,
                    block_length *
                    num_blocks,
                    block_length))
            # print(Hankel_matrix.shape, block_length*num_blocks, total_time_steps)
            for n_block in range(num_blocks):
                # print(n_block, subspace_matrices[n_block].shape)
                hankel_matrices[n_block] /= np.sqrt(block_length) * num_blocks
                # hankel_matrices[n_block] /= np.sqrt(num_blocks)
##################################

    #         extract_length = int(np.floor(total_time_steps/num_blocks))
    #
    #
    #
    #
    #         hankel_matrices = []
    #
    #         for n_block in range(num_blocks):
    #             # Extract reference time series
    #             this_measurement = signals[n_block*extract_length:(n_block+1)*extract_length,:]
    #             N = extract_length - p - q - 1
    #
    #             all_channels = ref_channels + roving_channels
    #             all_channels.sort()
    #
    #             refs = this_measurement[:,ref_channels]
    #
    #             #print('Creating block Hankel matrix...')
    #
    #             Y_minus = np.zeros((q*num_ref_channels,N))
    #             Y_plus = np.zeros(((p+1)*num_analised_channels,N))
    #
    #
    #             for ii in range(q):
    #                 Y_minus[(q-ii-1)*num_ref_channels:(q-ii)*num_ref_channels,:] = refs[ii:(ii+N)].T
    #
    #             for ii in range(p+1):
    #                 Y_plus[ii*num_analised_channels:(ii+1)*num_analised_channels,:] = this_measurement[(q+ii):(q+ii+N)].T
    #
    #             Hankel_matrix = np.vstack((Y_minus,Y_plus))
    #             Hankel_matrix /= np.sqrt(N)
    #             subspace_matrices.append(Hankel_matrix)

#             self.hankel_matrices = hankel_matrices

            H_dat_matrices = []
            R_11_matrices = []

            pbar = simplePbar(num_blocks * 2)

            # could eventually be parallelized
            for n_block in range(num_blocks):
                next(pbar)
                # num_block_columns*num_ref_channels + p*num_analised_channels, N
                # L,Q = lq_decomp(self.hankel_matrices[n_block],
                # mode='reduced', unique=True)#eventually change mode to 'r'
                # and omit Q
                L = lq_decomp(hankel_matrices[n_block], mode='r', unique=True)
                # num_block_columns*num_ref_channels + p*num_analised_channels,K; K, N

                R11 = L[0:num_ref_channels * num_block_columns,
                        0:num_ref_channels * num_block_columns]
                R_11_matrices.append(R11)

                R21 = L[num_ref_channels *
                        num_block_columns:num_ref_channels *
                        num_block_columns +
                        num_analised_channels *
                        (p +
                         1), 0:num_ref_channels *
                        num_block_columns]
                H_dat_matrices.append(R21)

            # num_ref_channels*num_block_columns,n_blocks*num_ref_channels*num_block_columns
            R_11_matrices = np.hstack(R_11_matrices)
            L_breve, Q_breve = lq_decomp(
                R_11_matrices, mode='reduced', unique=True)
            # num_ref_channels*num_block_columns,K;K,n_blocks*num_ref_channels*num_block_columns

            Q_11_matrices = np.hsplit(
                Q_breve,
                np.arange(
                    num_ref_channels *
                    num_block_columns,
                    num_blocks *
                    num_ref_channels *
                    num_block_columns,
                    num_ref_channels *
                    num_block_columns))

            for n_block in range(num_blocks):
                next(pbar)
                H_dat_matrix = H_dat_matrices[n_block]
                Q_11_matrix = Q_11_matrices[n_block]

                H_dat_matrices[n_block] = H_dat_matrix.dot(Q_11_matrix.T)

            # M = np.mean(H_dat_matrices, axis = 0)
            M = np.sum(H_dat_matrices, axis=0)
            M /= np.sqrt(num_blocks)

            # L,Q = lq_decomp(self.Hankel_matrix, mode='reduced')#, unique=True)#eventually change mode to 'r' and omit Q
            # # q*num_ref_channels + p*num_analised_channels,K; K, N
            # R21 = L[num_ref_channels*q:num_ref_channels*q + num_analised_channels*(p+1) , 0:num_ref_channels*q]
            # M = L[num_ref_channels*q:num_ref_channels*q + num_analised_channels*(p+1) , 0:num_ref_channels*q]

            self.subspace_matrices = H_dat_matrices
            self.subspace_matrix = np.mean(H_dat_matrices, axis=0)
            # self.M = M
        self.state[0] = True

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
        # matrices = subspace_matrices+[self.subspace_matrix]
        matrices = [self.subspace_matrix]
        for subspace_matrix in matrices:
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

        plot.show()

    def init_child_process(self, measurement_memory_, corr_matrices_mem_):
        # make the  memory arrays available to the child processes

        global measurement_memory
        measurement_memory = measurement_memory_

        global corr_matrices_mem
        corr_matrices_mem = corr_matrices_mem_

    def compute_covariance(
            self,
            curr_it,
            m_lags,
            block_length,
            ref_channels,
            all_channels,
            measurement_shape,
            corr_mats_shape,
            detrend=False):

        overlap = True

        # sys.stdout.flush()
        # normalize=False
        for this_it in curr_it:
            if len(this_it) > 2:
                print('.', end='', flush=True)
                del this_it[2]
            n_block, tau = this_it
            num_analised_channels = len(all_channels)

            measurement = np.frombuffer(
                measurement_memory.get_obj()).reshape(measurement_shape)
            if overlap:
                this_measurement = measurement[(
                    n_block) * block_length:(n_block + 1) * block_length + tau,:]  # /np.sqrt(block_length)
            else:
                this_measurement = measurement[(
                    n_block) * block_length:(n_block + 1) * block_length,:]

            if detrend:
                this_measurement = this_measurement - \
                    np.mean(this_measurement, axis=0)

            refs = (this_measurement[:-tau, ref_channels]).T

            current_signals = (this_measurement[tau:, all_channels]).T

            this_block = (np.dot(current_signals, refs.T)) / \
                current_signals.shape[0]

            corr_memory = corr_matrices_mem[n_block]

            corr_mats = np.frombuffer(
                corr_memory.get_obj()).reshape(corr_mats_shape)

            with corr_memory.get_lock():
                corr_mats[(tau - 1) * num_analised_channels:tau *
                          num_analised_channels,:] = this_block

    def compute_state_matrices(self, max_model_order=None, lsq_method='pinv'):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-hankel-matrix of covariances
        the state space model matrices are obtained by appropriate truncation
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        '''

        if max_model_order is not None:
            assert isinstance(max_model_order, int)
        assert lsq_method in ['pinv', 'qr']
        assert self.state[0]

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

    def prepare_sensitivities(self, variance_algo='fast', debug=False):

        assert variance_algo in ['fast', 'slow']

        logger.info('Preparing sensitivities for use with {} (co)variance algorithm...'.format(
            variance_algo))

        num_channels = self.prep_signals.num_analised_channels  # r
        num_ref_channels = self.prep_signals.num_ref_channels  # r_o
        num_block_columns = self.num_block_columns  # q
        num_block_rows = self.num_block_rows

        num_blocks = self.num_blocks
        subspace_method = self.subspace_method

        lsq_method = self.lsq_method
        max_model_order = self.max_model_order
        subspace_matrix = self.subspace_matrix

        # precomputation of T for fast algorithm
        if variance_algo == 'fast' or subspace_method == 'projection':
            subspace_matrices = self.subspace_matrices
            T = np.zeros(
                ((num_block_rows +
                  1) *
                 num_block_columns *
                 num_channels *
                 num_ref_channels,
                 num_blocks))

            # T *= np.sqrt(int(np.floor((self.prep_signals.total_time_steps-num_block_rows-num_block_columns)/num_blocks))*num_blocks)
            if 1:  # subspace_method == 'covariance':
                for n_block in range(num_blocks):
                    this_hankel = subspace_matrices[n_block]
                    T[:, n_block:n_block +
                        1] = vectorize(this_hankel - subspace_matrix)
                if num_blocks > 1:
                    # sqrt because, SIGMA = np.dot(T,T) squares up the
                    # denominator
                    T /= np.sqrt(num_blocks ** 2 * (num_blocks - 1))
#             elif subspace_method == 'projection':
#                 M = self.M
#                 for n_block in range(num_blocks):
#                     this_hankel = subspace_matrices[n_block]
#                     T[:,n_block:n_block+1]=vectorize(this_hankel-subspace_matrix)
#                 #T *= np.sqrt(int(np.floor((self.prep_signals.total_time_steps-num_block_rows-num_block_columns)/num_blocks))*num_blocks)
#                 T /= np.sqrt(num_blocks-1)
            self.hankel_cov_matrix = T

        # precomputation of Sigma_R and S3 for slow algorithm
        if variance_algo == 'slow' and subspace_method == 'covariance':
            corr_matrices = self.corr_matrices
            corr_mats_mean = self.corr_mats_mean

            sigma_R = np.zeros(
                ((num_block_columns + num_block_rows) * num_channels * num_ref_channels,
                 (num_block_columns + num_block_rows) * num_channels * num_ref_channels))
            for n_block in range(num_blocks):
                this_corr = vectorize(
                    corr_matrices[n_block]) - vectorize(corr_mats_mean)
                sigma_R += np.dot(this_corr, this_corr.T)
            sigma_R /= (num_blocks * (num_blocks - 1))
            self.sigma_R = sigma_R

            S3 = []
            for k in range(num_block_columns):
                S3.append(
                    sparse.kron(
                        sparse.identity(num_ref_channels),
                        sparse.hstack(
                            [
                                sparse.csr_matrix(
                                    ((num_block_rows + 1) * num_channels,
                                     (k) * num_channels)),
                                sparse.identity(
                                    (num_block_rows + 1) * num_channels,
                                    format='csr'),
                                sparse.csr_matrix(
                                    ((num_block_rows + 1) * num_channels,
                                     (num_block_columns - k - 1) * num_channels))])).T)
            S3 = sparse.hstack(S3).T
            self.S3 = S3

        elif variance_algo == 'slow' and subspace_method == 'projection':
            sigma_H = T.dot(T.T)
            self.sigma_H = sigma_H

        U = self.U
        S = self.S
        V_T = self.V_T

        O = self.O
        O_up = O[:num_channels * num_block_rows,:]
        O_down = O[num_channels:num_channels * (num_block_rows + 1),:]
        # Computation of Q_1 ... Q_4 in (36): For i = 1...n_b compute B_i,1 in (29) T_i,1 , T_i,2 (J_O,H T)_i in Remark 9 and the i-th block line of Q_1 ... Q_4 in (37)
        # S_1 in 3.1
        S1 = sparse.hstack(
            [
                sparse.identity(
                    (num_block_rows) *
                    num_channels,
                    format='csr'),
                sparse.csr_matrix(
                    ((num_block_rows) *
                     num_channels,
                     num_channels))])

        S2 = sparse.hstack(
            [
                sparse.csr_matrix(
                    ((num_block_rows) *
                     num_channels,
                     num_channels)),
                sparse.identity(
                    (num_block_rows) *
                    num_channels,
                    format='csr')])

        if debug:
            print(np.all(S1.dot(O) == O_up))
            print(np.all(S2.dot(O) == O_down))

        # Precomputation of J_AO for qr based state matrix computation
        if lsq_method == 'qr':
            ######
            # this whole method should be reformulated using less dot products
            # with selection matrices, but instead use slicing operations
            ######
            print('J_Rnmax')
            R_nmax = self.R_nmax
            Q_nmax = self.Q_nmax

            S_3 = sparse.lil_matrix((max_model_order ** 2, max_model_order ** 2))
            for k in range(1, max_model_order + 1):
                S_3[(k - 1) * max_model_order + k - 1,
                    (k - 1) * max_model_order + k - 1] += 1
            # S_3=S_3.toarray()
            S_4 = sparse.lil_matrix((max_model_order ** 2, max_model_order ** 2))
            for k1 in range(1, max_model_order - 1 + 1):
                for k2 in range(1, k1 + 1):
                    S_4[(k1) * max_model_order + (k2) - 1, (k1)
                        * max_model_order + (k2) - 1] += 1
            # S_4 = S_4.toarray()
            R_nmaxi = np.linalg.inv(R_nmax)

            P_nn = permutation(max_model_order, max_model_order)

            if debug:
                print(
                    np.all(S2.dot(O) == O[num_channels:num_channels * (num_block_rows + 1),:]))
                print(
                    np.all(S1.dot(O) == O[:num_channels * (num_block_rows),:]))
                a = np.random.random((max_model_order, max_model_order))
                b = vectorize(a)
                dia = S_3.dot(b)
                dia = dia[dia != 0]
                # print(dia)
                print(np.all(np.diag(a) == dia))
                print(
                    np.all(
                        np.triu(
                            a,
                            1) == np.reshape(
                            S_4.dot(b),
                            (max_model_order,
                             max_model_order),
                            order='F')))
            print(0)
            # first =  sparse.bsr_matrix(S_3 + S_4 + P_nn.T.dot(S_4.T).T)
            # print(0.1)
            # print(S1.shape, Q_nmax.T.shape, num_channels*num_block_rows,num_channels, Q_nmax.T.dot(S1.toarray()).shape)
            # second = sparse.kron(R_nmaxi.T,sparse.hstack([Q_nmax.T, sparse.bsr_matrix((max_model_order,num_channels))]))
            # print(0.2, type(first), type(second))
            # third = first.dot(second)
            # print(0.3)
            U_ = sparse.bsr_matrix(
                S_3 +
                S_4 +
                P_nn.T.dot(
                    S_4.T).T).dot(
                sparse.kron(
                    R_nmaxi.T,
                    sparse.hstack(
                        [
                            Q_nmax.T,
                            sparse.bsr_matrix(
                                (max_model_order,
                                 num_channels))])))
            print(1)
            J_Rnmax = sparse.kron(
                R_nmax.T, sparse.identity(max_model_order)).dot(U_)
            print(2)
            P_rn = permutation(num_block_rows * num_channels, max_model_order)
            print(3)
            first = sparse.kron(R_nmaxi.T, S1)
            print('a')
            second = sparse.kron(sparse.identity(max_model_order), Q_nmax)
            print('b')
            third = second.dot(U_)
            print('c')
            fourth = first - third
            print('d')
            fifth = P_rn.dot(fourth)
            print('e')
            sixth = sparse.kron(O_down.T, sparse.identity(max_model_order))
            print('f')
            seventh = sixth.dot(fifth)
            print('g')
            eighth = S2.T.dot(Q_nmax).T
            print('h')
            nineth = sparse.kron(sparse.identity(max_model_order), eighth)
            print('i')
            J_Snmax = seventh + nineth
#             J_Snmax = sparse.kron(O_down.T, sparse.identity(max_model_order)).dot(
#                              P_rn.dot(sparse.kron(R_nmaxi.T, S1)-
#                                       sparse.kron(sparse.identity(max_model_order),Q_nmax).dot(U_))
#                                       )\
#                         + sparse.kron(sparse.identity(max_model_order),S2.T.dot(Q_nmax).T)
            # print(4)
            self.J_Rnmax = J_Rnmax
            self.J_Snmax = J_Snmax

        # pre computation of I_OH at max model stratorder
        if variance_algo == 'slow':

            if subspace_method == 'covariance':
                BCS3 = []
                vuS3 = []

            if subspace_method == 'projection' or debug:
                BC = []
                vu = []

            P_p1rqr0 = permutation(
                (num_block_rows + 1) * num_channels,
                num_block_columns * num_ref_channels)

            S4 = np.zeros((max_model_order ** 2, max_model_order))
            for k in range(1, max_model_order + 1):
                S4[(k - 1) * max_model_order + k - 1, k - 1] += 1  # ?????

            pbar = simplePbar(max_model_order)
            for j in range(max_model_order):
                next(pbar)
                v_j_T = V_T[j:j + 1,:]
                u_j = U[:, j:j + 1]
                s_j = S[j]

                B_j = sparse.vstack([sparse.hstack([sparse.identity((num_block_rows + 1) * num_channels), -1 / s_j * subspace_matrix]),
                                     sparse.hstack([-1 / s_j * subspace_matrix.T, sparse.identity(num_block_columns * num_ref_channels)])])

                C_j = 1 / s_j * sparse.vstack([sparse.kron(v_j_T,
                                                           (sparse.identity((num_block_rows + 1) * num_channels)) - np.dot(u_j,
                                                                                                                           u_j.T)),
                                               P_p1rqr0.T.dot(sparse.kron(u_j.T,
                                                                          (sparse.identity(num_block_columns * num_ref_channels) - np.dot(v_j_T.T,
                                                                                                                                          v_j_T))).T).T])
                if subspace_method == 'covariance':
                    BCS3.append(
                        C_j.dot(S3).T.dot(
                            np.linalg.pinv(
                                B_j.toarray()).T).T)
                    vuS3.append(S3.T.dot(np.kron(v_j_T.T, u_j)).T)

                if subspace_method == 'projection' or debug:
                    BC.append(C_j.T.dot(np.linalg.pinv(B_j.toarray()).T).T)
                    vu.append(np.kron(v_j_T.T, u_j).T)

            if subspace_method == 'covariance':
                BCS3 = np.vstack(BCS3)
                vuS3 = np.vstack(vuS3)
                J_OHS3 = (0.5 * sparse.kron(sparse.identity(max_model_order),
                                            np.dot(self.U[:,:max_model_order],
                                                   np.diag(np.power(np.copy(self.S)[:max_model_order],
                                                                    -0.5)))).dot(S4).dot(vuS3) + sparse.kron(np.diag(np.power(np.copy(self.S)[:max_model_order],
                                                                                                                              0.5)),
                                                                                                             sparse.hstack([sparse.identity((num_block_rows + 1) * num_channels,
                                                                                                                                            format='csr'),
                                                                                                                            sparse.csr_matrix(((num_block_rows + 1) * num_channels,
                                                                                                                                               num_block_columns * num_ref_channels))])).dot(BCS3))

                self.J_OHS3 = J_OHS3

            if subspace_method == 'projection' or debug:
                BC = np.vstack(BC)
                vu = np.vstack(vu)

                J_OH = (0.5 * sparse.kron(sparse.identity(max_model_order),
                                          np.dot(self.U[:,:max_model_order],
                                                 np.diag(np.power(np.copy(self.S)[:max_model_order],
                                                                  -0.5)))).dot(S4).dot(vu) + sparse.kron(np.diag(np.power(np.copy(self.S)[:max_model_order],
                                                                                                                          0.5)),
                                                                                                         sparse.hstack([sparse.identity((num_block_rows + 1) * num_channels,
                                                                                                                                        format='csr'),
                                                                                                                        sparse.csr_matrix(((num_block_rows + 1) * num_channels,
                                                                                                                                           num_block_columns * num_ref_channels))])).dot(BC))
                self.J_OH = J_OH
            if debug:
                print('J_OH', np.allclose(
                    J_OH, self.J_OH[:max_model_order * num_block_rows * num_channels,:]))

        # Precomputation of J_OH*T for fast algorithm
        if variance_algo == 'fast':
            # print('J_OHT')
            if lsq_method == 'pinv':
                Q1 = np.zeros((max_model_order ** 2, num_blocks))
                Q2 = np.zeros((max_model_order ** 2, num_blocks))
                Q3 = np.zeros((max_model_order ** 2, num_blocks))
            if lsq_method == 'qr' or debug:
                J_OHT = np.zeros(
                    (max_model_order * (num_block_rows + 1) * num_channels, num_blocks))

            Q4 = np.zeros((max_model_order * num_channels, num_blocks))

            if debug:
                J_OH = np.zeros((max_model_order *
                                 (num_block_rows +
                                  1) *
                                 num_channels, num_block_columns *
                                 num_ref_channels *
                                 (num_block_rows +
                                  1) *
                                 num_channels))

            pbar = simplePbar(max_model_order)
            for order in range(max_model_order):
                next(pbar)

                beg, end = (order, order + 1)
                # beg,end=(i-1,i)
                v_j_T = V_T[beg:end,:]
                u_j = U[:, beg:end]
                s_j = S[beg]
                # print(S,s_i)

                # K_i, B_i,1;
                K_j = (np.identity(num_block_columns *
                                   num_ref_channels) +
                       np.vstack([np.zeros((num_block_columns *
                                            num_ref_channels -
                                            1, num_block_columns *
                                            num_ref_channels)), (2 *
                                                                 v_j_T)]) -
                       np.dot(subspace_matrix.T, subspace_matrix) /
                       (s_j ** 2))

                K_ji = np.linalg.inv(K_j)
                HK_j = np.dot(subspace_matrix, K_ji) / s_j
                B_j1 = np.hstack([np.identity((num_block_rows +
                                               1) *
                                              num_channels), np.dot(HK_j, subspace_matrix.T /
                                                                    s_j -
                                                                    np.vstack([np.zeros((num_block_columns *
                                                                                         num_ref_channels -
                                                                                         1, (num_block_rows +
                                                                                             1) *
                                                                                         num_channels)), u_j.T])).dot(HK_j)])

                # T_j,1; T_j,2

                T_j1 = sparse.kron(
                    sparse.identity(
                        num_block_columns *
                        num_ref_channels),
                    u_j.T).dot(T)
                T_j2 = sparse.kron(
                    v_j_T, sparse.identity(
                        (num_block_rows + 1) * num_channels)).dot(T)

                # (J_O,H T)_j

                J_OHT_j = (0.5 * s_j ** (-0.5) * np.dot(u_j,
                                                      T_j1.T.dot(v_j_T.T).T) + s_j ** (-0.5) * np.dot(B_j1,
                                                                                                    np.vstack([T_j2 - np.dot(u_j,
                                                                                                                             T_j2.T.dot(u_j).T),
                                                                                                               T_j1 - np.dot(v_j_T.T,
                                                                                                                             T_j1.T.dot(v_j_T.T).T)])))

                if debug:
                    sol_hank_K_j = np.linalg.solve(K_j.T, subspace_matrix.T).T

                    B_j1_o = np.hstack([np.identity((num_block_rows +
                                                     1) *
                                                    num_channels) +
                                        np.dot(sol_hank_K_j /
                                               s_j, (subspace_matrix.T /
                                                     s_j -
                                                     np.vstack([np.zeros((num_block_columns *
                                                                          num_ref_channels -
                                                                          1, (num_block_rows +
                                                                              1) *
                                                                          num_channels)), u_j.T]))), sol_hank_K_j /
                                        s_j])

                    C_j = 1 / s_j * np.vstack(
                        [
                            np.dot(
                                np.identity(
                                    (num_block_rows + 1) * num_channels) - np.dot(
                                    u_j,
                                    u_j.T),
                                np.kron(
                                    v_j_T,
                                    np.identity(
                                        (num_block_rows + 1) * num_channels))),
                            np.dot(
                                np.identity(
                                    num_block_columns * num_ref_channels) - np.dot(
                                    v_j_T.T,
                                    v_j_T),
                                np.kron(
                                    np.identity(
                                        num_block_columns * num_ref_channels),
                                    u_j.T))])

                    J_OH[beg * (num_block_rows + 1) * num_channels:end * (num_block_rows + 1) * num_channels,:] = 0.5 * s_j ** (-0.5) * np.dot(u_j, np.kron(v_j_T.T, u_j).T) + s_j ** (0.5) * np.dot(B_j1, C_j)

                if lsq_method == 'pinv':
                    Q1[beg * max_model_order:end * max_model_order,:] = O_up.T.dot(J_OHT_j[:num_channels * num_block_rows,:])  # np.dot(np.dot(Oi_up.T,S1),J_OHTi)
                    Q2[beg * max_model_order:end * max_model_order,:] = O_down.T.dot(J_OHT_j[:num_channels * num_block_rows,:])  # np.dot(np.dot(Oi_down.T,S1),J_OHTi)
                    Q3[beg * max_model_order:end * max_model_order,:] = O_up.T.dot(J_OHT_j[num_channels:num_channels * (num_block_rows + 1),:])  # np.dot(np.dot(Oi_up.T,S2),J_OHTi)
                    # Q1[beg*max_model_order:end*max_model_order,:] = S1.T.dot(O_up).T.dot(J_OHT_j) #np.dot(np.dot(Oi_up.T,S1),J_OHTi)
                    # Q2[beg*max_model_order:end*max_model_order,:] = S1.T.dot(O_down).T.dot(J_OHT_j) #np.dot(np.dot(Oi_down.T,S1),J_OHTi)
                    # Q3[beg*max_model_order:end*max_model_order,:] =
                    # S2.T.dot(O_up).T.dot(J_OHT_j)
                    # #np.dot(np.dot(Oi_up.T,S2),J_OHTi)

                if lsq_method == 'qr' or debug:
                    J_OHT[beg * (num_block_rows + 1) * num_channels:end *
                          (num_block_rows + 1) * num_channels,:] = J_OHT_j

                Q4[beg * num_channels:end * num_channels,:] = sparse.hstack([sparse.identity(num_channels,
                                                       format='csr'),
                                       sparse.csr_matrix((num_channels,
                                                          (num_block_rows) * num_channels))]).dot(J_OHT_j)

            if debug:
                self.J_OH = J_OH
                print(np.allclose(np.dot(J_OH, T), J_OHT))

            if lsq_method == 'qr':
                self.J_OHT = J_OHT

            if lsq_method == 'pinv':
                self.Q1 = Q1
                self.Q2 = Q2
                self.Q3 = Q3

            self.Q4 = Q4

        self.variance_algo = variance_algo
        self.state[1] = True
        self.state[2] = False  # previous modal params are invalid now

    def compute_modal_params(self, max_model_order=None, debug=False, qr=True):
        if max_model_order is not None:
            assert max_model_order <= self.max_model_order
            self.max_model_order = max_model_order

        assert self.state[1]

        logger.info(
            'Computing modal parameters with {} (co)variance computation...'.format(
                self.variance_algo))

        state_matrix = self.state_matrix
        output_matrix = self.output_matrix

        O = self.O

        subspace_method = self.subspace_method
        lsq_method = self.lsq_method
        variance_algo = self.variance_algo
        max_model_order = self.max_model_order
        sampling_rate = self.prep_signals.sampling_rate

        num_channels = self.prep_signals.num_analised_channels
        # num_ref_channels = self.prep_signals.num_ref_channels
        # num_block_columns = self.num_block_columns
        num_block_rows = self.num_block_rows

        # accel_channels = self.prep_signals.accel_channels
        # velo_channels = self.prep_signals.velo_channels

        if lsq_method == 'qr':
            R_nmax = self.R_nmax
            S_nmax = self.S_nmax
            J_Snmax = self.J_Snmax
            J_Rnmax = self.J_Rnmax

        if variance_algo == 'slow' and subspace_method == 'covariance':
            J_OHS3 = self.J_OHS3
        if variance_algo == 'slow' and subspace_method == 'projection':
            J_OH = self.J_OH
            sigma_H = self.sigma_H
        if variance_algo == 'slow' and subspace_method == 'covariance':
            sigma_R = self.sigma_R

        if variance_algo == 'fast':
            Q4 = self.Q4
        if variance_algo == 'fast' and lsq_method == 'qr':
            J_OHT = self.J_OHT
        if variance_algo == 'fast' and lsq_method == 'pinv':
            Q1 = self.Q1
            Q2 = self.Q2
            Q3 = self.Q3

        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=np.complex128)
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        std_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        std_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros(
            (num_channels,
             max_model_order,
             max_model_order),
            dtype=complex)
        std_mode_shapes = np.zeros(
            (num_channels, max_model_order, max_model_order), dtype=complex)

        # for future parallelization, may not even be necessary if numpy is using Intel MKL
        # params: order, max_model_order, num_channels, accel_channels, velo_channels, self.prep_signals.channel_factors
        # read: state_matrix, output_matrix, Oi,  Q1,Q2,Q3,Q4,
        # functions: remove_conjugates(), integrate_quantities(), self.rescale_mode_shape()
        # write: modal_frequencies, std_frequencies, modal_damping,
        # std_damping, mode_shapes, std_mode_shapes

        S1 = sparse.hstack(
            [
                sparse.identity(
                    (num_block_rows) *
                    num_channels,
                    format='csr'),
                sparse.csr_matrix(
                    ((num_block_rows) *
                     num_channels,
                     num_channels))])

        S2 = sparse.hstack(
            [
                sparse.csr_matrix(
                    ((num_block_rows) *
                     num_channels,
                     num_channels)),
                sparse.identity(
                    (num_block_rows) *
                    num_channels,
                    format='csr')])
        pbar = simplePbar(max_model_order)
        for order in range(1, max_model_order):
            next(pbar)

            On_up = O[:num_channels * num_block_rows,:order]

            if lsq_method == 'pinv':
                On_down = O[num_channels:num_channels *
                            (num_block_rows + 1),:order]
                state_matrix = np.dot(np.linalg.pinv(On_up), On_down)

            if lsq_method == 'pinv' and variance_algo == 'slow':
                P_p1rn = permutation(
                    (num_block_rows + 1) * num_channels, order)
                J_AO = (sparse.kron(sparse.identity(order), S2.T.dot(np.linalg.pinv(On_up).T).T) -
                        sparse.kron(state_matrix.T, S1.T.dot(np.linalg.pinv(On_up).T).T) +
                        P_p1rn.T.dot(np.kron(S1.T.dot(On_down).T - S1.T.dot(np.dot(state_matrix.T,
                                                                                   On_up.T).T
                                                                            ).T,
                                             np.linalg.inv(np.dot(On_up[:,:order].T, On_up[:,:order]))).T
                                     ).T)
            if lsq_method == 'qr':
                # R_n = self.R_nmax[:order,:order]
                S_n = S_nmax[:order,:order]
                R_ni = np.linalg.inv(R_nmax[:order,:order])
                state_matrix = np.dot(R_ni, S_n)

                rows = np.hstack(
                    [np.arange(order) + i * max_model_order for i in range(order)])

                J_Rn = J_Rnmax[rows,:order *
                               (num_block_rows + 1) * num_channels]
                J_Sn = J_Snmax[rows,:order *
                               (num_block_rows + 1) * num_channels]
                J_AO = -dot(np.kron(state_matrix.T, R_ni), J_Rn) + \
                    dot(sparse.kron(sparse.identity(order), R_ni), J_Sn)

                # T_n = sparse.vstack((sparse.identity(order, format='csr'), sparse.csr_matrix((max_model_order-order,order))), format='csr')
                # J_AO = J_Snmax.T.dot(sparse.kron(T_n.T,T_n.dot(R_ni.T).T).T).T \
                #        -J_Rnmax.T.dot(sparse.kron(T_n.dot(state_matrix).T,T_n.dot(R_ni.T).T).T).T
                # print(np.all(J_AOe==J_AO[:order**2, :order*(num_block_rows+1)*num_channels]), J_AOe.shape, J_AO[:order**2, :order*(num_block_rows+1)*num_channels].shape)
                if variance_algo == 'slow':
                    J_AO = J_AO[:order ** 2,:order * (num_block_rows + 1) * num_channels]
                if variance_algo == 'fast':
                    # J_AHT = J_AO.dot(J_OHT)
                    J_AHT = J_AO.dot(
                        J_OHT[:order * (num_block_rows + 1) * num_channels,:])

            eigval, eigvec_l, eigvec_r = scipy.linalg.eig(
                a=state_matrix, b=None, left=True, right=True)
            # eigvec_r = eigvec_r.T
            eigval, eigvec_l, eigvec_r = self.remove_conjugates(
                eigval, eigvec_l, eigvec_r)

            if variance_algo == 'slow':
                # J_AO for pinv based
                # J_OHS3 = J_OHS3[:(num_block_rows+1)*num_channels*order,:]

                J_CO = sparse.kron(
                    sparse.identity(order), sparse.hstack(
                        [
                            sparse.identity(
                                num_channels, format='csr'), sparse.csr_matrix(
                                (num_channels, (num_block_rows) * num_channels))]))
                # print(J_AO.shape, J_CO.shape, J_OHS3.shape)
                if subspace_method == 'covariance':
                    AS3 = sparse.vstack([J_AO, J_CO]).dot(
                        J_OHS3[:(num_block_rows + 1) * num_channels * order,:])
                    sigma_AC = AS3.dot(sigma_R).dot(AS3.T)  # with sigma_R
                if subspace_method == 'projection':
                    AS3 = sparse.vstack([J_AO, J_CO]).dot(
                        J_OH[:(num_block_rows + 1) * num_channels * order,:])
                    sigma_AC = AS3.dot(sigma_H).dot(AS3.T)

            if variance_algo == 'fast':
                Q4n = Q4[:num_channels * order,:]
#                 Q4n = sparse.hstack([sparse.identity(num_channels*order),
#                                            sparse.csr_matrix((num_channels*order,num_channels*(max_model_order-order)))]).dot(Q4)

            if variance_algo == 'fast' and lsq_method == 'pinv':
                # extraction of block rows from precomputed Q_i Matrices

                rows = np.hstack(
                    [np.arange(order) + i * max_model_order for i in range(order)])

#                 S4n = sparse.kron(sparse.hstack([sparse.identity(order, format='csr'),
#                                                              sparse.csr_matrix((order,max_model_order-order))]),
#                              sparse.hstack([sparse.identity(order, format='csr'),
#                                                   sparse.csr_matrix((order,max_model_order-order))]))
#
#                 Q1n = S4n.dot(Q1)
#                 Q2n = S4n.dot(Q2)
#                 Q3n = S4n.dot(Q3)

                Q1n = Q1[rows,:]
                Q2n = Q2[rows,:]
                Q3n = Q3[rows,:]

#                 print(np.all(Q1n==Q1n_))
#                 print(np.all(Q2n==Q2n_))
#                 print(np.all(Q3n==Q3n_))

                # Computation of (On_up On_up)^-1 , (P_nn + I_n2) Q1 and the
                # sum P Q2 +Q3
                On_up2 = np.dot(On_up.T, On_up)
                On_up2i = np.linalg.pinv(On_up2)

                P_nn = permutation(order, order)

                PQ1 = (P_nn + sparse.identity(order ** 2)).dot(Q1n)
                PQ23 = P_nn.dot(Q2n) + Q3n

            for i, lambda_i in enumerate(eigval):

                a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
                b_i = np.log(np.abs(lambda_i))
                freq_i = np.sqrt(a_i ** 2 + b_i ** 2) * sampling_rate / 2 / np.pi
                damping_i = 100 * np.abs(b_i) / np.sqrt(a_i ** 2 + b_i ** 2)

                if debug:
                    lambda_ci = np.log(complex(lambda_i)) * sampling_rate
                    freq_i = np.abs(lambda_ci) / 2 / np.pi
                    damping_i = -100 * np.real(lambda_ci) / np.abs(lambda_ci)

                mode_shape_i = np.dot(
                    output_matrix[:, 0:order], eigvec_r[:, i])
                mode_shape_i = np.array(mode_shape_i, dtype=complex)

                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                # mode_shape_i = self.integrate_quantities(mode_shape_i, accel_channels, velo_channels, complex(freq_i*2*np.pi))
                # if each channel was preconditioned to a common vibration level reverse this in the mode shapes
                # mode_shape_i*=self.prep_signals.channel_factors
                # scale mode shapes to unit modal displacement
                # mode_shape_i = self.rescale_mode_shape(mode_shape_i, doehler_style=True)
                k = np.argmax(np.abs(mode_shape_i))
                s_ik = mode_shape_i[k]
                t_ik = np.abs(s_ik)
                # alpha = np.arctan(sik.imag/sik.real)
                alpha_ik = np.angle(s_ik)
                e_k = np.zeros((num_channels, 1))  # , dtype=complex)
                e_k[k, 0] = 1
                mode_shape_i *= np.exp(-1j * alpha_ik)
                # alpha = np.arctan(sik.imag/sik.real)

                eigenvalues[order, i] = lambda_i
                modal_frequencies[order, i] = freq_i
                modal_damping[order, i] = damping_i
                mode_shapes[:, i, order] = mode_shape_i

                # uncertainty computation
                Phi_i = eigvec_r[:, i:i + 1]
                Chi_i = eigvec_l[:, i:i + 1]

                # Compute J_fili , J_xili in Lemma 5
                tlambda_i = (b_i + 1j * a_i) * sampling_rate
                J_fixiili = (sampling_rate / ((np.abs(lambda_i) ** 2) * np.abs(tlambda_i)) *
                             np.dot(np.dot(np.array([[1 / (2 * np.pi), 0],
                                                     [0, 100 / (np.abs(tlambda_i) ** 2)]]),
                                           np.array([[np.real(tlambda_i), np.imag(tlambda_i)],
                                                     [-(np.imag(tlambda_i) ** 2), np.real(tlambda_i) * np.imag(tlambda_i)]])),
                                    np.array([[np.real(lambda_i), np.imag(lambda_i)],
                                              [-np.imag(lambda_i), np.real(lambda_i)]]))
                             )
                if variance_algo == 'fast':
                    if lsq_method == 'pinv':
                        # Compute Q_i in (44)
                        Q_i = sparse.kron(
                            Phi_i.T, sparse.identity(order)).dot(
                            PQ23 - lambda_i * PQ1)
                        J_liHT = 1 / np.dot(Chi_i.T.conj(), Phi_i) * \
                            np.dot(Chi_i.conj().T, np.dot(On_up2i, Q_i))

                    if lsq_method == 'qr':
                        J_liA = 1 / np.dot(Chi_i.T.conj(), Phi_i) * \
                            np.kron(Phi_i.T, Chi_i.T.conj())
                        J_liHT = np.dot(J_liA, J_AHT)

                    # Compute U_fixi in (42)
                    U_fixi = np.dot(J_fixiili, np.vstack(
                        [np.real(J_liHT), np.imag(J_liHT)]))
                    if debug:
                        # avoid using the inverse of Oj_up2
                        J_liHTs = 1 / np.dot(Chi_i.T.conj(),
                                             Phi_i) * np.dot(Chi_i.conj().T,
                                                             np.linalg.solve(On_up2,
                                                                             Q_i))
                        print(np.allclose(J_liHT, J_liHTs))
                        J_liHT = J_liHTs

                    # Compute the covariance of fi and xi in (40)
                    # var_fixi=np.dot(U_fixi,U_fixi.T)
                    var_fixi = np.einsum('ij,ij->i', U_fixi, U_fixi)

                    # Compute J_phi,A J_A,O J_O,HT in (46)
                    if lsq_method == 'pinv':
                        J_PhiiHT = np.dot(
                            np.linalg.pinv(
                                lambda_i *
                                np.identity(order) -
                                state_matrix),
                            np.dot(
                                np.identity(order) -
                                np.dot(
                                    Phi_i,
                                    Chi_i.T.conj()) /
                                np.dot(
                                    Chi_i.T.conj(),
                                    Phi_i),
                                np.dot(
                                    On_up2i,
                                    Q_i)))

                    if lsq_method == 'qr':
                        J_PhiA = np.dot(
                            np.linalg.pinv(
                                lambda_i *
                                np.identity(order) -
                                state_matrix),
                            np.kron(
                                Phi_i.T,
                                np.identity(order) -
                                np.dot(
                                    Phi_i,
                                    Chi_i.T.conj()) /
                                np.dot(
                                    Chi_i.T.conj(),
                                    Phi_i)))
                        J_PhiiHT = np.dot(J_PhiA, J_AHT)

                    if debug:
                        # avoid using the inverse of Oj_up2
                        J_PhiiHTs = np.dot(
                            np.linalg.pinv(
                                lambda_i *
                                np.identity(order) -
                                state_matrix),
                            np.dot(
                                np.identity(order) -
                                np.dot(
                                    Phi_i,
                                    Chi_i.T.conj()) /
                                np.dot(
                                    Chi_i.T.conj(),
                                    Phi_i),
                                np.linalg.solve(
                                    On_up2,
                                    Q_i)))
                        print(np.allclose(J_PhiiHT, J_PhiiHTs))
                        J_PhiiHT = J_PhiiHTs

                    # Compute U_phi from (41) and (45) unit modal displacement scheme
                    # k = np.argmax(np.abs(mode_shape_i))
#                     J_mshiHT = (1/mode_shape_i[k]*
#                                 np.dot(np.identity(num_channels, dtype=complex)-np.hstack([np.zeros((num_channels,k),dtype=complex),
#                                                                                            np.reshape(mode_shape_i,(num_channels,1)),
#                                                                                            np.zeros((num_channels,num_channels-(k+1)),dtype=complex)]),
#                                        np.dot(output_matrix[:, 0:order],J_PhiiHT) + np.dot(np.kron(Phi_i.T,np.identity(num_channels)),
# Q4n)))

                    J_phiiHT = np.exp(-1j * alpha_ik) * np.dot(-1j * np.power(t_ik,
                                                                              -2) * np.dot(np.dot(output_matrix[:,:order],
                                                                                                  Phi_i),
                                                                                           np.hstack([-np.imag(s_ik) * e_k.T,
                                                                                                      np.real(s_ik) * e_k.T])) + np.hstack([np.identity(num_channels),
                                                                                                                                            1j * np.identity(num_channels)]),
                                                               np.vstack([np.dot(output_matrix[:,:order],
                                                                                 np.real(J_PhiiHT)) + np.dot(np.kron(np.real(Phi_i).T,
                                                                                                                     np.identity(num_channels)),
                                                                                                             Q4n),
                                                                          np.dot(output_matrix[:,:order],
                                                                                 np.imag(J_PhiiHT)) + np.dot(np.kron(np.imag(Phi_i).T,
                                                                                                                     np.identity(num_channels)),
                                                                                                             Q4n)]))

#                     (1/mode_shape_i[k]*
#                                 np.dot(np.identity(num_channels, dtype=complex)-np.hstack([np.zeros((num_channels,k),dtype=complex),
#                                                                                            np.reshape(mode_shape_i,(num_channels,1)),
#                                                                                            np.zeros((num_channels,num_channels-(k+1)),dtype=complex)]),
#                                        np.dot(output_matrix[:, 0:order],J_PhiiHT) + np.dot(np.kron(Phi_i.T,np.identity(num_channels)),
# Q4n)))

                    U_phii = np.vstack([np.real(J_phiiHT), np.imag(J_phiiHT)])

                    # Compute the covariance of phi in (40)
                    # var_phii=np.dot(U_phii,U_phii.T)
                    # print(U_phii.shape)
                    # print('1',var_phii)
                    var_phii = np.einsum('ij,ij->i', U_phii, U_phii)

                if variance_algo == 'slow':

                    J_liA = 1 / np.dot(Chi_i.T.conj(), Phi_i) * \
                        np.kron(Phi_i.T, Chi_i.T.conj())
                    J_fixiA = np.dot(J_fixiili, np.vstack(
                        [np.real(J_liA), np.imag(J_liA)]))
                    var_fixi = np.dot(np.hstack([J_fixiA, np.zeros((2, num_channels * order))]), sigma_AC.dot(
                        np.hstack([J_fixiA, np.zeros((2, num_channels * order))]).T))
                    var_fixi = np.diag(var_fixi)

                    J_PhiA = np.dot(
                        np.linalg.pinv(
                            lambda_i *
                            np.identity(order) -
                            state_matrix),
                        np.kron(
                            Phi_i.T,
                            (np.identity(order) -
                             np.dot(
                                Phi_i,
                                Chi_i.T.conj()) /
                                np.dot(
                                Chi_i.T.conj(),
                                Phi_i))))

                    J_phiiAC = np.exp(-1j * alpha_ik) * np.dot(-1j * np.power(t_ik,
                                                                              -2) * np.dot(np.dot(output_matrix[:,
                                                                                                                0:order],
                                                                                                  Phi_i),
                                                                                           np.hstack([-np.imag(s_ik) * e_k.T,
                                                                                                      np.real(s_ik) * e_k.T])) + np.hstack([np.identity(num_channels),
                                                                                                                                            1j * np.identity(num_channels)]),
                                                               np.vstack([np.hstack([np.dot(output_matrix[:,
                                                                                                          0:order],
                                                                                            np.real(J_PhiA)),
                                                                                     np.kron(np.real(Phi_i).T,
                                                                                             np.identity(num_channels))]),
                                                                          np.hstack([np.dot(output_matrix[:,
                                                                                                          0:order],
                                                                                            np.imag(J_PhiA)),
                                                                                     np.kron(np.imag(Phi_i).T,
                                                                                             np.identity(num_channels))])]))

                    var_phii = np.dot(np.vstack([np.real(J_phiiAC), np.imag(J_phiiAC)]), sigma_AC.dot(
                        np.vstack([np.real(J_phiiAC), np.imag(J_phiiAC)]).T))
                    var_phii = np.diag(var_phii)

                std_frequencies[order, i] = np.sqrt(var_fixi[0])
                std_damping[order, i] = np.sqrt(var_fixi[1])

                std_mode_shapes.real[:, i, order] = np.sqrt(
                    var_phii[:num_channels])
                std_mode_shapes.imag[:, i, order] = np.sqrt(
                    var_phii[num_channels:2 * num_channels])

                if debug:
                    print('Frequency: {}, Std_Frequency: {}'.format(
                        freq_i, std_frequencies[order, i]))
                    print('Damping: {}, Std_damping: {}'.format(
                        damping_i, std_damping[order, i]))
                    print('Mode_Shape: {}, Std_Mode_Shape: {}'.format(
                        mode_shape_i, std_mode_shapes[:, i, order]))
        self.eigenvalues = eigenvalues

        self.modal_frequencies = modal_frequencies
        self.std_frequencies = std_frequencies

        self.modal_damping = modal_damping
        self.std_damping = std_damping

        self.mode_shapes = mode_shapes
        self.std_mode_shapes = std_mode_shapes

        self.state[2] = True

#     def compute_modal_params(self, max_model_order=None, debug=False):
#
#         if max_model_order is not None:
#             assert max_model_order<=self.max_model_order
#             self.max_model_order=max_model_order
#
#         assert self.state[1]
#
#         print('Computing modal parameters...')
#         state_matrix = self.state_matrix
#         output_matrix = self.output_matrix
#         subspace_matrix = self.subspace_matrix
#         O = self.O
#
#         lsq_method = self.lsq_method
#         max_model_order = self.max_model_order
#         sampling_rate = self.prep_signals.sampling_rate
#         num_channels = self.prep_signals.num_analised_channels
#         num_ref_channels = self.prep_signals.num_ref_channels
#         num_block_columns = self.num_block_columns
#         num_block_rows = self.num_block_rows
#
#         accel_channels=self.prep_signals.accel_channels
#         velo_channels=self.prep_signals.velo_channels
#
#         modal_frequencies = np.zeros((max_model_order, max_model_order))
#         std_frequencies = np.zeros((max_model_order, max_model_order))
#         modal_damping = np.zeros((max_model_order, max_model_order))
#         std_damping = np.zeros((max_model_order, max_model_order))
#         mode_shapes = np.zeros((num_channels, max_model_order, max_model_order),dtype=complex)
#         std_mode_shapes = np.zeros((num_channels, max_model_order, max_model_order),dtype=complex)
#
#         S3 = self.S3
#         S1 = sparse.hstack([sparse.identity((num_block_rows)*num_channels),
#                                   sparse.csr_matrix(((num_block_rows)*num_channels,num_channels))])
#
#         S2 = sparse.hstack([ sparse.csr_matrix(((num_block_rows)*num_channels,num_channels)),
#                                   sparse.identity((num_block_rows)*num_channels)])
#
#         for order in range(1,max_model_order):
#             print('(c) Step up order: ',order)
#
#             On_up = O[:num_channels * (num_block_rows),:order]
#             On_down = O[num_channels:num_channels * (num_block_rows+1) ,:order]
#             if lsq_method == 'pinv':
#                 state_matrix = np.dot(np.linalg.pinv(On_up), On_down)
#
#                 P_p1rn = permutation((num_block_rows+1)*num_channels, order)
#                 J_AO=(sparse.kron(sparse.identity(order),S2.T.dot(np.linalg.pinv(On_up).T).T)-
#                       sparse.kron(state_matrix.T,S1.T.dot(np.linalg.pinv(On_up).T).T)+
#                       P_p1rn.T.dot(np.kron(S1.T.dot(On_down).T - S1.T.dot(np.dot(state_matrix.T,
#                                                                                      On_up.T).T
#                                                                               ).T,
#                                       np.linalg.inv(np.dot(On_up[:,:order].T,On_up[:,:order]))).T
#                       ).T)
#             if lsq_method == 'qr':
#                 R_n = self.R_nmax[:order,:order]
#                 R_ni = np.linalg.inv(R_n)
#                 S_n = self.S_nmax[:order,:order]
#                 state_matrix = np.dot(R_ni, S_n)
#
#                 T_n = np.vstack((np.identity(order), np.zeros((max_model_order-order,order))))
#                 J_AO = np.dot(np.kron(T_n.T,np.dot(R_ni, T_n.T)),self.J_Snmax) \
#                         -np.dot(np.kron(np.dot(state_matrix.T, T_n.T),np.dot(R_ni, T_n.T)), self.J_Rnmax)
#                 J_AO = J_AO[:order**2, :order*(num_block_rows+1)*num_channels]
#                 #J_AOHT = np.dot(J_AO, self.J_OHT)
#
#             eigval, eigvec_l, eigvec_r = scipy.linalg.eig(a=state_matrix,b=None,left=True,right=True)
#             eigval, eigvec_l, eigvec_r = self.remove_conjugates(eigval, eigvec_l, eigvec_r)
#
#             # K_i, B_i,1;
#
#
#             if debug:
#                 A=sparse.vstack([J_AO,J_CO]).dot(J_OH)
#                 sigma_ACT = A.dot(np.dot(self.hankel_cov_matrix,self.hankel_cov_matrix.T)).dot(A.T)# with sigma_H from T
#                 print('Sigma_AC (R,T)',np.allclose(sigma_AC, sigma_ACT))
#
#                 S4n = sparse.kron(sparse.hstack([sparse.identity(order),np.zeros((order,max_model_order-order))]),
#                              sparse.hstack([sparse.identity(order),np.zeros((order,max_model_order-order))]))
#
#                 Q1n = S4n.dot(self.Q1)
#                 Q2n = S4n.dot(self.Q2)
#                 Q3n = S4n.dot(self.Q3)
#                 Q4n = sparse.hstack([sparse.identity(num_channels*order),sparse.csr_matrix((num_channels*order,num_channels*(max_model_order-order)))]).dot(self.Q4)
#
#                 #Computation of (Oj_up Oj_up)^-1 , (P_nn + I_n2) Q1 and the sum P Q2 +Q3
#
#                 On_up2 = np.dot(On_up.T, On_up)
#                 On_up2i = np.linalg.inv(On_up2)
#
#                 P_nn = permutation(order,order)
#
#                 PQ1 = (P_nn + sparse.identity(order**2)).dot(Q1n)
#                 PQ23 = P_nn.dot(Q2n) + Q3n
#
#                 J_AHT= np.dot(np.kron(np.identity(order),np.linalg.inv(On_up2)),np.dot(-1*np.kron(self.state_matrix.T,np.identity(order)),PQ1)+PQ23)
#                 J_AHTQ=J_AO.dot(np.dot(self.J_OH[:order*(num_block_rows+1)*num_channels,:],self.hankel_cov_matrix))
#                 print('J_AOHT',np.allclose(J_AHT, J_AHTQ))
#
#                 J_CHT = np.dot(np.dot(self.J_OH[:order*(num_block_rows+1)*num_channels,:],self.hankel_cov_matrix))
#                 J_CHTQ=Q4n
#                 print('J_COHT',np.allclose(J_CHTQ, J_CHT))
#
#                 U_AC = np.vstack([J_AHT,J_CHT])
#                 sigma_ACQ=np.dot(U_AC,U_AC.T)
#
#                 print('Sigma_AC (R,Q)', np.allclose(sigma_AC, sigma_ACQ))
#
#
#             for i,lambda_i in enumerate(eigval):
#
#                 a_i = np.abs(np.arctan2(np.imag(lambda_i),np.real(lambda_i)))
#                 b_i = np.log(np.abs(lambda_i))
#                 freq_i = np.sqrt(a_i**2+b_i**2)*sampling_rate/2/np.pi
#                 damping_i = 100*np.abs(b_i)/np.sqrt(a_i**2+b_i**2)
#
#                 if debug:
#                     lambda_ci=np.log(complex(lambda_i))*sampling_rate
#                     freq_i=np.abs(lambda_ci)/2/np.pi
#                     damping_i=-100*np.real(lambda_ci)/np.abs(lambda_ci)
#
#                 mode_shape_i = np.dot(output_matrix[:, 0:order], eigvec_r[:,i])
#                 mode_shape_i = np.array(mode_shape_i, dtype=complex)
#
#                 # integrate acceleration and velocity channels to level out all channels in phase and amplitude
#                 #mode_shape_i = self.integrate_quantities(mode_shape_i, accel_channels, velo_channels, complex(freq_i*2*np.pi))
#                 # if each channel was preconditioned to a common vibration level reverse this in the mode shapes
#                 #mode_shape_i*=self.prep_signals.channel_factors
#                 # scale mode shapes to unit modal displacement
#                 #mode_shape_i = self.rescale_mode_shape(mode_shape_i)
#
#                 k = np.argmax(np.abs(mode_shape_i))
#                 s_ik = mode_shape_i[k]
#                 t_ik = np.abs(s_ik)
#                 #alpha = np.arctan(sik.imag/sik.real)
#                 alpha_ik = np.angle(s_ik)
#                 mode_shape_i *= np.exp(-1j*alpha_ik)
#
#                 modal_frequencies[order,i] = freq_i
#                 modal_damping[order,i] = damping_i
#                 mode_shapes[:,i,order] = mode_shape_i
#
#                 # Uncertainty Computation
#                 Phi_i = eigvec_r[:,i:i+1]
#                 Chi_i = eigvec_l[:,i:i+1]
#
#                 J_liA = 1/np.dot(Chi_i.T.conj(),Phi_i)*np.kron(Phi_i.T,Chi_i.T.conj())
#                 J_PhiA= np.dot(np.linalg.pinv(lambda_i*np.identity(order)-state_matrix),
#                                np.kron(Phi_i.T,(np.identity(order)-np.dot(Phi_i,Chi_i.T.conj())/np.dot(Chi_i.T.conj(),Phi_i))))
#
#                 #Compute J_fili , J_xili in Lemma 5
#                 tlambda_i = (b_i+1j*a_i)*sampling_rate
#
#                 J_fixiili=(sampling_rate/((np.abs(lambda_i)**2) * np.abs(tlambda_i))*
#                  np.dot(np.dot(np.array([[1/2/np.pi,    0                         ],
#                                          [0,            100/(np.abs(tlambda_i)**2)]]),
#                                np.array([[np.real(tlambda_i),       np.imag(tlambda_i)],
#                                          [-(np.imag(tlambda_i)**2),   np.real(tlambda_i)*np.imag(tlambda_i)]])),
#                         np.array([[np.real(lambda_i),   np.imag(lambda_i)],
#                                   [-np.imag(lambda_i),  np.real(lambda_i)]]))
#                  )
#
#                 J_fixiA = np.dot(J_fixiili,np.vstack([np.real(J_liA),np.imag(J_liA)]))
#                 var_fixi = np.dot(np.hstack([J_fixiA, np.zeros((2,num_channels*order))]),sigma_AC.dot(np.hstack([J_fixiA, np.zeros((2,num_channels*order))]).T))
#
#                 #k = np.argmax(np.abs(mode_shape_i))
# #                 J_phiiAC = (1/mode_shape_i[k]*
# #                             np.dot(np.identity(num_channels, dtype=complex)-np.hstack([np.zeros((num_channels,k),dtype=complex),
# #                                                                                        np.reshape(mode_shape_i,(num_channels,1)),
# #                                                                                        np.zeros((num_channels,num_channels-(k+1)),dtype=complex)]),
# #                                    np.hstack([np.dot(output_matrix[:, 0:order],J_PhiA),np.kron(Phi_i.T,
# #                                                                                                np.identity(num_channels))])))
#                 e_k = np.zeros((num_channels,1))#, dtype=complex)
#                 e_k[k,0]=1
#                 J_phiiAC = np.exp(-1j*alpha_ik)*\
#                             np.dot(-1j*np.power(t_ik,-2)*np.dot(np.dot(output_matrix[:, 0:order], Phi_i),np.hstack([-np.imag(s_ik)*e_k.T,np.real(s_ik)*e_k.T]))
#                                    +np.hstack([np.identity(num_channels), 1j*np.identity(num_channels)]),
#                                    np.vstack([np.hstack([np.dot(output_matrix[:, 0:order],np.real(J_PhiA)), np.kron(np.real(Phi_i).T,np.identity(num_channels))]),
#                                               np.hstack([np.dot(output_matrix[:, 0:order],np.imag(J_PhiA)), np.kron(np.imag(Phi_i).T,np.identity(num_channels))])]))
#
#                 var_phii= np.dot(np.vstack([np.real(J_phiiAC),np.imag(J_phiiAC)]),sigma_AC.dot(np.vstack([np.real(J_phiiAC),np.imag(J_phiiAC)]).T))
#
#                 std_frequencies[order,i]=np.sqrt(var_fixi[0,0])
#                 std_damping[order, i]=np.sqrt(var_fixi[1,1])
#
#                 std_mode_shapes.real[:,i,order]=np.sqrt(var_phii[range(num_channels),range(num_channels)])
#                 std_mode_shapes.imag[:,i,order]=np.sqrt(var_phii[range(num_channels,2*num_channels),range(num_channels,2*num_channels)])
#
#                 if debug:
#                     print('Frequency: {}, Std_Frequency: {}'.format(freq_i, std_frequencies[order,i]))
#                     print('Damping: {}, Std_damping: {}'.format(damping_i, std_damping[order, i]))
#                     print('Mode_Shape: {}, Std_Mode_Shape: {}'.format(mode_shape_i, std_mode_shapes[:,i,order]))
#
#         self.modal_frequencies = modal_frequencies
#         self.std_frequencies = std_frequencies
#
#         self.modal_damping = modal_damping
#         self.std_damping = std_damping
#
#         self.mode_shapes = mode_shapes
#         self.std_mode_shapes = std_mode_shapes
#
#         self.state[2]=True
#
#         #return sigma_AC, eigval, eigvec_l, eigvec_r
#
#
#     @staticmethod
#     def remove_conjugates_new (eigval, eigvec_l, eigvec_r):
#         '''
#         removes conjugates
#
#         eigvec_l.shape = [order+1, order+1]
#         eigval.shape = [order+1,1]
#         '''
#         #return vectors, eigval
#         num_val=len(eigval)
#         conj_indices=deque()
#
#         for i in range(num_val):
#             this_val=eigval[i]
#             this_conj_val = np.conj(this_val)
#             if this_val == this_conj_val: #remove real eigvals
#                 conj_indices.append(i)
#             #ind=np.argmax(eigval[i+1:]==this_conj_val)
#             #if ind: conj_indices.append(ind+i+1)
#             for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
#                 if eigval[j] == this_conj_val:
#
#                     #if not np.allclose(eigvec_l[j],eigvec_l[i].conj()):
#                     #    print('eigval is complex conjugate but eigvec_l is not')
#                     #    continue
#
#                     #if not np.allclose(eigvec_r[j],eigvec_r[i].conj()):
#                     #    print('eigval is complex conjugate but eigvec_r is not')
#                     #    continue
#
#                     conj_indices.append(j)
#                     break
#
#         #print('indices of complex conjugate: {}'.format(conj_indices))
#         conj_indices=list(set(range(num_val)).difference(conj_indices))
#         #print('indices to keep and return: {}'.format(conj_indices))
#
#
#         eigvec_l = eigvec_l[:,conj_indices]
#         eigvec_r = eigvec_r[:,conj_indices]
#         eigval = eigval[conj_indices]
#
#         return eigval,eigvec_l,eigvec_r
#
#     @staticmethod
#     def integrate_quantities(vector, accel_channels, velo_channels, omega):
#         # input quantities = [a, v, d]
#         # output quantities = [d, d, d]
#         # converts amplitude and phase
#         #                     phase + 180; magn / omega^2
#
#         vector[accel_channels] *= -1       / (omega ** 2)
#         #                    phase + 90; magn / omega
#         vector[velo_channels] *=  1j        / omega
#
#         return vector

    def save_state(self, fname):

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time

        if self.state[0]:  # subspace matrices

            out_dict['self.subspace_method'] = self.subspace_method
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.num_blocks'] = self.num_blocks

            if self.subspace_method == 'covariance':
                out_dict['self.corr_mats_mean'] = self.corr_mats_mean
                out_dict['self.corr_matrices'] = self.corr_matrices
            out_dict['self.subspace_matrix'] = self.subspace_matrix
            out_dict['self.subspace_matrices'] = self.subspace_matrices

        if self.state[1]:  # state models and sensitivities

            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
            out_dict['self.O'] = self.O
            out_dict['self.U'] = self.U
            out_dict['self.S'] = self.S
            out_dict['self.V_T'] = self.V_T

            out_dict['self.variance_algo'] = self.variance_algo
            if self.variance_algo == 'slow' and self.subspace_method == 'covariance':
                out_dict['self.sigma_R'] = self.sigma_R  # slow and covariance
                out_dict['self.S3'] = self.S3  # slow and covariance
                out_dict['self.J_OHS3'] = self.J_OHS3  # slow and covariance
            if self.variance_algo == 'slow' and self.subspace_method == 'projection':
                out_dict['self.sigma_H'] = self.sigma_H  # slow and projection
                out_dict['self.J_OH'] = self.J_OH  # slow and projection
            if self.variance_algo == 'fast' or self.subspace_method == 'projection':
                # fast or projection
                out_dict['self.hankel_cov_matrix'] = self.hankel_cov_matrix

            out_dict['self.lsq_method'] = self.lsq_method
            if self.lsq_method == 'qr':
                out_dict['self.Q_nmax'] = self.Q_nmax
                out_dict['self.R_nmax'] = self.R_nmax
                out_dict['self.S_nmax'] = self.S_nmax
                out_dict['self.J_Rnmax'] = self.J_Rnmax
                out_dict['self.J_Snmax'] = self.J_Snmax
            if self.variance_algo == 'fast' and self.lsq_method == 'pinv':
                out_dict['self.Q1'] = self.Q1
                out_dict['self.Q2'] = self.Q2
                out_dict['self.Q3'] = self.Q3
            if self.variance_algo == 'fast' and self.lsq_method == 'qr':
                out_dict['self.J_OHT'] = self.J_OHT  # fast
            if self.variance_algo == 'fast':
                out_dict['self.Q4'] = self.Q4

        if self.state[2]:  # modal params

            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.std_frequencies'] = self.std_frequencies
            out_dict['self.std_damping'] = self.std_damping
            out_dict['self.std_mode_shapes'] = self.std_mode_shapes

        np.savez_compressed(fname, **out_dict)

        logger.info('Modal results saved to {}'.format(fname))

    @classmethod
    def load_state(cls, fname, prep_signals):
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

#         for this_state, state_string in zip(state, ['Subspace Matrices Built',
#                                                     'State Matrices and Sensitivities Computed',
#                                                     'Modal Parameters Computed',
#                                                     ]):
#             if this_state: print(state_string)

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        # start_time = in_dict['self.start_time'].item()
        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        # prep_signals = in_dict['self.prep_signals'].item()
        ssi_object = cls(prep_signals)
        ssi_object.state = state
        if state[0]:  # covariances

            ssi_object.subspace_method = str(in_dict['self.subspace_method'])
            ssi_object.num_block_columns = int(
                in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
            ssi_object.num_blocks = int(in_dict['self.num_blocks'])

            if ssi_object.subspace_method == 'covariance':
                ssi_object.corr_mats_mean = in_dict['self.corr_mats_mean']
                ssi_object.corr_matrices = in_dict['self.corr_matrices']
            ssi_object.subspace_matrix = in_dict['self.subspace_matrix']
            ssi_object.subspace_matrices = in_dict['self.subspace_matrices']

            logger.debug('Subspace Matrices Built: {}, {} block_rows'.format(
                ssi_object.subspace_method, ssi_object.num_block_rows))
        if state[1]:  # state models

            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix = in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
            ssi_object.O = in_dict['self.O']
            ssi_object.U = in_dict['self.U']
            ssi_object.S = in_dict['self.S']
            ssi_object.V_T = in_dict['self.V_T']

            ssi_object.variance_algo = str(in_dict['self.variance_algo'])
            if ssi_object.variance_algo == 'slow' and ssi_object.subspace_method == 'covariance':
                # slow and covariance
                ssi_object.sigma_R = in_dict['self.sigma_R']
                ssi_object.S3 = in_dict['self.S3']  # slow and covariance
                # slow and covariance
                ssi_object.J_OHS3 = in_dict['self.J_OHS3']
            if ssi_object.variance_algo == 'slow' and ssi_object.subspace_method == 'projection':
                # slow and projection
                ssi_object.sigma_H = in_dict['self.sigma_H']
                ssi_object.J_OH = in_dict['self.J_OH']  # slow and projection
            if ssi_object.variance_algo == 'fast' or ssi_object.subspace_method == 'projection':
                # fast or projection
                ssi_object.hankel_cov_matrix = in_dict['self.hankel_cov_matrix']

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
                ssi_object.J_OHT = in_dict['self.J_OHT']  # fast
            if ssi_object.variance_algo == 'fast':
                ssi_object.Q4 = in_dict['self.Q4']

            logger.debug('State Matrices and Sensitivities Computed: {} up to order {}'.format(
                ssi_object.lsq_method, ssi_object.max_model_order))
        if state[2]:  # modal params
            ssi_object.eigenvalues = in_dict['self.eigenvalues']
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
            ssi_object.std_frequencies = in_dict['self.std_frequencies']
            ssi_object.std_damping = in_dict['self.std_damping']
            ssi_object.std_mode_shapes = in_dict['self.std_mode_shapes']

            logger.debug('Modal Parameters Computed')
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
#             return modeshape
# def update_svd():
#     '''
#     A is the full matrix
#     r_l is the lower bound where to start the svd
#     that means all columns and rows above r are set to zero and the svd
#     is incrementally updated
#     in each step, first a column and then a row is added/updated
#
#     the above procedure only works for low rank matrices where
#     r is approximately the sqrt of the first dimension
#
#     '''
#
#
#     A= np.random.random((5,4))
#     B=np.copy(A)
#
#     r=A.shape[1]-1
#     B[:,r:]=0
#
#     U,S,V_T = np.linalg.svd(B,0)
#     U_,S_,V_T_ = update_column(U[:,:r],S[:r],V_T[:r,:],A[:,r])
#
#     oU,oS,oV_T = np.linalg.svd(A,0)
#     print(np.allclose(np.abs(oU),np.abs(U_)))
#     print(np.allclose(np.abs(oS),np.abs(S_)))
#     print(np.allclose(np.abs(oV_T),np.abs(V_T_)))
#     print(np.allclose(A,oU.dot(np.diag(S_)).dot(oV_T)))
#     print('',A,'\n',U_.dot(np.diag(S_)).dot(V_T_))
#
#
# def update_column(U,S,V_T,y,r):
#     '''
#     updates column r in the matrix formed by U.dot(np.diag(s)).dot(V_T)
#     to the provided column y
#     '''
#     col_num = V_T.shape[1]
#     b=np.zeros((col_num,1))
#     b[-1]=1
#     a = np.expand_dims(y,1)
#     m = U.T.dot(a)
#     p_ = a - U.dot(m)
#     p = np.sqrt(a.T.dot(p_))
#     P = p/p_
#     mat = np.vstack([np.hstack([np.diag(S),m]),np.expand_dims(np.append(np.zeros_like(S),p),0)])
#
#     U_,S_,V_T_ = np.linalg.svd(mat,0)
#     U__ = np.hstack((U,P)).dot(U_)
#     V_T__ = np.hstack((V_T.T,b)).dot(V_T_.T).T
#     return U__,S_,V_T__
#


def main():
    # update_svd()
    # exit()
    permutation(2, 2)
    # test decompositions derived from the qr decomposition
    a = np.random.random((1024, 3072))
    a = a.T
    r, q = rq_decomp(a)
    print(np.allclose(a, r.dot(q)))
    q, l = ql_decomp(a)
    print(np.allclose(a, q.dot(l)))
    l, q = lq_decomp(a)
    print(np.allclose(a, l.dot(q)))


if __name__ == '__main__':
    # pass
    main()
