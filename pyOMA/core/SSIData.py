# -*- coding: utf-8 -*-
"""
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

Created on Thu Oct 16 14:41:56 2014

@author: volkmar
"""

import numpy as np
import scipy.linalg

import os

from .Helpers import lq_decomp, validate_array, simplePbar

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

'''
.. TODO::
     * define unit tests to check functionality after changes
     * add switch to keep synthesized time-histories
'''


class SSIDataMC(ModalBase):

    def __init__(self, *args, **kwargs):
        '''
        channel definition: channels start at 0
        '''
        super().__init__(*args, **kwargs)

        self.state = [False, False, False, False]

        self.num_block_rows = None
        self.num_blocks = 1

        self.P_i_ref = None
        self.P_i_1 = None
        self.Y_i_i = None
        self.S = None
        self.U = None
        self.V_T = None

        self.max_model_order = None

        self.modal_contributions = None

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        assert os.path.exists(conf_file)
        assert isinstance(prep_signals, PreProcessSignals)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_rows = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        ssi_object = cls(prep_signals)
        ssi_object.build_block_hankel(num_block_rows)
        ssi_object.compute_modal_params(max_model_order)

        return ssi_object

    def build_block_hankel(self, num_block_rows=None, reduced_projection=True):
        '''
        Builds a Block-Hankel Matrix of the measured time series with varying
        time lags and estimates the subspace matrix from its LQ decomposition. 

        ::

              <- num_time samples - num_block_rows->      _
            [     y_0      y_1      ...      y_(j-1)     ]^
            [     y_1      y_2      ...      y_j         ]num_block_rows (=i)*n_l
            [     ...      ...      ...      ...         ]v
            [     y_(2i-1)   y_(2i)  ...     y_(2i+j-2)  ]_
            
            
        The notation mostly follows Peeters 1999.
        
        Parameters
        -------
            num_block_rows: integer, required
                The number of block rows of the Subspace matrix
        '''

        if num_block_rows is None:
            num_block_rows = self.num_block_rows

        assert isinstance(num_block_rows, int)

        self.num_block_rows = num_block_rows

        signals = self.prep_signals.signals
        total_time_steps = self.prep_signals.total_time_steps

        ref_channels = sorted(self.prep_signals.ref_channels)
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels

        q = num_block_rows
        p = num_block_rows
        N = int(total_time_steps - 2 * p)

        logger.info(f'Building Block-Hankel matrix with {p} block-columns and {q} block rows')

        Y_minus = np.zeros((q * n_r, N))
        Y_plus = np.zeros(((p + 1) * n_l, N))

        for ii in range(q):
            Y_minus[(q - ii - 1) * n_r:(q - ii) * n_r,:] = signals[(ii):(ii + N), ref_channels].T
        for ii in range(p + 1):
            Y_plus[ii * n_l:(ii + 1) * n_l,:] = signals[(q + ii):(q + ii + N)].T

        Hankel_matrix = np.vstack((Y_minus, Y_plus))
        Hankel_matrix /= np.sqrt(N)

        logger.debug(Hankel_matrix.shape)

        # self.Hankel_matrix = Hankel_matrix

        l, q = lq_decomp(Hankel_matrix, mode='full')

        logger.info('Estimating subspace matrix...')

        a = n_r * p
        b = n_r
        c = n_l - n_r
        d = n_l * p

        P_i_ref = l[a:a + b + c + d,: a] @ q[:a,:]

        if reduced_projection:
            [U, S, V_T] = np.linalg.svd(l[a:a + b + c + d,: a], full_matrices=False)
            V_T = None
        else:
            [U, S, V_T] = np.linalg.svd(P_i_ref, full_matrices=False)

        P_i_1 = l[a + b + c:a + b + c + d,: a + b] @ q[: a + b,: ]
        # Y_i_i = l[a : a + b + c, : a + b + c] @ q[ : a + b + c, : ]
        Y_i_i = Hankel_matrix[a: a + b + c,: ]
        # print(np.sum(np.abs(Y_i_i-l[a : a + b + c, : a + b + c] @ q[ : a + b + c, : ])))

        self.L_red = l[a:,:a + b]
        self.Q_red = q[:a + b,:]
        self.P_i_1 = P_i_1
        self.P_i_ref = P_i_ref
        self.Y_i_i = Y_i_i

        self.S = S
        self.U = U
        self.V_T = V_T

        # self.max_model_order = self.S.shape[0]

        self.state[0] = True

    def compute_modal_params(self, max_model_order=None,
                             j=None, validation_blocks=None,
                             synth_sig=True):
        '''
        Perform a multi-order computation of modal parameters. Successively
        calls
        
         * estimate_state(order,)
         * modal_analysis(A,C)
         * synthesize_signals(A, C, Q, R, S, j) 
        
        at ascending model orders, up to max_model_order. 
        See the explanations in the the respective methods, for a detailed 
        explanation of parameters.
        
        Parameters
        ----------
            max_model_order: integer, optional
                Maximum model order, where to interrupt the algorithm. If not given,
                it is determined from the previously computed subspace matrix.
        '''
        assert self.state[0]

        if max_model_order is not None:
            assert isinstance(max_model_order, int)
        else:
            max_model_order = self.self.S.shape[0]

        assert max_model_order <= self.S.shape[0]

        # num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_signals.num_analised_channels
        # num_ref_channels = self.prep_signals.num_ref_channels
        # sampling_rate = self.prep_signals.sampling_rate

        if j is None and validation_blocks is None:
            j = self.prep_signals.total_time_steps
        elif validation_blocks is None:
            assert j <= self.prep_signals.signals.shape[0]

        logger.info('Computing modal parameters...')
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=np.complex128)
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros(
            (num_analised_channels,
             max_model_order,
             max_model_order),
            dtype=complex)

        if synth_sig:
            modal_contributions = np.zeros((max_model_order, max_model_order))
        else:
            modal_contributions = None

        pbar = simplePbar(max_model_order - 1)
        for order in range(1, max_model_order):
            next(pbar)

            A, C, Q, R, S = self.estimate_state(order)

            f, d, phi, lamda, = self.modal_analysis(A, C)
            modal_frequencies[order,:order] = f
            modal_damping[order,:order] = d
            mode_shapes[:phi.shape[0],:order, order] = phi
            eigenvalues[order,:order] = lamda

            if synth_sig:
                _, delta = self.synthesize_signals(A, C, Q, R, S, j=j, validation_blocks=validation_blocks)
                modal_contributions[order,:order] = delta

        self.max_model_order = max_model_order

        self.modal_contributions = modal_contributions
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues

        self.state[2] = True

    def estimate_state(self, order,):
        '''
        Estimate the state matrices A, C and noise covariances Q, R and S from
        the subspace / projection matrix. Several methods exist, e.g. 
        
         * Peeters 1999 Reference Based Stochastic Subspace Identification for OMA
         * DeCock 2007 Subspace Identification Methods
         * the algorithm used in BRSSICovRef.
        
        Here, the first algorithm, a residual-based computation of Q, R  and S, 
        is implemented.
        
        
        Parameters
        ----------
            order: integer, required
                The model order, at which to truncate the singular values of the
                projection Matrix P_i_ref
                
        Returns
        -------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
            Q: numpy.ndarray
                state noise covariance matrix: Symmetric array of shape (order, order)
            
            R: numpy.ndarray
                signal noise covariance matrix: Array of shape (num_analised_channels, num_analised_channels)
            
            S: numpy.ndarray
                system noise - signal noise covariance matrix: Array of shape (order, num_analised_channels)
        '''
        num_block_rows = self.num_block_rows
        num_analised_channels = self.prep_signals.num_analised_channels
        # num_ref_channels = self.prep_signals.num_ref_channels

        U = self.U[:,:order]
        S = self.S[:order]

        P_i_1 = self.P_i_1  # n_l * p x n_r * (p + 1) x N
        Y_i_i = self.Y_i_i  # n_l x n_r *p + n_l x N

        # compute state-space model
        S_2 = np.power(S, 0.5)
        O = U * S_2[np.newaxis,:]

        O_i_1 = O[:num_analised_channels * num_block_rows,:order]
        O_i = O[:,:order]

        if self.V_T is not None:
            V_T = self.V_T[:order,:]
            X_i = S_2[:, np.newaxis] * V_T
        else:
            P_i_ref = self.P_i_ref  # (p + 1) * n_l x n_r * p x N
            X_i = np.linalg.pinv(O_i) @ P_i_ref
        X_i_1 = np.linalg.pinv(O_i_1) @ P_i_1

        X_i_1_Y_i = np.vstack((X_i_1, Y_i_i))

        AC = X_i_1_Y_i @ np.linalg.pinv(X_i)
        A = AC[:order,:]
        C = AC[order:,:]

        roh_w_v = X_i_1_Y_i - AC @ X_i

        QSR = roh_w_v @ roh_w_v.T

        Q = QSR[:order,:order]
        S = QSR[:order, order:order + num_analised_channels]
        R = QSR[order:order + num_analised_channels,
                order:order + num_analised_channels]

        return A, C, Q, R, S

    def modal_analysis(self, A, C, rescale_fun=None):
        '''
        Computes the modal parameters from a given state space model as described 
        by Peeters 1999 and Döhler 2012. Mode shapes are scaled to unit modal 
        displacements. Complex conjugate and real modes are removed prior to 
        further processing. Typically, order // 2 modes are in the returned arrays.
                
        Parameters
        ----------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
         
        Returns
        -------
            modal_frequencies: numpy.ndarray, shape (order,)
                Array holding the modal frequencies for each mode
            modal_damping: numpy.ndarray, shape (order,)
                Array holding the modal damping ratios (0,100) for each mode
            mode_shapes: numpy.ndarray, shape (n_l, order,)
                Complex array holding the mode shapes 
            eigenvalues: numpy.ndarray, shape (order,)
                Complex array holding the eigenvalues for each mode
        '''
        # collect variables
        accel_channels = self.prep_signals.accel_channels
        velo_channels = self.prep_signals.velo_channels
        sampling_rate = self.prep_signals.sampling_rate

        n_l = self.num_analised_channels

        order = A.shape[0]
        assert order == A.shape[1]

        # allocate output arrays
        modal_frequencies = np.full((order), np.nan)
        modal_damping = np.full((order), np.nan)
        mode_shapes = np.full((n_l, order), np.nan, dtype=complex)
        eigenvalues = np.full((order), np.nan, dtype=complex)

        # compute modal model
        eigvals, eigvecs_r = np.linalg.eig(A)
        Phi = C.dot(eigvecs_r)

        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        for i, ind in enumerate(conj_indices):

            lambda_i = eigvals[ind]
            mode_shape_i = Phi[:, ind]

            a_i = np.abs(np.arctan2(np.imag(lambda_i), np.real(lambda_i)))
            b_i = np.log(np.abs(lambda_i))
            freq_i = np.sqrt(a_i ** 2 + b_i ** 2) * sampling_rate / 2 / np.pi
            damping_i = 100 * np.abs(b_i) / np.sqrt(a_i ** 2 + b_i ** 2)

            if rescale_fun is not None:
                mode_shape_i = rescale_fun(mode_shape_i)

            # scale modeshapes to modal displacements
            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)

            # rotate mode shape in complex plane
            mode_shape_i = self.rescale_mode_shape(mode_shape_i)

            modal_frequencies[i] = freq_i
            modal_damping[i] = damping_i
            mode_shapes[:mode_shape_i.shape[0], i] = mode_shape_i
            eigenvalues[i] = lambda_i

        argsort = np.argsort(modal_frequencies)
        return modal_frequencies[argsort], modal_damping[argsort], mode_shapes[:, argsort], eigenvalues[argsort],

    def synthesize_signals(self, A, C, Q, R, S, j=None, **kwargs):
        '''
        Computes the modal response signals and the contribution of each mode.
        The algorithm follows Peeters 1999 and the Lyapunov equation is solved
        as a discrete-time algebraic Riccati equation (DARE). For long signals,
        the computation may become time-consuming, thus only time steps up to j
        may be used to synthesize the signal.
        
        
        Parameters
        ----------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
            Q: numpy.ndarray
                state noise covariance matrix: Symmetric array of shape (order, order)
            
            R: numpy.ndarray
                signal noise covariance matrix: Array of shape (num_analised_channels, num_analised_channels)
            
            S: numpy.ndarray
                system noise - signal noise covariance matrix: Array of shape (order, num_analised_channels)
                
            j: integer, optional
                length of signal to synthesize (number of timesteps)
         
        Returns
        -------
            sig_synth: numpy.ndarray, shape (num_analised_channels, j, order // 2)
                Array holding the modally decomposed input signals for
                each channel n_l and all modes
                
            modal_contributions: numpy.ndarray, shape (order, )
                Array holding the contributions of each mode to the input
                signals.
        '''

        order = A.shape[0]
        assert order == A.shape[1]

        if j is None:
            j = self.prep_signals.total_time_steps

        signals = self.prep_signals.signals[:j,:]
        n_l = self.prep_signals.num_analised_channels

        modal_contributions = np.zeros((order))
        sig_synth = np.zeros((n_l, j, order // 2))

        try:
            P = scipy.linalg.solve_discrete_are(
                a=A.T, b=C.T, q=Q, r=R, s=S, balanced=True)
        except:
            logger.warning('Correlations of residuals are not symmetric. Skiping Modal Contributions')
            return sig_synth, modal_contributions

        APCS = A @ P @ C.T + S
        CPCR = C @ P @ C.T + R
        K = np.linalg.solve(CPCR.T, APCS.T,).T

        eigvals, eigvecs_r = np.linalg.eig(A)
        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)

        A_0 = np.diag(eigvals)
        C_0 = C @ eigvecs_r
        K_0 = np.linalg.solve(eigvecs_r, K)

        states = np.zeros((order, j), dtype=complex)

        AKC = A_0 - K_0 @ C_0
        K_0m = K_0 @ signals.T

        for k in range(j - 1):
            states[:, k + 1] = K_0m[:, k] + AKC @ states[:, k]

        Y = signals.T
        norm = 1 / np.einsum('ji,ji->j', Y, Y)

        for i, ind in enumerate(conj_indices):

            lambda_i = eigvals[ind]

            ident = eigvals == lambda_i.conj()
            ident[ind] = 1

            C_0I = C_0[:, ident]

            this_sig_synth = C_0I @ states[ident,:]
            if not np.all(np.isclose(this_sig_synth.imag, 0)):
                logger.warning(f'Synthetized signals are complex at mode index {order}:{ind}.')

            sig_synth[:,:, i] = this_sig_synth.real

            mYT = np.einsum('ji,ji->j', sig_synth[:,:, i], Y)

            modal_contributions[i] = np.mean(norm * mYT)

        self._sig_synth = sig_synth
        self._modal_conributions = modal_contributions

        return sig_synth, modal_contributions

    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time

        if self.state[0]:  # subspace matrix
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.num_blocks'] = self.num_blocks
            # out_dict['self.P_i_1'] = self.P_i_1
            # out_dict['self.P_i_ref'] = self.P_i_ref

            out_dict['self.L_red'] = self.L_red
            out_dict['self.Q_red'] = self.Q_red
            out_dict['self.Y_i_i'] = self.Y_i_i
            out_dict['self.S'] = self.S
            out_dict['self.U'] = self.U
            out_dict['self.V_T'] = self.V_T
        if self.state[2]:  # modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.modal_contributions'] = self.modal_contributions
            out_dict['self.max_model_order'] = self.max_model_order

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())

        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        # prep_signals = in_dict['self.prep_signals'].item()
        ssi_object = cls(prep_signals)

        if state[0]:  # subspace matrix
            ssi_object.num_block_rows = validate_array(in_dict['self.num_block_rows'])
            ssi_object.num_blocks = validate_array(in_dict['self.num_blocks'])

            ssi_object.L_red = validate_array(in_dict['self.L_red'])
            ssi_object.Q_red = validate_array(in_dict['self.Q_red'])

            # rebuild projections from reduced L and Q matrices
            # saves ~ 95 % storage (Xe2 MB) compared to storing projections directly
            a = prep_signals.num_ref_channels * ssi_object.num_block_rows
            b = prep_signals.num_ref_channels
            c = prep_signals.num_analised_channels - prep_signals.num_ref_channels
            d = prep_signals.num_analised_channels * ssi_object.num_block_rows
            ssi_object.P_i_ref = ssi_object.L_red[:b + c + d,: a] @ ssi_object.Q_red[:a,:]
            ssi_object.P_i_1 = ssi_object.L_red[b + c:b + c + d,: ] @ ssi_object.Q_red[:,: ]

            ssi_object.Y_i_i = validate_array(in_dict['self.Y_i_i'])
            ssi_object.S = validate_array(in_dict['self.S'])
            ssi_object.U = validate_array(in_dict['self.U'])
            ssi_object.V_T = validate_array(in_dict['self.V_T'])
            ssi_object.S = validate_array(in_dict['self.S'])
            ssi_object.U = validate_array(in_dict['self.U'])
            ssi_object.V_T = validate_array(in_dict['self.V_T'])
        if state[2]:  # modal params
            ssi_object.modal_frequencies = validate_array(in_dict['self.modal_frequencies'])
            ssi_object.modal_damping = validate_array(in_dict['self.modal_damping'])
            ssi_object.eigenvalues = validate_array(in_dict['self.eigenvalues'])
            ssi_object.mode_shapes = validate_array(in_dict['self.mode_shapes'])
            ssi_object.modal_contributions = validate_array(in_dict['self.modal_contributions'])
            ssi_object.max_model_order = validate_array(in_dict['self.max_model_order'])

        ssi_object.state = state
        return ssi_object


class SSIData(SSIDataMC):

    def compute_modal_params(self, max_model_order):

        '''
        Perform a multi-order computation of modal parameters. Successively
        calls
        
         * estimate_state(order,)
         * modal_analysis(A,C)
        
        at ascending model orders, up to max_model_order. 
        See the explanations in the the respective methods, for a detailed 
        explanation of parameters.
        
        Parameters
        ----------
            max_model_order: integer, optional
                Maximum model order, where to interrupt the algorithm. If not given,
                it is determined from the previously computed subspace matrix.
        '''
        super().compute_modal_params(max_model_order, synth_sig=False)

    def estimate_state(self, order, max_modes=None, algo='svd'):
        '''
        
        Compute the state matrix A and output matrix C  from the singular values 
        and vectors of the projection matrix, truncated at the requested order. Estimation of the
        state matrix can be performed by QR decomposition or Singular Value decomposition
        of the shifted observability matrix. If max_modes is specified, the singular
        value decomposition is truncated additionally, also known as Crystal Clear SSI.
        
        Parameters
        ----------
            order: integer, required
                Model order, at which the state matrices should be estimated
            
            max_modes: integer, optional
                Maximum number of modes, that are known to be present in the signal,
                to suppress noise modes
            
            algo: str, optional
                Algorithm to use for estimation of A. Either 'svd' or 'qr'.
                
        Returns
        -------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
        '''
        if order > self.S.shape[0]:
            raise RuntimeError(f'Order cannot be higher than {self.S.shape[0]}. Consider using more block_rows/block_columns.')

        assert algo in ['svd', 'qr']

        n_l = self.num_analised_channels

        num_block_rows = self.num_block_rows

        U = self.U[:,:order]
        S = self.S[:order]

        # compute state-space model
        S_2 = np.power(S, 0.5)
        O = U * S_2[np.newaxis,:]

        On_up = O[:n_l * num_block_rows,:order]
        On_down = O[n_l:n_l * (num_block_rows + 1),:order]

        if algo == 'svd':
            if max_modes is not None:
                [u, s, v_t] = np.linalg.svd(On_up, 0)
                s = 1. / s[:max_modes]
                # On_up_i = np.dot(np.transpose(v_t[:max_modes, :]), np.multiply(
                #     s[:, np.newaxis], np.transpose(u[:, :max_modes])))

                On_up_i = v_t[:max_modes,:].T @ (s[:, np.newaxis] * u[:,:max_modes].T)
            else:
                On_up_i = np.linalg.pinv(On_up)  # , rcond=1e-12)
            A = On_up_i @ On_down

        elif algo == 'qr':
            Q, R = np.linalg.qr(On_up)
            S = Q.T.dot(On_down)
            A = np.linalg.solve(R, S)

        C = O[:n_l,:order]  # output matrix

        return A, C, None, None, None


class SSIDataCV(SSIDataMC):

    def build_block_hankel(self, num_block_rows=None, num_blocks=1, training_blocks=None, reduced_projection=True):
        '''
        Builds serveral Block-Hankel Matrices of the measured time series with varying
        time lags and estimates the subspace matrices from their LQ decompositions.
        Uniqueness of the subspace estimates is ensured by an intermediate LQ
        decomposition, where the diagonals of the L matrices are constrained to 
        positive values. A subspace matrix estimate is computed by the mean over 
        the training blocks leaving any remainig blocks for validation.
        
        Note: Blocks are not completely i.i.d. as we borrow p+q timesteps from the
        previous block for the projection of a full block (assembly of Hankel matrix)
        
        .. TODO::
          * investigate correct scaling of the subspace matrices 
            [sqrt(N_b), sqrt(N_b * num_blocks), sqrt(N_b*n_training_blocks)] ?
          * use sparse SVD (scipy.sparse.svds) to save memory and cpu time
        
        Parameters
        -------
            num_block_rows: integer, required
                The number of block rows of the Subspace matrix
                
            num_blocks: integer, optional
                The number of blocks, used for cross-validation
                
            training_blocks: list, optional
                The selected blocks to use for system identification (=training)
        '''

        if num_block_rows is None:
            num_block_rows = self.num_block_rows

        assert isinstance(num_block_rows, int)
        assert isinstance(num_blocks, int)

        if training_blocks is None:
            training_blocks = np.arange(num_blocks)
        elif isinstance(training_blocks, (list, tuple)):
            training_blocks = np.array(training_blocks)
        elif not isinstance(training_blocks, np.ndarray):
            raise RuntimeError(f"Argument 'training_blocks' must be an iterable but is type {type(training_blocks)}")

        assert training_blocks.max() < num_blocks
        n_training_blocks = training_blocks.shape[0]

        self.num_block_rows = num_block_rows
        self.num_blocks = num_blocks

        signals = self.prep_signals.signals
        total_time_steps = self.prep_signals.total_time_steps

        ref_channels = sorted(self.prep_signals.ref_channels)
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels

        q = num_block_rows
        p = num_block_rows

        logger.info(f'Building Block-Hankel matrix from  {n_training_blocks} out of {num_blocks} signal blocks with {p} block-columns and {q} block rows.')

        N_b = int(np.floor((total_time_steps - q - p) / num_blocks))
        if N_b < n_r * q:
            raise RuntimeError(f'Block-length ({N_b}) must not be smaller than the number of reference channels * number of block rows (={n_r * q}).')
        # might omit some timesteps in favor of equally sized blocks
        N = N_b * num_blocks
        # shorten signals by omitted samples to have them available for Kalman-Filter startup later
        N_offset = total_time_steps - q - p - N

        signals = signals[N_offset:,:]

        K = min((q * n_r) + (p + 1) * n_l, N_b)

        Y_minus = np.zeros((q * n_r, N))
        Y_plus = np.zeros(((p + 1) * n_l, N))

        for ii in range(q):
            Y_minus[(q - ii - 1) * n_r:(q - ii) * n_r,:] = signals[ii:ii + N, ref_channels].T
        for ii in range(p + 1):
            Y_plus[ii * n_l:(ii + 1) * n_l,:] = signals[q + ii:q + ii + N,:].T

        Hankel_matrix = np.vstack((Y_minus, Y_plus))
        Hankel_matrix /= np.sqrt(N)

        logger.debug(Hankel_matrix.shape)

        hankel_matrices = np.hsplit(Hankel_matrix, np.arange(N_b, N_b * num_blocks, N_b))

        np.hstack([hankel_matrices[i_block] for i_block in training_blocks])

        # R_matrices = []
        R_matrices = np.empty((n_r * p + n_l * (p + 1), K * n_training_blocks))
        Q_matrices = []

        # R_unique_matrices = []
        # Q_unique_matrices = []
        Q_unique_matrices = np.empty((q * n_r + (p + 1) * n_l, N))

        pbar = simplePbar(n_training_blocks * 3)
        for i in range(n_training_blocks):
            i_block = training_blocks[i]
            next(pbar)
            L, Q = lq_decomp(hankel_matrices[i_block], mode='reduced', unique=True)

            # R_matrices.append(L)
            R_matrices[:, i * K:(i + 1) * K] = L
            Q_matrices.append(Q)

        logger.debug(f'R shapes: actual: {np.hstack(R_matrices).shape} expected: {(n_r * p + n_l * (p + 1), K* n_training_blocks)}')

        # R_full_breve, Q_full_breve = lq_decomp(np.hstack(R_matrices), mode='reduced', unique=True)
        R_full_breve, Q_full_breve = lq_decomp(R_matrices, mode='reduced', unique=True)
        [next(pbar) for _ in range(n_training_blocks)]
        del R_matrices

        logger.debug(f'Q_breve shapes: actual: {Q_full_breve.shape} expected: ,{(q * n_r + (p + 1) * n_l, K * n_training_blocks)}')
        Q_breve_matrices = np.hsplit(Q_full_breve, np.arange(K, n_training_blocks * K, K))  # list of views into Q_full_breve
        logger.debug(f'Q_breve_j shapes: actual: {Q_breve_matrices[0].shape}, expected: {(q * n_r + (p + 1) * n_l, K)}')

        del Q_full_breve

        for i in range(n_training_blocks):
            next(pbar)
            Q_breve_matrix = Q_breve_matrices[i]
            # R_matrix = R_matrices[i]
            # R_matrix = R_matrices[:, i * K:(i + 1) * K]
            Q_matrix = Q_matrices[i]

            # R_unique_matrices.append(R_matrix @ Q_breve_matrix.T)
            # Q_unique_matrices.append(Q_breve_matrix @ Q_matrix) # (q * n_r + (p + 1) * n_l, N_b)
            Q_unique_matrices[:, i * N_b: (i + 1) * N_b ] = Q_breve_matrix @ Q_matrix
        del Q_breve_matrix
        del Q_breve_matrices
        del Q_matrix
        del Q_matrices

        logger.info('Estimating subspace matrix...')

        # L, Q = R_full_breve, np.concatenate(Q_unique_matrices, axis=1)
        L, Q = R_full_breve, Q_unique_matrices

        a = n_r * q
        b = n_r
        c = n_l - n_r
        d = n_l * p

        P_i_ref = L[a:a + b + c + d,: a] @ Q[:a,:]  # (p + 1) * n_l x n_r * q x N
        if reduced_projection:
            [U, S, V_T] = np.linalg.svd(L[a:a + b + c + d,: a], full_matrices=False)
            V_T = None
        else:
            [U, S, V_T] = np.linalg.svd(P_i_ref, full_matrices=False)

        P_i_1 = L[a + b + c:a + b + c + d,: a + b] @ Q[: a + b,: ]  # n_l * p x n_r * (q + 1) x N
        Y_i_i = L[a: a + b + c,: a + b + c] @ Q[: a + b + c,: ]  # n_l x n_r *q + n_l x N

        self.L_red = L[a:,:a + b]  # (p + 1) n_l x n_r * (q + 1)
        self.Q_red = Q[:a + b,:]  # n_r * (q + 1) x N
        self.P_i_1 = P_i_1
        self.P_i_ref = P_i_ref
        self.Y_i_i = Y_i_i

        self.S = S
        self.U = U
        self.V_T = V_T

        # self.max_model_order = self.S.shape[0]

        self.state[0] = True

    def synthesize_signals(self, A, C, Q, R, S, validation_blocks=None, N_offset=None, **kwargs):
        '''
        Computes the modal response signals and the contribution of each mode.
        The algorithm follows Peeters 1999 and the Lyapunov equation is solved
        as a discrete-time algebraic Riccati equation (DARE). For long signals,
        the computation may become time-consuming, thus only time steps up to j
        may be used to synthesize the signal.
        
        
        Parameters
        ----------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
            Q: numpy.ndarray
                state noise covariance matrix: Symmetric array of shape (order, order)
            
            R: numpy.ndarray
                signal noise covariance matrix: Array of shape (num_analised_channels, num_analised_channels)
            
            S: numpy.ndarray
                system noise - signal noise covariance matrix: Array of shape (order, num_analised_channels)
                
            validation_blocks: list, optional
                The selected blocks to be synthethized and used for system validation.
                
            N_offset: integer, optional
                The number of samples to be used from any previous block for 
                Kalman-Filter startup.
         
        Returns
        -------
            sig_synth: numpy.ndarray, shape (num_analised_channels, j, order // 2)
                Array holding the modally decomposed input signals for
                each channel n_l and all modes
                
            modal_contributions: numpy.ndarray, shape (order, )
                Array holding the contributions of each mode to the input
                signals.
        '''

        order = A.shape[0]
        assert order == A.shape[1]

        num_blocks = self.num_blocks

        if validation_blocks is None:
            validation_blocks = np.arange(num_blocks)
        elif isinstance(validation_blocks, (list, tuple)):
            validation_blocks = np.array(validation_blocks)
        elif not isinstance(validation_blocks, np.ndarray):
            raise RuntimeError(f"Argument 'validation_blocks' must be an iterable but is type {type(validation_blocks)}")

        assert validation_blocks.max() < num_blocks
        n_validation_blocks = validation_blocks.shape[0]

        n_l = self.prep_signals.num_analised_channels

        signals = self.prep_signals.signals

        total_time_steps = self.prep_signals.total_time_steps
        q = self.num_block_rows
        p = self.num_block_rows
        N_b = int(np.floor((total_time_steps - q - p) / num_blocks))
        # might omit some timesteps in favor of equally sized blocks
        N = N_b * num_blocks

        # blocks start at N_0_offset + p + q
        # (in training we virtually borrow p + q timesteps from the previous block)
        N_0_offset = total_time_steps - N
        if N_offset is None:
            N_offset = N_b // 5
        if 0 in validation_blocks and N_0_offset < N_offset:
            logger.info(f"Block '0' is in the validation dataset, but only has {N_0_offset} startup-samples (recommended/chosen: {N_offset}) from any previous block for the Kalman Filter. Expect a degraded performance.")

        modal_contributions = np.zeros((order))

        P = scipy.linalg.solve_discrete_are(
            a=A.T, b=C.T, q=Q, r=R, s=S, balanced=True)

        APCS = A @ P @ C.T + S
        CPCR = C @ P @ C.T + R
        K = np.linalg.solve(CPCR.T, APCS.T,).T

        eigvals, eigvecs_r = np.linalg.eig(A)
        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)

        A_0 = np.diag(eigvals)
        C_0 = C @ eigvecs_r
        K_0 = np.linalg.solve(eigvecs_r, K)

        AKC = A_0 - K_0 @ C_0

        block_starts = validation_blocks * N_b + N_0_offset

        all_sig_synth = []

        start_states = [None for _ in range(num_blocks + 1)]

        for i in np.argsort(validation_blocks):
            i_block = validation_blocks[i]

            sig_synth = np.zeros((n_l, N_b, order // 2))

            start_state = start_states[i]

            if i_block == 0:
                _N_offset = N_0_offset
            elif start_state is not None:
                _N_offset = 1
            else:
                _N_offset = N_offset

            states = np.zeros((order, N_b + _N_offset), dtype=complex)

            block_start = block_starts[i] - _N_offset
            block_end = block_starts[i] + N_b

            signals_block = signals[block_start:block_end,:]

            K_0m = K_0 @ signals_block.T

            if start_state is not None:
                states[:, 0] = start_state

            for k in range(N_b + _N_offset - 1):
                states[:, k + 1] = K_0m[:, k] + AKC @ states[:, k]

            start_states[i + 1] = states[:, -1]

            # remove start-up samples
            states = states[:, _N_offset:]
            Y = signals_block[_N_offset:,:].T

            norm = 1 / np.einsum('ji,ji->j', Y, Y)

            for i, ind in enumerate(conj_indices):

                lambda_i = eigvals[ind]

                ident = eigvals == lambda_i.conj()
                ident[ind] = 1

                C_0I = C_0[:, ident]

                this_sig_synth = C_0I @ states[ident,:]
                if not np.all(np.isclose(this_sig_synth.imag, 0)):
                    logger.warning(f'Synthetized signals are complex at mode index {order}:{ind}.')

                sig_synth[:,:, i] = this_sig_synth.real

                mYT = np.einsum('ji,ji->j', sig_synth[:,:, i], Y)

                modal_contributions[i] += np.mean(norm * mYT)

            all_sig_synth.append(sig_synth)

        modal_contributions /= n_validation_blocks

        self._sig_synth = all_sig_synth
        self._modal_contributions = modal_contributions

        return all_sig_synth, modal_contributions


def plot_sig_synth(modal_data, modelist=None, channel_inds=None, ref_channel_inds=None, axes=None, i_block=None):

    import matplotlib.pyplot as plt
    prep_signals = modal_data.prep_signals

    sig_synth = modal_data._sig_synth
    if isinstance(sig_synth, list):  # multi-block for cross-validation SSIDataCV
        sig_synth = sig_synth[i_block]
        num_blocks = modal_data.num_blocks
        total_time_steps = prep_signals.total_time_steps
        q = modal_data.num_block_rows
        p = modal_data.num_block_rows
        N_b = int(np.floor((total_time_steps - q - p) / num_blocks))
        N = N_b * num_blocks
        N_0_offset = total_time_steps - q - p - N
        # N_offset = N_b // 15
        block_starts = i_block * N_b + N_0_offset + p + q
        t = prep_signals.t[block_starts:block_starts + N_b]

        signals = prep_signals.signals.T[:, block_starts:block_starts + N_b]
    else:
        t = prep_signals.t
        signals = prep_signals.signals.T

    # ref_channels = modal_data.prep_signals.ref_channels
    sampling_rate = modal_data.prep_signals.sampling_rate
    channel_headers = modal_data.prep_signals.channel_headers
    modal_contributions = modal_data._modal_contributions

    if channel_inds is None:
        channel_inds = np.arange(modal_data.prep_signals.num_analised_channels)
    num_channels = len(channel_inds)
    if ref_channel_inds is None:
        ref_channel_inds = np.arange(modal_data.prep_signals.num_ref_channels)
    # num_ref_channels = len(ref_channel_inds)

    # Plot signals for each mode and each channel
    num_modes = sig_synth.shape[-1]
    if modelist is None:
        modelist = list(range(num_modes))
    num_plots = len(modelist) + 2

    fig1, axes = plt.subplots(num_plots, 1, sharex='col', sharey='col', squeeze=False)

    for ip, i in enumerate(modelist):
        rho = modal_contributions[i]
        this_signals = sig_synth[:,:, i]
        for j in range(num_channels):

            i_l = channel_inds[j]
            color = str(np.linspace(0, 1, len(channel_inds) + 2)[j + 1])
            ls = 'solid'

            axes[ip, 0].plot(t, this_signals[i_l,:], color=color, ls=ls)

        axes[ip, 0].set_ylabel(f'$\delta_{{{i + 1}}}$={rho:1.2f}',
                               rotation=0, labelpad=40, va='center', ha='left')

    for j in range(num_channels):

        i_l = channel_inds[j]
        color = str(np.linspace(0, 1, len(channel_inds) + 2)[j + 1])
        ls = 'solid'

        this_signals = signals[i_l,:]
        this_signals_synth = np.sum(sig_synth, axis=2)[i_l,:]

        axes[-1, 0].plot(t, this_signals, color=color, ls=ls,
                         label=f'{channel_headers[i_l]}')
        axes[-2, 0].plot(t, this_signals_synth, color=color, ls=ls,)

        axes[-1, 0].set_ylabel('Measured', rotation=0, labelpad=50, va='center', ha='left')
        axes[-2, 0].set_ylabel(f'$\sum\delta$={np.sum(modal_contributions):1.2f}',
                               rotation=0, labelpad=50, va='center', ha='left')

    axes[-1, 0].set_xlabel('$t$ [\si{\second}]')

    for ax in axes.flat:
        ax.set_yticks([])
    fig1.legend(title='Channels')

    # Plot power spectral density functions for each channel and all modes

    fig2, axes = plt.subplots(num_channels, 1, sharex='col', sharey='col', squeeze=False)

    ft_freq = np.fft.rfftfreq(len(t), d=(1 / sampling_rate))

    for j in range(num_channels):
        i_l = channel_inds[j]

        this_signals = signals[i_l,:]
        this_signals_synth = sig_synth[i_l,:,:]

        ft_meas = np.fft.rfft(this_signals * np.hanning(len(t)))

        if j == 0: label = f'Inp.'
        else: label = None

        axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_meas)), ls='solid', color='k', label=label)

        for ip, i in enumerate(modelist):
            ft_synth = np.fft.rfft(this_signals_synth[:, i] * np.hanning(len(t)))

            color = str(np.linspace(0, 1, len(modelist) + 2)[ip + 1])
            ls = ['-', '--', ':', '-.'][i % 4]
            if j == 0: label = f'm={i+1}'
            else: label = None

            axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_synth)), color=color, ls=ls, label=label)
            axes[j, 0].set_ylabel(f'{channel_headers[i_l]}',
                                 rotation=0, labelpad=20, va='center', ha='center')

    axes[-1, 0].set_xlabel('$f$ [\si{\hertz}]')
    for ax in axes.flat:
        ax.set_yticks([])
        ax.set_xlim(0, 1 / 2 * sampling_rate)

    fig2.legend(title='Mode')

    return fig1, fig2


def main():
    pass


if __name__ == '__main__':
    main()
