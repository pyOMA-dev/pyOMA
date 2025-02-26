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

Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015-2018
'''

import os
import warnings
import copy

import numpy as np
import scipy.linalg

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .Helpers import validate_array, simplePbar

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class BRSSICovRef(ModalBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.
        self.state = [False, False, False]

        self.num_block_columns = None
        self.num_block_rows = None

        self.U = None
        self.S = None
        self.V_T = None

        self.modal_contributions = None

    @property
    def accel_channels(self):
        return self.prep_signals.accel_channels

    @property
    def velo_channels(self):
        return self.prep_signals.velo_channels

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        assert os.path.exists(conf_file)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_columns = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        ssi_object = cls(prep_signals)
        ssi_object.build_toeplitz_cov(num_block_columns)
        ssi_object.compute_modal_params(max_model_order)

        return ssi_object

    def build_toeplitz_cov(
            self,
            num_block_columns=None,
            num_block_rows=None,
            shift=0):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags and
        decomposes it by a Singular Value decomposition.
        
        ::

              <-num_block_columns * n_r ->  _
            [     R_m      R_m-1      ...      R_1    ]^
            [     R_m+1    R_m        ...      R_2    ]num_block_rows * n_l
            [     ...      ...        ...      ...    ]
            [     R_2m-1   ...        ...      R_m    ]v
        
        The total number of block columns and block rows should not exceed the
        maximum time lag of pre-computed correlation functions:
        
        num_block_columns + num_block_rows + shift < prep_signals.m_lags
        
        Parameters
        ----------
            num_block_columns: integer, optional
                Number of block columns. By default, half the number of time
                lags are used
            
            num_block_rows: integer, optional
                Number of block rows. By default it is set equal to num_block_columns
            
            shift: integer, optional
                Allows the assembly of a shifted Block-Toeplitz matrix, s. t. 
                the correlation function starting at shift is assembled into the 
                block Toeplitz matrix
        '''

        max_lags = self.prep_signals.m_lags

        if num_block_columns is not None:
            assert isinstance(num_block_columns, int)
        else:
            if max_lags is None:
                raise RuntimeError('Either num_block_columns, or pre-computed correlation functions must be provided.')

            if num_block_rows is not None:
                assert isinstance(num_block_rows, int)
                num_block_columns = max_lags - num_block_rows - shift
            else:
                num_block_columns = (max_lags - shift) // 2

        if num_block_rows is None:
            num_block_rows = num_block_columns

        logger.info('Assembling Toeplitz matrix using pre-computed correlation functions'
              ' {} block-columns and {} block rows'.format(num_block_columns, num_block_rows + 1))

        m_lags = num_block_rows + 1 + num_block_columns - 1 + shift

        if max_lags is None:
            max_lags = self.prep_signals.m_lags

        if max_lags is not None and max_lags < m_lags:
            logger.warning('The pre-computed correlation function is too short for the requested matrix dimensions.')

        n_l = self.num_analised_channels
        n_r = self.num_ref_channels

        corr_matrix = self.prep_signals.correlation(m_lags)

        Toeplitz_matrix = np.zeros((n_l * (num_block_rows + 1), n_r * num_block_columns))

        for ii in range(num_block_columns):
            tau = num_block_columns - ii + shift
            this_block = corr_matrix[:,:, tau - 1]

            Toeplitz_matrix[:n_l, ii * n_r:(ii * n_r + n_r)] = this_block

        for i in range(1, num_block_rows + 1):
            # shift previous block row down and left
            previous_Toeplitz_row = (i - 1) * n_l
            this_block = Toeplitz_matrix[previous_Toeplitz_row:(
                previous_Toeplitz_row + n_l), 0:n_r * (num_block_columns - 1)]
            begin_Toeplitz_row = i * n_l
            Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row + n_l),
                            n_r:(n_r * num_block_columns)] = this_block

            # fill right most block
            tau = num_block_columns + i + shift
            this_block = corr_matrix[:,:, tau - 1]

            Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row + n_l),
                            0:n_r] = this_block

        logger.info('Decomposing Toeplitz matrix')

        U, S, V_T = scipy.linalg.svd(Toeplitz_matrix, 1)

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows

        self.U = U
        self.S = S
        self.V_T = V_T

        self.state[0] = True

    def compute_modal_params(self, max_model_order=None,
                             max_modes=None, algo='svd',
                             modal_contrib=True):
        '''
        Perform a multi-order computation of modal parameters. Successively
        calls 
        
         * estimate_state(order, max_modes, algo)
         * modal_analysis(A,C)
         * synthesize_correlation(A,C, G), if modal_contrib == True
        
        At ascending model orders, up to max_model_order. 
        See the explanations in the the respective methods, for a detailed 
        explanation of parameters.
        
        Parameters
        ----------
            max_model_order: integer, optional
                Maximum model order, where to interrupt the algorithm. If not given,
                it is min(num_channels * (num_block_rows + 1), num_reference_channels * num_block_columns)
        '''

        if max_model_order is not None:
            assert isinstance(max_model_order, int)
        else:
            max_model_order = self.S.shape[0]

        assert max_model_order <= self.S.shape[0]

        num_analised_channels = self.num_analised_channels

        logger.info('Computing modal parameters...')

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        mode_shapes = np.zeros((num_analised_channels, max_model_order, max_model_order),
                               dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_model_order), dtype=complex)

        if modal_contrib:
            modal_contributions = np.zeros((max_model_order, max_model_order))
        else:
            modal_contributions = None

        pbar = simplePbar(max_model_order - 1)
        for order in range(1, max_model_order):
            next(pbar)

            A, C, G = self.estimate_state(order, max_modes, algo)

            f, d, phi, lamda, = self.modal_analysis(A, C)
            modal_frequencies[order,:order] = f
            modal_damping[order,:order] = d
            mode_shapes[:phi.shape[0],:order, order] = phi
            eigenvalues[order,:order] = lamda

            if modal_contrib:
                _, delta = self.synthesize_correlation(A, C, G)
                modal_contributions[order,:order] = delta

        self.max_model_order = max_model_order

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues
        self.modal_contributions = modal_contributions

        self.state[2] = True

    def estimate_state(self, order, max_modes=None, algo='svd'):
        '''
        
        Compute the state matrix A, output matrix C and next-state-output 
        covariance matrix G from the singular values and vectors of the 
        block Toeplitz matrix, truncated at the requested order. Estimation of the
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
                
            G: numpy.ndarray
                next-state-output covariance matrix : Array of shape (order, num_ref_channels)
        '''
        if order > self.S.shape[0]:
            raise RuntimeError(f'Order cannot be higher than {self.S.shape[0]}. Consider using more block_rows/block_columns.')

        assert algo in ['svd', 'qr']

        n_l = self.num_analised_channels
        n_r = self.num_ref_channels

        num_block_rows = self.num_block_rows

        U = self.U[:,:order]
        S = self.S[:order]
        V_T = self.V_T[:order,:]

        # compute state-space model
        S_2 = np.power(S, 0.5)
        O = U * S_2[np.newaxis,:]
        Z = S_2[:, np.newaxis] * V_T

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
        G = Z[:order, -n_r:]  # next-state-output covariance matrix

        return A, C, G

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
            modal_frequencies: (order,) numpy.ndarray 
                Array holding the modal frequencies for each mode
            modal_damping: (order,) numpy.ndarray 
                Array holding the modal damping ratios (0,100) for each mode
            mode_shapes: (n_l, order,) numpy.ndarray 
                Complex array holding the mode shapes 
            eigenvalues: (order,) numpy.ndarray
                Complex array holding the eigenvalues for each mode
        '''
        # collect variables
        accel_channels = self.accel_channels
        velo_channels = self.velo_channels
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

    def synthesize_correlation(self, A, C, G):
        '''
        Correlation function synthetization in a modal decoupled form follows 
        Reynders-2012-SystemIdentificationMethodsFor(Operational)ModalAnalysisReviewAndComparison
        Eq. 161 p. 74 (24) where \Lambda are the correlation functions of the identified system
                
        Parameters
        ----------
            A: numpy.ndarray
                State matrix: Array of shape (order, order)
                
            C: numpy.ndarray
                Output matrix: Array of shape (num_analised_channels, order)
                
            G: numpy.ndarray
                next-state-output covariance matrix : Array of shape (order, num_ref_channels)
        
        Returns
        -------
            corr_matrix_synth: (n_l, n_r, m_lags, n_modes) numpy.ndarray
                Array holding the modally decomposed correlation functions for 
                each channel n_l and reference channel n_r and all modes
                
            modal_contributions: (order,) numpy.ndarray
                Array holding the contributions of each mode to the input 
                correlation function.
                
                
        To use cross-validation for the modal contributions, manually replace 
        the correlation matrix, between steps system identification and modal analysis:
        
        .. code-block:: python
        
            order=40
            m_lags= 150
            
            n_blocks = 40
            k = 10
            
            cardinality = n_blocks // k
            block_indices = np.arange(cardinality*k)
            np.random.shuffle(block_indices)
            
            prep_signals.corr_blackman_tukey(max_m_lags, num_blocks=n_blocks, refs_only=True)
            
            i = 1 # subset i out of k
            
            test_set = block_indices[i * cardinality:(i + 1) * cardinality]
            training_set = np.take(block_indices, np.arange((i + 1) * cardinality, (i + k) * cardinality), mode='wrap')
            
            # use the training set for building the Toeplitz matrix
            prep_signals.corr_matrix_bt = np.mean(prep_signals.corr_matrices_bt[training_set,...,:this_m_lags], axis=0)
            modal_data.build_toeplitz_cov(int(this_m_lags // 2))
            
            # use the test set for modal analysis
            prep_signals.corr_matrix_bt = np.mean(prep_signals.corr_matrices_bt[test_set,...,:this_m_lags], axis=0)
            
            A, C, G = modal_data.estimate_state(order)
            this_modal_frequencies, this_modal_damping, this_mode_shapes, this_eigenvalues, = modal_data.modal_analysis(A, C)
            _, this_modal_contributions = modal_data.synthesize_correlation(A, C, G)
            
        . . .
        '''

        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns

        n_l = self.num_analised_channels
        n_r = self.num_ref_channels

        order = A.shape[0]
        assert order == A.shape[1]

        m_lags = num_block_rows + 1 + num_block_columns - 1
        # m_lags = self.prep_signals.m_lags
        corr_matrix_data = self.prep_signals.corr_matrix[:,:,:m_lags]

        corr_mats_shape = (n_l, n_r, m_lags, order // 2)
        corr_matrix_synth = np.zeros(corr_mats_shape, dtype=np.float64)

        Sigma_data = np.zeros((n_l * n_r))
        Sigma_synth = np.zeros((n_l * n_r))
        Sigma_data_synth = np.zeros((n_l * n_r, order))

        modal_contributions = np.zeros((order))

        # redundant: eigendecomposition is recomputed here for better readability of code
        eigvals, eigvecs_r = np.linalg.eig(A)
        Phi = C.dot(eigvecs_r)

        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)

        # Peeters-2000-SystemIdentificationAndDamageDetectionInCivilEngineering Eq. 2.57
        G_m = np.linalg.solve(eigvecs_r, G)

        if logger.isEnabledFor(logging.DEBUG):
            tau = 21
            logger.debug(C @ eigvecs_r @ np.diag(eigvals) ** tau @ np.linalg.inv(eigvecs_r) @ G)
            logger.debug(Phi @ np.diag(eigvals) ** tau @ G_m)

        for i, ind in enumerate(conj_indices):

            lambda_i = eigvals[ind]

            conjs_ind = eigvals == lambda_i.conj()
            conjs_ind[ind] = 1

            conj_eigvals = eigvals[conjs_ind][np.newaxis]

            conj_Phis = Phi[:, conjs_ind]
            conj_Gms = G_m[conjs_ind,:]

            eigspowtau = conj_eigvals[np.newaxis, ...] ** np.arange(m_lags)[:, np.newaxis, np.newaxis]
            this_corr_synth = (eigspowtau * conj_Phis[np.newaxis, ...]).dot(conj_Gms)

            if not np.all(np.isclose(this_corr_synth.imag, 0)):
                logger.warning(f'Synthetized correlation functions are complex for mode index {ind}. Something is wrong!')

            this_corr_synth = np.transpose(this_corr_synth.real, (1, 2, 0))

            corr_matrix_synth[:,:,:, i] = this_corr_synth

        if logger.isEnabledFor(logging.DEBUG):
            Sigma_data_synthtot = np.zeros((n_l * n_r))

        for i_r in range(n_r):
            for i_l in range(n_l):
                corr_data = corr_matrix_data[i_l, i_r,:]
                corr_synth = np.sum(corr_matrix_synth, axis=3)[i_l, i_r,:]

                Sigma_data[i_r * n_l + i_l] = corr_data.dot(corr_data.T)
                Sigma_synth[i_r * n_l + i_l] = corr_synth.dot(corr_synth.T)

                if logger.isEnabledFor(logging.DEBUG):
                    Sigma_data_synthtot[i_r * n_l + i_l] = corr_data.dot(corr_synth.T)

                for i, ind in enumerate(conj_indices):
                    Sigma_data_synth[i_r * n_l + i_l, i] = corr_data @ corr_matrix_synth[i_l, i_r,:, i]

        for i, ind in enumerate(conj_indices):
            rho = (Sigma_data_synth[:, i] / np.sqrt(Sigma_data * Sigma_synth))
            modal_contributions[i] = rho.mean()

        self._corr_matrix_synth = corr_matrix_synth
        self._modal_contributions = modal_contributions

        return corr_matrix_synth, modal_contributions

    def synthesize_spectrum(self, A, C, G):
        '''

        L = N*dt (duration = number_of_samples*sampling_period)
        P = N*df (maximal frequency = number of samples * frequency inverval)

        dt * df = 1/N
        L * P = N
        '''
        logger.warning('Implementation: Spectrum estimation is not tested.')
        # f_max = self.prep_signals.sampling_rate / 2
        m_lags = self.prep_signals.m_lags
        delta_t = 1 / self.prep_signals.sampling_rate

        num_analised_channels = self.num_analised_channels
        num_ref_channels = self.num_ref_channels

        order = A.shape[0]
        assert order == A.shape[1]

        psd_mats_shape = (num_analised_channels, num_ref_channels, m_lags)
        psd_matrix = np.zeros(psd_mats_shape, dtype=np.float64)

        I = np.identity(order)

        Lambda_0 = self.prep_signals.signal_power()

        for n in range(m_lags):

            z = np.exp(0 + 1j * n * delta_t)
            psd_matrix[:,:, n] = C.dot(np.linalg.solve(
                z * I - A, G)) + Lambda_0 + G.T.dot(np.linalg.solve(1 / z * I - A.T, C.T))

        self._psd_matrix = psd_matrix

    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        # out_dict['self.prep_signals']=self.prep_signals
        if self.state[0]:  # covariances
            # out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.U'] = self.U
            out_dict['self.S'] = self.S
            out_dict['self.V_T'] = self.V_T
        if self.state[2]:  # modal params

            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.modal_contributions'] = self.modal_contributions
            out_dict['self.max_model_order'] = self.max_model_order

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname)
        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = validate_array(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Covariance Matrices Built',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state:
                logger.info(state_string)

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = validate_array(in_dict['self.setup_name'])
        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        ssi_object = cls(prep_signals)
        ssi_object.state = state
        if state[0]:  # covariances
            # ssi_object.toeplitz_matrix = in_dict['self.toeplitz_matrix']
            ssi_object.num_block_columns = validate_array(in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = validate_array(in_dict['self.num_block_rows'])
            ssi_object.U = validate_array(in_dict['self.U'])
            ssi_object.S = validate_array(in_dict['self.S'])
            ssi_object.V_T = validate_array(in_dict['self.V_T'])
        if state[2]:  # modal params
            ssi_object.modal_frequencies = validate_array(in_dict['self.modal_frequencies'])
            ssi_object.modal_damping = validate_array(in_dict['self.modal_damping'])
            ssi_object.mode_shapes = validate_array(in_dict['self.mode_shapes'])
            ssi_object.eigenvalues = validate_array(in_dict['self.eigenvalues'])
            ssi_object.modal_contributions = validate_array(in_dict.get(
                'self.modal_contributions', None))
            ssi_object.max_model_order = validate_array(in_dict['self.max_model_order'])

        return ssi_object


def show_channel_reconstruction(modal_data, modelist=None, channel_list=None, ref_channel_list=None, axes=None):

    import matplotlib.pyplot as plt
    corr_matrix_synth = modal_data._corr_matrix_synth

    corr_matrix_data = modal_data.prep_signals.corr_matrix
    if channel_list is None:
        channel_list = np.arange(modal_data.prep_signals.num_analised_channels)
    if ref_channel_list is None:
        ref_channel_list = np.arange(modal_data.prep_signals.num_ref_channels)

    num_modes = corr_matrix_synth.shape[-1]
    if modelist is None:
        modelist = list(range(num_modes))

    ratio = len(channel_list) / len(ref_channel_list)

    num_plots = len(modelist)
    n_rows = int(num_plots / ratio)
    n_cols = int(np.ceil(num_plots / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)

    RMS_err = np.zeros((len(channel_list), len(ref_channel_list), num_modes))

    for mode in modelist:
        # Error is not normalized and tends to be larger for channels with stronger signals
        RMS_err[..., mode] = np.sqrt(np.mean(np.power(corr_matrix_synth[..., mode] - corr_matrix_data, 2), axis=-1))
    vmin, vmax = np.min(RMS_err), np.max(RMS_err)

    for mode in modelist:
        mappable = axes.flat[mode].imshow(RMS_err[..., mode], vmin=vmin, vmax=vmax)
        if axes.flat[mode].get_subplotspec().is_first_col():
            axes.flat[mode].set_yticks(np.arange(len(channel_list)), modal_data.prep_signals.channel_headers)

        if axes.flat[mode].get_subplotspec().is_last_row():
            axes.flat[mode].set_xticks(np.arange(len(ref_channel_list)), np.array(modal_data.prep_signals.channel_headers)[modal_data.prep_signals.ref_channels])
        # axes.flat[mode].set_title(f'{modal_data._modal_frequencies[mode]:1.3f} Hz')
    cbar = fig.colorbar(mappable)
    cbar.set_label('RMS error')


def plot_corr_synth(modal_data, modelist=None, channel_inds=None, ref_channel_inds=None, axes=None):

    import matplotlib.pyplot as plt
    corr_matrix_synth = modal_data._corr_matrix_synth
    m_lags = corr_matrix_synth.shape[2]
    corr_matrix_data = modal_data.prep_signals.corr_matrix[:,:,:m_lags]

    ref_channels = modal_data.prep_signals.ref_channels
    sampling_rate = modal_data.prep_signals.sampling_rate
    channel_headers = modal_data.prep_signals.channel_headers
    modal_contributions = modal_data._modal_contributions

    if channel_inds is None:
        channel_inds = np.arange(modal_data.prep_signals.num_analised_channels)
    num_channels = len(channel_inds)
    if ref_channel_inds is None:
        ref_channel_inds = np.arange(modal_data.prep_signals.num_ref_channels)
    num_ref_channels = len(ref_channel_inds)

    # build non-repeating channel combinations
    # contains indices for all_channels (=channel numbers) and ref_channels
    # channel numbers for ref_channels can be obtained from prep_signals.ref_channels
    i_l_i_r = np.full((num_channels * num_ref_channels, 2), np.nan)
    j = 0
    for index_l in channel_inds:

        if index_l in ref_channels:
            index_l_in_ref_channels = ref_channels.index(index_l)
        else:
            index_l_in_ref_channels = None

        for index_r in ref_channel_inds:
            if index_l_in_ref_channels is None:
                i_l_i_r[j, 0] = index_l
                i_l_i_r[j, 1] = index_r
                j += 1
            else:
                index_r_in_all_channels = ref_channels[index_r]
                inds_inverted = np.array([[index_r_in_all_channels, index_l_in_ref_channels]])
                if not (np.any(np.all(i_l_i_r == inds_inverted , axis=1))):
                    i_l_i_r[j, 0] = index_l
                    i_l_i_r[j, 1] = index_r
                    j += 1

    i_l_i_r = i_l_i_r[~np.all(np.isnan(i_l_i_r), axis=1),:].astype(int)

    # Plot correlation functions for each mode and all channel combinations
    num_modes = corr_matrix_synth.shape[-1]
    if modelist is None:
        modelist = list(range(num_modes))
    num_plots = len(modelist) + 2

    fig1, axes = plt.subplots(num_plots, 1, sharex='col', sharey='col', squeeze=False)

    taus = np.linspace(0, m_lags / sampling_rate, m_lags)

    for ip, i in enumerate(modelist):
        rho = modal_contributions[i]
        this_corr_synth = corr_matrix_synth[:,:,:, i]
        for j in range(len(i_l_i_r)):
            i_l, i_r = i_l_i_r[j,:]

            color = str(np.linspace(0, 1, len(i_l_i_r) + 2)[j + 1])
            ls = 'solid'

            axes[ip, 0].plot(taus, this_corr_synth[i_l, i_r,:], color=color, ls=ls)

        axes[ip, 0].set_ylabel(f'$\delta_{{{i + 1}}}$={rho:1.2f}',
                               rotation=0, labelpad=40, va='center', ha='left')

    for j in range(len(i_l_i_r)):
        i_l, i_r = i_l_i_r[j,:]

        color = str(np.linspace(0, 1, len(i_l_i_r) + 2)[j + 1])
        ls = 'solid'

        this_corr_data = corr_matrix_data[i_l, i_r,:]
        this_corr_synth = np.sum(corr_matrix_synth, axis=3)[i_l, i_r,:]

        axes[-1, 0].plot(taus, this_corr_data, color=color, ls=ls,
                         label=f'{channel_headers[i_l]} $\leftrightarrow$  {channel_headers[ref_channels[i_r]]}')
        axes[-2, 0].plot(taus, this_corr_synth, color=color, ls=ls,)

        axes[-1, 0].set_ylabel('Measured', rotation=0, labelpad=50, va='center', ha='left')
        axes[-2, 0].set_ylabel(f'$\sum\delta$={np.sum(modal_contributions):1.2f}',
                               rotation=0, labelpad=50, va='center', ha='left')

    axes[-1, 0].set_xlabel('$\\tau$ [\si{\second}]')

    for ax in axes.flat:
        ax.set_yticks([])
        ax.set_xlim(0, taus.max() / 2)
    fig1.legend(title='Channels')
    fig1.subplots_adjust(left=None, bottom=None, right=0.97, top=0.97, wspace=None, hspace=0.1,)

    # Plot power spectral density functions for each channel combination and all modes
    num_plots = len(i_l_i_r)

    fig2, axes = plt.subplots(num_plots, 1, sharex='col', sharey='col', squeeze=False)

    ft_freq = np.fft.rfftfreq(m_lags, d=(1 / sampling_rate))

    for j in range(num_plots):
        i_l, i_r = i_l_i_r[j,:]

        this_corr_data = corr_matrix_data[i_l, i_r,:]
        ft_meas = np.fft.rfft(this_corr_data * np.hanning(m_lags))

        if j == 0: label = f'Inp.'
        else: label = None

        axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_meas)), ls='solid', color='k', label=label)

        for ip, i in enumerate(modelist):
            ft_synth = np.fft.rfft(corr_matrix_synth[i_l, i_r,:, i] * np.hanning(m_lags))

            color = str(np.linspace(0, 1, len(modelist) + 2)[ip + 1])
            ls = ['-', '--', ':', '-.'][i % 4]
            if j == 0: label = f'm={i+1}'
            else: label = None

            axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_synth)), color=color, ls=ls, label=label)
            axes[j, 0].set_ylabel(f'{channel_headers[i_l]}\n $\leftrightarrow$ \n{channel_headers[ref_channels[i_r]]}',
                                 rotation=0, labelpad=20, va='center', ha='center')

    axes[-1, 0].set_xlabel('$f$ [\si{\hertz}]')
    for ax in axes.flat:
        ax.set_yticks([])
        ax.set_xlim(0, 1 / 2 * sampling_rate)

        ax.set_ylim(ymin=-90)
    fig2.legend(title='Mode')
    fig2.subplots_adjust(left=None, bottom=None, right=0.97, top=0.97, wspace=None, hspace=0.1,)

    return fig1, fig2


class PogerSSICovRef(BRSSICovRef):
    '''
    "In the PoGER approach, first a nonparametric system model is identified
    for each setup separately. In the time domain, this nonparametric model
    consists of the correlations between all measured outputs.
    In a second step, the output correlations obtained from the different
    setups are stacked on top of each other. Extracting the modal parameters
    from the resulting correlation function yields global values for the
    eigenfrequencies and damping ratios. The identified partial mode shapes
    are stacked on top of each other in a global mode shape. However, due
    to the non-stationary ambient excitation level and the non-stationary
    ambient excitation color, it is necessary to re-scale the partial mode
    shapes in a least-squares sense, for instance to the reference DOFs of the
    first partial mode shape, just as in the PoSER approach"

    from:
    Döhler, M.; Reynders, E.; Magalhaes, F.; Mevel, L.; Roeck, G. D. & Cunha, A.
    Pre-and post-identification merging for multi-setup OMA with covariance-driven SSI
    28th International Modal Analysis Conference, 2010 , 57-70

    Analysis steps:
    
        * Create your geometry definitions
        * Create configuration files and channel-dof-assignments for each setup
        * Pre-process each setup using PreProcessData
        * Pre-compute correlations functions using 
          PreProcessData.compute_correlation_functions
          (note: m_lags >= num_block_columns + num_block_rows >= 2 * num_block_columns + 1)
        * add the PreProcessData objects of each setup using add_setup
        * call pair_channels(), build_merged_subspace_matrix(), estimate_state(),
          compute_modal_params()

    Notes on the reference channels:
    There are two different uses of reference channels:
    
        1. Reference channels for reducing the computational effort /
           improving results if noisy channels are present
        2. Reference channels for mode shape rescaling when multiple
           setups should be merged

    In PoGER merging the first group of reference channels are required
    for  joint identification. In this case, reference-based correlation
    functions are "stacked on top of each other" and then assembled into
    a joint Hankel matrix. Here, only the reference channels, that are
    present in all setups can be used.

    Based on each setups' channel-dof-assignments and selected reference
    channels, the PogerSSICovRef class automatically determines the
    reference channels for:
    
        * joint identification and
        * mode shape rescaling / merging.
     
    Thus, by changing the reference channel definition in each setup,
    the used reference channels in joint identification can be influenced.
    The reference channels for modeshape rescaling are automatically
    generated, regardless of the the definition in the setup. Rescaling
    is always done with respect to the first setup, so a "good" setup should
    always be added first.
    
    .. TODO::
        * Add modal contributions
        * Implement PreGER merging with variance computation in a new class
    '''

    def __init__(self,):
        '''
        Initializes class and all class variables
        channel definition: channels start at 0
        '''
        super().__init__()

        self.state = [False, False, False, False, False]

        # __init__
        self.setup_name = 'merged_'
        # self.start_times = []

        # add_setup
        self.setups = []
        self.sampling_rate = None
        self.num_ref_channels = None
        self.m_lags = None

        # pair_channels
        self.ssi_ref_channels = None
        self.merged_chan_dofs = None
        self.merged_accel_channels = None
        self.merged_velo_channels = None
        self.merged_disp_channels = None
        self.merged_num_channels = None
        self.num_analised_channels = None
        # self.start_time = None

        # build_merged_subspace_matrix
        self.subspace_matrix = None
        self.num_block_columns = None
        self.num_block_rows = None
        self.U = None
        self.S = None
        self.V_T = None

    def add_setup(self, prep_signals):
        '''
        todo:
        check that ref_channels are equal in each setup (by number and by DOF)
        '''
        assert isinstance(prep_signals, PreProcessSignals)

        # assure chan_dofs were assigned
        assert prep_signals.chan_dofs

        if self.sampling_rate is not None:
            assert prep_signals.sampling_rate == self.sampling_rate
        else:
            self.sampling_rate = prep_signals.sampling_rate

        if self.num_ref_channels is not None:
            if self.num_ref_channels != prep_signals.num_ref_channels:
                warnings.warn(
                    'This setup contains a different number of reference channels ({}), than the previous setups ({})!'.format(
                        prep_signals.num_ref_channels, self.num_ref_channels))
                self.num_ref_channels = min(
                    self.num_ref_channels, prep_signals.num_ref_channels)
        else:
            self.num_ref_channels = prep_signals.num_ref_channels

        if self.m_lags is not None:
            self.m_lags = min(self.m_lags, prep_signals.m_lags)
        else:
            self.m_lags = prep_signals.m_lags

        self.setup_name += prep_signals.setup_name + '_'
        # self.start_times.append(prep_signals.start_time)

        # extract needed information and store them in a dictionary
        self.setups.append({'setup_name': prep_signals.setup_name,
                            'num_analised_channels': prep_signals.num_analised_channels,
                            'chan_dofs': prep_signals.chan_dofs,
                            'ref_channels': prep_signals.ref_channels,
                            # 'roving_channels': prep_signals.roving_channels,
                            'accel_channels': prep_signals.accel_channels,
                            'velo_channels': prep_signals.velo_channels,
                            'disp_channels': prep_signals.disp_channels,
                            'corr_matrix': prep_signals.corr_matrix,
                            'start_time': prep_signals.start_time,
                            })

        logger.info(
            'Added setup "{}" with {} channels'.format(
                prep_signals.setup_name,
                prep_signals.num_analised_channels))

        # assign last setup, to be able to display spectra in stabil_plot
        # this actually created a bug in modal_analysis, where accel_channels and velo_channels
        # were passed from prep_signals, instead of the merged version
        # this bug would have been caught if prep_signals wouldn't exist in the object
        self.prep_signals = prep_signals

        self.state[3] = True

    def pair_channels(self,):
        '''
        pairs channels from all given setups for the poger merging methods

        ssi_reference channels are common to all setups
        rescale reference channels are common to at least two setups

        finds common dofs from all setups and their respective channels
        generates new channel_dof_assignments with ascending channel numbers
        rescale reference channels are assumed to be equal to ssi_reference channels
        '''

        logger.info('Pairing channels and dofs...')
        setups = self.setups
        merged_chan_dofs = []
        merged_accel_channels = []
        merged_velo_channels = []
        merged_disp_channels = []

        # extract dofs from each setup and exclude channel numbers
        # merged_chan_dofs will be a list of chan_dof lists
        '''
        merged_chan_dofs = [[dof of setup 0 channel 0,
                             dof of setup 0 channel 1,
                             ...
                             dof of setup 0 channel num_analised_channels]
                            [dof of setup 1 channel 0,
                             dof of setup 1 channel 1,
                             ...
                             dof of setup 1 channel num_analised_channels]
                            ...
                            [dof of setup num_setups channel 0,
                             dof of setup num_setups channel 1,
                             ...
                             dof of setup num_setups channel num_analised_channels]
                            ]
        '''

        for setup in setups:
            chan_dofs = []
            accel_channels = []
            velo_channels = []
            disp_channels = []

            this_chan_dofs = setup['chan_dofs']
            this_num_analised_channels = setup['num_analised_channels']
            # this_ref_channels = setup['ref_channels']
            # this_rov_channels = setup['roving_channels']

            this_accel_channels = setup['accel_channels']
            this_velo_channels = setup['velo_channels']
            this_disp_channels = setup['disp_channels']

            # chan dofs are now sorted by channel number
            this_chan_dofs.sort(key=lambda x: x[0])

            for channel in range(this_num_analised_channels):

                for chan_dof in this_chan_dofs:
                    if channel == chan_dof[0]:
                        node, az, elev = chan_dof[1:4]
                        if len(chan_dof) == 5:
                            name = chan_dof[4]
                        else:
                            name = ''
                        chan_dofs.append([node, az, elev, name])
                        break
                # if channel has not been assigned to a DOF
                else:
                    chan_dofs.append([None, 0, 0, ''])

                accel_channels.append(channel in this_accel_channels)
                velo_channels.append(channel in this_velo_channels)
                disp_channels.append(channel in this_disp_channels)

            merged_chan_dofs.append(chan_dofs)

            merged_accel_channels.append(accel_channels)
            merged_velo_channels.append(velo_channels)
            merged_disp_channels.append(disp_channels)

        # find dofs common to all setups
        # takes the dofs of the first setup as ssi_ref_dofs
        # loops over all setups and only keeps channels that are present in
        # the previous ssi_ref_dofs and the current setup

        # only ssi_ref_dofs can be used in the assembly of the hankel matrix
        # for mode shape rescaling the ref_dofs between the respective combination of two setups could be used
        # but the logic for this is not included here, therefore
        # only ssi_ref_dofs will be used for mode shape rescaling

        ssi_ref_dofs = copy.deepcopy(merged_chan_dofs[0])
        for chan_dofs in merged_chan_dofs[1:]:
            new_ref_dofs = []
            for node, az, elev, name in chan_dofs:
                if node is None:
                    continue
                for rnode, raz, relev, rname in ssi_ref_dofs:
                    if node == rnode and az == raz and elev == relev and name == rname:
                        new_ref_dofs.append((rnode, raz, relev, rname))
                        break
            ssi_ref_dofs = new_ref_dofs
            if len(ssi_ref_dofs) == 0:
                raise RuntimeError(
                    'Could not find any DOF that is common to all setups.')

        # find channels for each common dof
        #     add the channel number to rescale_ref_channels; these will
        #        be used to get the reference modal coordinates in mode shape rescaling
        #     add the index of the channel in setups' ref_channels to
        #        ssi_ref_channels; these will be used to assemble the Hankel
        #        matrix were each column is assembled from a single reference DOF (!)
        #        if reference channel orders have changed between setups, reordering
        #        is needed to achieve constistent columns of the Hankel matrix

        ssi_ref_channels = []
        # base_ssi_ref_channels = ssi_ref_channels[0]
        rescale_ref_channels = []
        for setup, chan_dofs in zip(setups, merged_chan_dofs):
            # prep_signals = setup['prep_signals']
            this_ssi_ref_channels = []
            this_rescale_ref_channels = []
            for rnode, raz, relev, rname in ssi_ref_dofs:
                index = None
                for channel, node, az, elev, name in setup['chan_dofs']:
                    if node == rnode and az == raz and elev == relev and name == rname:
                        # index = i
                        # channel = setup['chan_dofs'][index][0]

                        this_rescale_ref_channels.append(int(channel))

                        if channel not in setup['ref_channels']:
                            warnings.warn(
                                'Channel {} ({}) is common to multiple setups but not chosen as a reference channel.'.format(
                                    channel, name))
                        else:
                            this_ref_index = setup['ref_channels'].index(
                                channel)
                            this_ssi_ref_channels.append(this_ref_index)

                        break
                else:
                    raise RuntimeError(
                        'Oops! Something went wrong. This should not happen!')
            rescale_ref_channels.append(this_rescale_ref_channels)
            ssi_ref_channels.append(this_ssi_ref_channels)
            # print(this_ssi_ref_channels)

        # reorder chan_dofs, accel_channels, etc. of the first setup
        # s.t. references come first, followed by rovings
        # refs are ordered by ssi_ref_dofs order
        # rovs are ordered by ascending channel number of the underlying setup

        new_chan_dofs, new_accel_channels, new_velo_channels, new_disp_channels = [], [], [], []
        chan_dofs, accel_channels, velo_channels, disp_channels = merged_chan_dofs[
            0], merged_accel_channels[0], merged_velo_channels[0], merged_disp_channels[0]
        # print(chan_dofs,accel_channels, velo_channels, disp_channels )
        for rnode, raz, relev, rname in ssi_ref_dofs:
            for i, (node, az, elev, name) in enumerate(chan_dofs):
                if node == rnode and az == raz and elev == relev and name == rname:
                    new_chan_dofs.append(chan_dofs[i])
                    new_accel_channels.append(accel_channels[i])
                    new_velo_channels.append(velo_channels[i])
                    new_disp_channels.append(disp_channels[i])
                    break
            else:
                raise RuntimeError(
                    'This should not happen, as all ref_dofs were previously checked to be present in each setup.')
            del chan_dofs[i]
            del accel_channels[i]
            del velo_channels[i]
            del disp_channels[i]
        new_chan_dofs += chan_dofs
        new_accel_channels += accel_channels
        new_velo_channels += velo_channels
        new_disp_channels += disp_channels

        merged_chan_dofs[0], merged_accel_channels[0], merged_velo_channels[0], merged_disp_channels[
            0] = new_chan_dofs, new_accel_channels, new_velo_channels, new_disp_channels

        # delete channels of the reference dofs
        for chan_dofs, accel_channels, velo_channels, disp_channels in zip(
                merged_chan_dofs[1:], merged_accel_channels[1:], merged_velo_channels[1:], merged_disp_channels[1:]):
            # prep_signals = setup['prep_signals']
            for rnode, raz, relev, rname in ssi_ref_dofs:
                index = None
                for i, (node, az, elev, name) in enumerate(chan_dofs):
                    if node == rnode and az == raz and elev == relev and name == rname:
                        index = i
                        break
                else:
                    raise RuntimeError(
                        'This should not happen, as all ref_dofs were previously checked to be present in each setup.')
                # remove the channel_dof_assignment of the reference channels
                # for all setups
                del chan_dofs[index]
                del accel_channels[index]
                del velo_channels[index]
                del disp_channels[index]

        # flatten chan_dofs and add ascending channel numbers
        flattened = []
        channel = 0
        for sublist in merged_chan_dofs:
            for val in sublist:
                val.insert(0, channel)
                flattened.append(val)
                channel += 1
        merged_chan_dofs = flattened

        flattened = []
        channel = 0
        for sublist in merged_accel_channels:
            for val in sublist:
                if val:
                    flattened.append(channel)
                channel += 1
        merged_accel_channels = flattened

        flattened = []
        channel = 0
        for sublist in merged_velo_channels:
            for val in sublist:
                if val:
                    flattened.append(channel)
                channel += 1
        merged_velo_channels = flattened

        flattened = []
        channel = 0
        for sublist in merged_disp_channels:
            for val in sublist:
                if val:
                    flattened.append(channel)
                channel += 1
        merged_disp_channels = flattened

        num_analised_channels = sum(
            [setup['num_analised_channels'] for setup in setups])

        self.merged_accel_channels = merged_accel_channels
        self.merged_velo_channels = merged_velo_channels
        self.merged_disp_channels = merged_disp_channels

        self.ssi_ref_channels = ssi_ref_channels
        self.rescale_ref_channels = rescale_ref_channels
        self.merged_chan_dofs = merged_chan_dofs
        self.merged_num_channels = len(merged_chan_dofs)

        self.num_analised_channels = num_analised_channels
        self.start_time = min([stp['start_time'] for stp in setups])

        self.state[1] = True

        return ssi_ref_channels, merged_chan_dofs

    @property
    def accel_channels(self):
        return self.merged_accel_channels

    @property
    def velo_channels(self):
        return self.merged_velo_channels

    def build_merged_subspace_matrix(
            self,
            num_block_columns,
            num_block_rows=None):
        '''
        Builds a Block-Hankel Matrix of Covariances with varying time lags

        ::

              <- num_block_columns*num_ref_channels-> _
            [     R_1      R_2      ...      R_i     ]^
            [     R_2      R_3      ...      R_2     ]num_block_rows*(num_num_ref_channels*num_setups)
            [     ...      ...      ...      ...     ]v
            [     R_i      ...      ...      R_2i-1  ]_

            R_1 =   [ R_1^1          ]
                    [ R_1^2          ]
                    [ ...            ]
                    [ R_1^num_setups ]

        '''

        assert isinstance(num_block_columns, int)

        if num_block_rows is None:
            num_block_rows = num_block_columns  # -10
        assert isinstance(num_block_rows, int)

        if not num_block_columns + num_block_columns + 1 <= self.m_lags:
            raise RuntimeError(
                'Correlation functions were pre-computed '
                'up to {} time lags, which is sufficient for assembling '
                'a Hankel-Matrix with up to {} x {} blocks. You requested '
                '{} x {} blocks'.format(
                    self.m_lags,
                    self.m_lags // 2 + 1,
                    self.m_lags // 2,
                    num_block_rows + 1,
                    num_block_columns))

        setups = self.setups

        logger.info(
            'Assembling subspace matrix using pre-computed correlation'
            ' functions from {} setups with {} block-columns and {} '
            'block rows'.format(
                len(setups),
                num_block_columns,
                num_block_rows + 1))

        ssi_ref_channels = self.ssi_ref_channels

        num_analised_channels = self.num_analised_channels
        num_ref_channels = len(ssi_ref_channels[0])
        m_lags = self.m_lags

        subspace_matrix = np.zeros(
            ((num_block_rows + 1) * num_analised_channels,
             num_block_columns * num_ref_channels))
        end_row = None
        for block_row in range(num_block_rows + 1):
            sum_analised_channels = 0
            for this_ssi_ref_channels, setup in zip(ssi_ref_channels, setups):

                this_analised_channels = setup['num_analised_channels']
                this_corr_matrix = setup['corr_matrix']

                this_corr_matrix = this_corr_matrix[:,
                                                    this_ssi_ref_channels,:]
                # for each setup the order of the reference channels must be equal with respect to their DOFs
                #     we need a list of (local) ref_channels that corresponds to the ref_channels' DOFs of the first setup
                #     ssi_ref_channels = [[ref_DOF_1 -> index of ref_channel setup A, ref_DFO_2 -> index of ref_channel setup A, ...],
                #                         [ref_DOF_1 -> index of ref_channel setup B, ref_DFO_2 -> index of ref_channel setup B, ...],
                #                         ...]
                # this_corr_matrix = this_corr_matrix[:,this_ssi_ref_channels]
                # # just reorders the columns corresponding to the ref_channels
                # of current setup

                this_corr_matrix = this_corr_matrix.reshape(
                    (this_analised_channels, num_ref_channels * m_lags), order='F')
                this_block_column = this_corr_matrix[:, block_row * num_ref_channels:(
                    num_block_columns + block_row) * num_ref_channels]

                begin_row = block_row * num_analised_channels + sum_analised_channels
                if end_row is not None:
                    assert begin_row >= end_row
                end_row = begin_row + this_analised_channels

                subspace_matrix[begin_row:end_row,:] = this_block_column

                sum_analised_channels += this_analised_channels

            # block_row    0                                            1
            # setup 0: row 0*this_analised_channels ... 1*this_analised_channels,   3*this_analised_channels ... 4*this_analised_channels
            # setup 1: row 1*this_analised_channels ... 2*this_analised_channels,   4*this_analised_channels ... 5*this_analised_channels
            # setup 2: row 2*this_analised_channels ... 3*this_analised_channels,   5*this_analised_channels ... 6*this_analised_channels
            # (bc*num_setups+setup)*num_ref_channels
        assert (subspace_matrix != 0).all()

        U, S, V_T = scipy.linalg.svd(subspace_matrix, 1)

        self.U = U
        self.S = S
        self.V_T = V_T

        self.max_model_order = S.shape[0]
        # self.subspace_matrix = subspace_matrix
        self.num_block_rows = num_block_rows
        self.num_block_columns = num_block_columns

        self.state[0] = True

    def compute_modal_params(self, max_model_order=None,
                             max_modes=None, algo='svd'):
        super().compute_modal_params(max_model_order, max_modes, algo, modal_contrib=False)
        self.mode_shapes = self.mode_shapes[:self.merged_num_channels,:,:]

    def modal_analysis(self, A, C,):
        return super().modal_analysis(A, C, rescale_fun=self.rescale_by_references)

    def rescale_by_references(self, mode_shape):
        '''
        This is PoGer Rescaling

         
         * extracts each setup's reference and roving parts of the modeshape
         * compute rescaling factor from all setup's reference channels using a least-squares approach 
         * rescales each setup's roving channels and assembles final modeshape vector

        reference channel_pairs and final channel-dof-assignments have been determined by function pair_channels
        note: reference channels for SSI need not necessarily be reference channels for rescaling and vice versa

        :math:`S_\\phi \\times \\alpha = [n \\times 1, 0 .. 0]`

        :math:`\\phi^{ref}_i` : Reference-sensor part of modeshape estimated from setup :math:`i = 0 .. n`
        :matH:`j_{max} = \\operatorname{argmax}(\\Pi_i |\\phi^{ref}_i|)` : maximal modal component in all setups → will be approximately scaled to 1, must belong to the same sensor in each setup

        .. math::

            S_\\phi =  \\begin{bmatrix}
            \\phi^{ref}_{0,j_{max}}&  \\phi^{ref}_{1,j_{max}}& ..&            ..&               \\phi^{ref}_{n,j_{max}} \\\\
            \\phi^{ref}_0&            -\\phi^{ref}_1&          0&             ..&               0                      \\\\
            \\phi^{ref}_0&            0&                      -\\phi^{ref}_2& ..&               0                      \\\\
            .                        &.                       &.             & .             & .                      \\\\
            .                        &.                       &.             & .             & .                      \\\\
            \\phi^{ref}_0&            0&                      0&             ..&               -\\phi^{ref}_n          \\\\
            0&                       \\phi^{ref}_1&           -\\phi^{ref}_2& ..&               0                      \\\\
            .                        &.                      & .             & .             & .                      \\\\
            .                        &.                      & .             & .             & .                      \\\\
            0&                       \\phi^{ref}_1&           0&             ..&               -\\phi^{ref}_n          \\\\
            .                        &.                      & .             & .             & .                      \\\\
            .                        &.                      & .             & .             & .                      \\\\
            0&                       0&                      \\phi^{ref}_2&  ..&               -\\phi^{ref}_n          \\\\
            .                        &.                      & .             & .             & .                      \\\\
            .                        &.                      & .             & .            &  .                      \\\\
            0&                       0&                      0&             \\phi^{ref}_{n-1}& -\\phi^{ref}_n
            \\end{bmatrix}

        if references are the same in all setups

        dimensions :math:`= 1 + (n_{setups} ! )* n_{ref_{channels}} \\times n_{setups}`

        not quite exact, since different setups may share different references

        → list based assembly of the :math:`S_\\phi` matrix

        '''

        new_mode_shape = np.zeros((self.merged_num_channels), dtype=complex)

        start_row_scaled = 0
        end_row_scaled = self.setups[0]['num_analised_channels']

        new_mode_shape[start_row_scaled:end_row_scaled] = mode_shape[start_row_scaled:end_row_scaled]

        num_setups = len(self.setups)

        S_phi = []

        # assemble the first line
        all_ref_modes = []
        for setup_num, setup in enumerate(self.setups):
            this_refs = self.rescale_ref_channels[setup_num]
            mode_refs_this = mode_shape[this_refs]
            all_ref_modes.append(mode_refs_this)
        all_ref_modes = np.array(all_ref_modes).T
        max_ind = np.argmax(np.product(np.abs(all_ref_modes), axis=1))
        S_phi.append(all_ref_modes[max_ind:max_ind + 1,:])

        # assemble S_phi
        row_unscaled_1 = 0
        for setup_num_1, setup_1 in enumerate(self.setups):
            row_unscaled_2 = 0
            for setup_num_2, setup_2 in enumerate(self.setups):
                if setup_num_2 <= setup_num_1:

                    row_unscaled_2 += setup_2['num_analised_channels']
                    continue
                # ssi_ref_channels is ref_channels with respect to setup not to
                # merged mode shape

                base_refs = self.rescale_ref_channels[setup_num_1]
                this_refs = self.rescale_ref_channels[setup_num_2]

                base_refs = [int(ref + row_unscaled_1) for ref in base_refs]
                this_refs = [int(ref + row_unscaled_2) for ref in this_refs]

                mode_refs_base = mode_shape[base_refs]
                mode_refs_this = mode_shape[this_refs]

                this_S_phi = np.zeros(
                    (len(base_refs), num_setups), dtype=complex)
                this_S_phi[:, setup_num_1] = mode_refs_base
                this_S_phi[:, setup_num_2] = -1 * mode_refs_this

                S_phi.append(this_S_phi)
                row_unscaled_2 += setup_2['num_analised_channels']
            row_unscaled_1 += setup_1['num_analised_channels']

        S_phi = np.vstack(S_phi)

        # compute scaling factors
        rhs = np.zeros(S_phi.shape[0], dtype=complex)
        rhs[0] = num_setups + 0j
        alpha = np.linalg.pinv(S_phi).dot(rhs)

        end_row_scaled = 0  # self.setups[0]['num_analised_channels']
        row_unscaled = 0  # self.setups[0]['num_analised_channels']
        for setup_num, setup in enumerate(self.setups):

            this_refs = self.rescale_ref_channels[setup_num]
            # setup['roving_channels']+setup['ref_channels']
            this_all = range(setup['num_analised_channels'])
            this_rovs = list(set(this_all).difference(this_refs))

            this_rovs = [rov + row_unscaled for rov in this_rovs]

            mode_rovs_this = mode_shape[this_rovs]

            scale_fact = alpha[setup_num]

            if 0:
                base_refs = self.rescale_ref_channels[0]

                this_refs = [int(ref + row_unscaled) for ref in this_refs]

                mode_refs_base = mode_shape[base_refs]
                mode_refs_this = mode_shape[this_refs]
                mode_refs_this_conj = mode_refs_this.conj()

                numer = np.inner(mode_refs_this_conj, mode_refs_base)
                denom = np.inner(mode_refs_this_conj, mode_refs_this)
                scale_fact = numer / denom

            if setup_num == 0:
                mode_refs_this = mode_shape[this_refs]
                start_row_scaled = end_row_scaled
                end_row_scaled += len(this_refs)
                new_mode_shape[start_row_scaled:end_row_scaled] = scale_fact * \
                    mode_refs_this

            start_row_scaled = end_row_scaled
            end_row_scaled += len(this_rovs)

            new_mode_shape[start_row_scaled:end_row_scaled] = scale_fact * \
                mode_rovs_this

            row_unscaled += setup['num_analised_channels']
        return new_mode_shape

    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name

        if self.state[3]:  # add_setup
            out_dict['self.setups'] = self.setups
            out_dict['self.sampling_rate'] = self.sampling_rate
            out_dict['self.num_ref_channels'] = self.num_ref_channels
            out_dict['self.m_lags'] = self.m_lags
        if self.state[1]:  # pair_channels
            out_dict['self.ssi_ref_channels'] = self.ssi_ref_channels
            out_dict['self.rescale_ref_channels'] = self.rescale_ref_channels
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.merged_accel_channels'] = self.merged_accel_channels
            out_dict['self.merged_velo_channels'] = self.merged_velo_channels
            out_dict['self.merged_disp_channels'] = self.merged_disp_channels
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.num_analised_channels'] = self.num_analised_channels
            out_dict['self.start_time'] = self.start_time
        if self.state[0]:  # build_merged_subspace_matrix
            # out_dict['self.subspace_matrix'] = self.subspace_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
            out_dict['self.U'] = self.U
            out_dict['self.S'] = self.S
            out_dict['self.V_T'] = self.V_T
            out_dict['self.max_model_order'] = self.max_model_order
        if self.state[2]:  # compute_modal_params
            out_dict['self.eigenvalues'] = self.eigenvalues
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.mode_shapes'] = self.mode_shapes

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname,):
        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            raise RuntimeError('The result file is missing required components (self.state)')

        for this_state, state_string in zip(state, ['Setups added',
                                                    'Channels paired, channel-DOF assignments generated',
                                                    'Subspace matrix built',
                                                    'State matrices computed',
                                                    'Modal parameters computed',
                                                    ]):
            if this_state:
                logger.info(state_string)

        setup_name = str(in_dict['self.setup_name'].item())

        ssi_object = cls()
        ssi_object.setup_name = setup_name

        ssi_object.state = state
        # debug_here
        if state[3]:  # add_setup
            ssi_object.setups = validate_array(in_dict['self.setups'])
            ssi_object.sampling_rate = validate_array(in_dict['self.sampling_rate'])
            ssi_object.num_ref_channels = validate_array(in_dict['self.num_ref_channels'])
            ssi_object.m_lags = validate_array(in_dict['self.m_lags'])
        if state[1]:  # pair_channels
            ssi_object.ssi_ref_channels = validate_array(in_dict['self.ssi_ref_channels'])
            ssi_object.rescale_ref_channels = validate_array(in_dict['self.rescale_ref_channels'])
            ssi_object.merged_chan_dofs = [[int(float(cd[0])), str(cd[1]), float(cd[2]), float(
                cd[3]), str(cd[4] if len(cd) == 5 else '')] for cd in in_dict['self.merged_chan_dofs']]
            ssi_object.merged_accel_channels = validate_array(in_dict['self.merged_accel_channels'])
            ssi_object.merged_velo_channels = validate_array(in_dict['self.merged_velo_channels'])
            ssi_object.merged_disp_channels = validate_array(in_dict['self.merged_disp_channels'])
            ssi_object.merged_num_channels = validate_array(in_dict['self.merged_num_channels'])
            ssi_object.num_analised_channels = validate_array(in_dict['self.num_analised_channels'])
            ssi_object.start_time = validate_array(in_dict['self.start_time'])
        if state[0]:  # build_merged_subspace_matrix
            # ssi_object.subspace_matrix = validate_array(in_dict['self.subspace_matrix'])
            ssi_object.num_block_columns = validate_array(in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = validate_array(in_dict['self.num_block_rows'])
            ssi_object.U = validate_array(in_dict['self.U'])
            ssi_object.S = validate_array(in_dict['self.S'])
            ssi_object.V_T = validate_array(in_dict.get('self.V_T', None))
            ssi_object.max_model_order = validate_array(in_dict['self.max_model_order'])
        if state[2]:  # compute_modal_params
            ssi_object.eigenvalues = validate_array(in_dict['self.eigenvalues'])
            ssi_object.modal_damping = validate_array(in_dict['self.modal_damping'])
            ssi_object.modal_frequencies = validate_array(in_dict['self.modal_frequencies'])
            ssi_object.mode_shapes = validate_array(in_dict['self.mode_shapes'])

        return ssi_object


if __name__ == '__main__':
    pass
