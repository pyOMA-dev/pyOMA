# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Poly-reference Complex Exponential (PRCE) identification method."""

import numpy as np
import os

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .Helpers import ConfigFile
# from StabilDiagram import main_stabil, StabilPlot, nearly_equal

# import pydevd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class PRCE(ModalBase):
    """Poly-reference Complex Exponential (PRCE) identification method.

    Identifies modal parameters from a 3-D tensor of cross-correlation
    functions using the Complex Exponential approach.  The standard workflow is:

    1. :meth:`build_corr_tensor` — assemble the correlation tensor.
    2. :meth:`compute_modal_params` — run the multi-order identification.
    3. Pass the result to :class:`~pyOMA.core.StabilDiagram.StabilCalc` for
       stabilisation-diagram analysis.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        Pre-processed signal object.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs
            Passed to :class:`~pyOMA.core.ModalBase.ModalBase`.
        """
        super().__init__(*args, **kwargs)

        #             0         1
        # self.state= [Corr. Tensor, Modal Par.
        self.state = [False, False]

        self.num_corr_samples = None
        self.x_corr_Tensor = None

    @classmethod
    def init_from_config(cls, mod_ID_file, prep_signals):
        cfg = ConfigFile(mod_ID_file)
        num_corr_samples = cfg.int('Number of Correlation Samples')
        max_model_order = cfg.int('Maximum Model Order')

        prce_object = cls(prep_signals)
        logger.debug('num_corr_samples=%s, max_model_order=%s', num_corr_samples, max_model_order)
        prce_object.build_corr_tensor(num_corr_samples)
        prce_object.compute_modal_params(max_model_order)

        return prce_object

    def build_corr_tensor(self, num_corr_samples):
        '''
        Builds a 3D Tensor of cross correlation functions with the following directions:
        1 - related to reference channels
        2 - all channels
        3 - time
        '''
        assert isinstance(num_corr_samples, int)

        self.num_corr_samples = num_corr_samples
        self.prep_signals.correlation(2 * num_corr_samples + 1)

        self.x_corr_Tensor = np.transpose(
            self.prep_signals.corr_matrix, [
                1, 0, 2])  # x_corr_Tensor
        self.state[0] = True


    def compute_modal_params(self, max_model_order):

        # if max_model_order is not None:
        #    assert max_model_order<=self.max_model_order
        #    self.max_model_order=max_model_order

        assert isinstance(max_model_order, int)
        self.max_model_order = max_model_order

        assert self.state[0]
        x_corr_Tensor = self.x_corr_Tensor

        logger.info('Computing modal parameters...')
        max_model_order = self.max_model_order
        num_corr_samples = self.num_corr_samples
        # state_matrix = self.state_matrix
        # output_matrix = self.output_matrix
        sampling_rate = self.prep_signals.sampling_rate
        # List of ref. channel numbers
        # ref_channels = sorted(self.prep_signals.ref_channels)

        num_analised_channels = self.prep_signals.num_analised_channels
        num_ref_channels = self.prep_signals.num_ref_channels

#         all_channels = ref_channels + roving_channels
#         all_channels.sort()

        # Compute the modal solutions for all model orders

        modal_frequencies = np.zeros(
            (max_model_order, int(num_ref_channels * max_model_order / 2)))
        modal_damping = np.zeros((max_model_order,
                                  int(num_ref_channels * max_model_order / 2)))
        mode_shapes = np.ones((num_analised_channels, int(
            num_ref_channels * max_model_order / 2), max_model_order), dtype=complex)

        # print("size of modal_frequencies = ", np.shape(modal_frequencies))

        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for this_model_order in range(1, max_model_order + 1):
            while this_model_order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)

            # Prepare l.h.s. matrix and r.h.s. vector for correlation functions
            # #

            rows_system = num_ref_channels * this_model_order
            cols_system = num_analised_channels * num_corr_samples

            LHS_matrix = np.zeros((rows_system, cols_system))
            RHS_matrix = np.zeros((num_ref_channels, cols_system))

            # Construct Hankel matrices for l.h.s. and r.h.s.

            for jj in range(num_analised_channels):
                for row_index in range(this_model_order):
                    this_blockrow = x_corr_Tensor[:, jj, row_index:(
                        row_index + num_corr_samples)]
                    LHS_matrix[row_index *
                               num_ref_channels:(row_index +
                                                 1) *
                               num_ref_channels, jj *
                               num_corr_samples:(jj +
                                                 1) *
                               num_corr_samples] = this_blockrow

                this_RHS = x_corr_Tensor[:, jj, this_model_order:(
                    this_model_order + num_corr_samples)]

                RHS_matrix[:, jj * \
                    num_corr_samples:(jj + 1) * num_corr_samples] = -this_RHS

            # Solve system of equations for beta values

            LHS_inv = np.linalg.inv(np.dot(LHS_matrix, LHS_matrix.T))
            RHS_LHS_t = np.dot(RHS_matrix, LHS_matrix.T)
            B_matrix = np.dot(RHS_LHS_t, LHS_inv)

            # Compute complex eigenvalues

            companion_matrix = np.zeros(
                (this_model_order * num_ref_channels,
                 this_model_order * num_ref_channels))

            for ii in range(this_model_order):
                beta_tmp = B_matrix[:, (this_model_order - (ii + 1)) *
                                    num_ref_channels:(this_model_order - ii) * num_ref_channels]
                companion_matrix[0:num_ref_channels, ii *
                                 num_ref_channels: (ii + 1) * num_ref_channels] = -beta_tmp

            companion_matrix[num_ref_channels:this_model_order *
                             num_ref_channels, 0:(this_model_order -
                                                  1) *
                             num_ref_channels] = np.identity((this_model_order -
                                                              1) *
                                                             num_ref_channels)

            mu_vect, eigenvectors = np.linalg.eig(companion_matrix)
            # print("mu_vect: ", mu_vect)

            # Compute residue

            W_matrix = eigenvectors[(this_model_order -
                                     1) *
                                    num_ref_channels:this_model_order *
                                    num_ref_channels,:]
            Lambda_matrix = np.diag(mu_vect)

            W_Lambda_matrix = np.zeros(
                ((this_model_order + 1) * num_ref_channels,
                 this_model_order * num_ref_channels),
                dtype=complex)
            # W_Lambda_matrix = np.zeros(((this_model_order)*num_ref_channels, 2* this_model_order), dtype=complex)

            # print("W_Lambda_matrix = ", W_Lambda_matrix)
            # print("W_matrix = ", W_matrix)

            for ii in range(this_model_order + 1):

                Lambda_pow_matrix = Lambda_matrix ** ii

                # print("Lambda_pow_matrix = ", Lambda_pow_matrix)

                W_Lambda_matrix[ii *
                                num_ref_channels:(ii +
                                                  1) *
                                num_ref_channels,:] = np.dot(W_matrix, Lambda_pow_matrix)

            H_j_matrix = np.zeros(
                ((this_model_order + 1) * num_ref_channels,
                 num_analised_channels))

            for jj in range(num_analised_channels):
                for ii in range(this_model_order + 1):

                    H_j_matrix[ii *
                               num_ref_channels:(ii +
                                                 1) *
                               num_ref_channels, jj] = x_corr_Tensor[:, jj, ii]

            W_Lambda_herm = np.conj(W_Lambda_matrix).T
            tmp_1 = np.linalg.inv(np.dot(W_Lambda_herm, W_Lambda_matrix))
            tmp_2 = np.dot(tmp_1, W_Lambda_herm)
            A_j1_matrix = np.dot(tmp_2, H_j_matrix)

            # Compute eigenvectors from residuals

            # step 1: set scaling factors Q_r=1 for all modes r
            # all modal components of dof 1 are obtained as sqrt of the first
            # column of A_j1_matrix

            psi_matrix = np.zeros(
                (num_analised_channels,
                 this_model_order *
                 num_ref_channels),
                dtype=complex)

            psi_matrix[0,:] = np.sqrt(A_j1_matrix[:, 0])

            # step 2: obtain all other modal components
            # by dividing the respective residuals by the first modal component

            other_psi = A_j1_matrix[:, 1:num_analised_channels]

            for r in range(2 * this_model_order):

                other_psi[r,:] = other_psi[r,:] / psi_matrix[0, r]

            # psi_matrix[1:2*this_model_order,:] = other_psi.T
            psi_matrix[1:num_analised_channels,:] = other_psi.T

            # Remove complex conjugate solutions and compute nat. frequencies +
            # modal damping

            eigenvalues_single, eigenvectors_single = \
                self.remove_conjugates(mu_vect, psi_matrix)

            for index, k in enumerate(eigenvalues_single):
                lambda_k = np.log(complex(k)) * sampling_rate
                freq_j = np.abs(lambda_k) / (2 * np.pi)
                damping_j = np.real(lambda_k) / np.abs(lambda_k) * (-100)
                # mode_shapes_j = np.dot(output_matrix[:, 0:order + 1], eigenvectors_single[:,index])

                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                # mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))

                # mode_shapes_j*=self.prep_signals.channel_factors

                modal_frequencies[(this_model_order - 1), index] = freq_j
                modal_damping[(this_model_order - 1), index] = damping_j
                # mode_shapes[:,index,this_model_order]=mode_shapes_j
                mode_shapes[:, index, (this_model_order - 1)
                            ] = eigenvectors_single[:, index]

        print('.', end='\n', flush=True)

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes

        self.state[1] = True

        '''
                lambda_vect = np.log(mu_vect) * sampling_rate

                lambda_vect_filt = np.zeros((1,max_model_order), dtype = complex)
                current_mode_shapes = np.zeros((num_analised_channels, max_model_order), dtype = complex)
                jj = 0

                for ii in range(len(lambda_vect)-1):

                    if lambda_vect[ii] == np.conj(lambda_vect[ii+1]):

                        lambda_vect_filt[0,jj] = lambda_vect[ii]
                        current_mode_shapes[:,jj] = psi_matrix[:,ii]

                        jj = jj + 1

                freq_vect = np.abs(lambda_vect_filt) / (2*np.pi)
                damping_vect = - np.real(lambda_vect_filt) / np.abs(lambda_vect_filt) * 100

                modal_frequencies[(this_model_order-1), :] = freq_vect
                modal_damping[(this_model_order-1), :] = damping_vect
                mode_shapes[:,:,(this_model_order-1)] = current_mode_shapes

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes

        self.state[1]=True
    '''

    def save_state(self, fname):

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0         1
        # self.state= [Corr. Tensor, Modal Par.
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        # out_dict['self.prep_signals']=self.prep_signals
        if self.state[0]:  # cross correlation tensor
            out_dict['self.x_corr_Tensor'] = self.x_corr_Tensor
        if self.state[1]:  # modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.max_model_order'] = self.max_model_order

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        #             0         1
        # self.state= [Corr. Tensor, Modal Par.
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Correlation Functions Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state:
                print(state_string)

        assert isinstance(prep_signals, PreProcessSignals)
        # setup_name = str(in_dict['self.setup_name'].item())
        # prep_signals = in_dict['self.prep_signals'].item()
        prce_object = cls(prep_signals)
        prce_object.state = state
        if state[0]:  # covariances
            prce_object.x_corr_Tensor = in_dict['self.x_corr_Tensor']
        if state[1]:  # modal params
            prce_object.modal_frequencies = in_dict['self.modal_frequencies']
            prce_object.modal_damping = in_dict['self.modal_damping']
            prce_object.mode_shapes = in_dict['self.mode_shapes']
            prce_object.max_model_order = int(in_dict['self.max_model_order'])

        return prce_object


def main():
    pass


if __name__ == '__main__':
    main()
