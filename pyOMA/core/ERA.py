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
import numpy as np
import scipy.linalg
from collections import deque
import os
from .PreProcessingTools import PreProcessSignals

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ERA(object):

    def __init__(self, prep_signals):
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_signals, PreProcessSignals)
        self.prep_signals = prep_signals
        self.setup_name = prep_signals.setup_name
        self.start_time = prep_signals.start_time
        #             0         1           2
        # self.state= [SHankelMatrix, State Mat., Modal Par.
        self.state = [False, False, False]

        self.num_block_columns = None
        self.num_block_rows = None
        self.toeplitz_matrix = None
        self.hankel_matrix = None  # anil

        self.max_model_order = None
        self.state_matrix = None
        self.output_matrix = None

        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None

    def CalculateFRF(self):
        '''
        function by anil
        FRF(Frequency response function) is convertion of signal from time to frequency domain.
        The following function performs this conversion.
        '''
        measurement = self.prep_signals.signals
        num_channels = measurement.shape[1]
        num_time_steps = self.prep_signals.F.shape[0]
        acceleration_fft = np.zeros(
            (num_time_steps // 2 + 1, num_channels), dtype=complex)

        F_fft = np.fft.rfft(np.hamming(num_time_steps) * self.prep_signals.F)

        for channel in range(num_channels):  # loop over channels
            fft_this_channel = np.fft.rfft(np.hamming(
                num_time_steps) * measurement[:, channel])
            acceleration_fft[:, channel] = fft_this_channel

        FRF = np.zeros_like(acceleration_fft)

        for channel in range(num_channels):
            FRF[:, channel] = acceleration_fft[:, channel] / F_fft

        IRF = np.zeros((num_time_steps, num_channels))

        for channel in range(num_channels):  # loop over channels
            ifft_this_channel = np.fft.irfft(FRF[:, channel])

            IRF[:, channel] = ifft_this_channel

        self.IFRF = IRF.T

    def build_hankel_matrix(self, num_block_columns):
        '''
        author: Anil
        Constructs a shifted hankel matrix.
        '''

        IRFT = self.IFRF
        num_channels = self.prep_signals.num_analised_channels
        num_block_rows = num_block_columns + 1

        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows

        Hankel_matrix = np.zeros(
            (num_channels *
             num_block_rows,
             num_block_columns),
            dtype=complex)
        for i in range(0, num_block_rows):
            j = i + 1
            this_block = IRFT[0:num_channels, j:(num_block_columns + j)]
            begin_row = i * num_channels
            Hankel_matrix[begin_row:(
                begin_row + num_channels), 0:num_block_columns] = this_block

        self.hankel_matrix = Hankel_matrix
        self.state[0] = True

    def compute_state_matrices(self, max_model_order=None):
        '''

        '''
        if max_model_order is not None:
            assert isinstance(max_model_order, int)

        assert self.state[0]

        hankel_matrix = self.hankel_matrix  # anil
        num_channels = self.prep_signals.num_analised_channels
        # num_block_columns = self.num_block_columns
        # num_block_rows = self.num_block_rows
        print('Computing state matrices...')

        [U, S, _] = np.linalg.svd(hankel_matrix, 0)  # anil

        # anil
        S1 = np.diag(S)
        S_sqrt = np.sqrt(S1)
        p1 = np.dot(U, S_sqrt)
        # p2=np.dot(S_sqrt,V_T)

        # A=np.dot(np.linalg.pinv(p1), hankel_matrix, np.linalg.pinv(p2))
        # A=A.real
        C = p1[:num_channels,:]
        # C=C.real
        # p1=p1.real

        self.Oi = p1
        # self.state_matrix = A
        self.output_matrix = C
        self.max_model_order = max_model_order

        self.state[1] = True
        self.state[2] = False  # previous modal params are invalid now

    def compute_modal_params(self, max_model_order=None):

        if max_model_order is not None:
            assert max_model_order <= self.max_model_order
            self.max_model_order = max_model_order

        assert self.state[1]
        print('Computing modal parameters...')
        max_model_order = self.max_model_order
        num_analised_channels = self.prep_signals.num_analised_channels
        num_block_rows = self.num_block_rows
        # state_matrix = self.state_matrix
        Oi = self.Oi
        output_matrix = self.output_matrix
        sampling_rate = self.prep_signals.sampling_rate

        modal_frequencies = np.zeros((max_model_order, max_model_order))
        modal_damping = np.zeros((max_model_order, max_model_order))
        eigenvalues = np.zeros(
            (max_model_order, max_model_order), dtype=complex)
        mode_shapes = np.zeros(
            (num_analised_channels,
             max_model_order,
             max_model_order),
            dtype=complex)

        for order in range(1, max_model_order, 1):

            Oi0 = Oi[:(num_analised_channels * (num_block_rows - 1)),:order]
            Oi1 = Oi[num_analised_channels:(
                num_analised_channels * num_block_rows),:order]

            a = np.dot(np.linalg.pinv(Oi0), Oi1)
            eigenvalues_paired, _, eigenvectors_paired = scipy.linalg.eig(
                a=a[0:order, 0:order], b=None, left=True, right=True)

            eigenvalues_single, eigenvectors_single = self.remove_conjugates_new(
                eigenvalues_paired, eigenvectors_paired)

            for index, k in enumerate(eigenvalues_single):

                lambda_k = np.log(complex(k)) * sampling_rate
                freq_j = np.abs(lambda_k) / (2 * np.pi)
                damping_j = np.real(lambda_k) / np.abs(lambda_k) * (-100)
                mode_shapes_j = np.dot(
                    output_matrix[:, 0:order], eigenvectors_single[:, index])

                modal_frequencies[order, index] = freq_j
                modal_damping[order, index] = damping_j
                eigenvalues[order, index] = k
                mode_shapes[:, index, order] = mode_shapes_j

        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.eigenvalues = eigenvalues

        self.state[2] = True

    @staticmethod
    def remove_conjugates_new(eigval, eigvec_r, eigvec_l=None):
        '''
        removes conjugates

        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''
        # return vectors, eigval
        num_val = len(eigval)
        conj_indices = deque()

        for i in range(num_val):
            this_val = eigval[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val:  # remove real eigvals
                conj_indices.append(i)
            for j in range(
                    i + 1, num_val):  # catches unordered conjugates but takes slightly longer
                if eigval[j] == this_conj_val:

                    # if not np.allclose(eigvec_l[j],eigvec_l[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_l is not')
                    #    continue

                    # if not np.allclose(eigvec_r[j],eigvec_r[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_r is not')
                    #    continue

                    conj_indices.append(j)
                    break

        # print('indices of complex conjugate: {}'.format(conj_indices))
        conj_indices = list(set(range(num_val)).difference(conj_indices))
        # print('indices to keep and return: {}'.format(conj_indices))

        if eigvec_l is None:

            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_r

        else:
            eigvec_l = eigvec_l[:, conj_indices]
            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_l, eigvec_r

    def save_state(self, fname):

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        #             0         1           2
        # self.state= [SHankelMatrix, State Mat., Modal Par.]
        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time
        # out_dict['self.prep_signals']=self.prep_signals
        if self.state[0]:  # SHankelMatrix
            # out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.hankel_matrix'] = self.hankel_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:  # state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[2]:  # modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.eigenvalues'] = self.eigenvalues

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname)
        #             0         1           2
        # self.state= [SHankelMatrix, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Shifted Hankel Matrices Built',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state:
                print(state_string)

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        # start_time = in_dict['self.start_time'].item()
        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        # prep_signals = in_dict['self.prep_signals'].item()
        ssi_object = cls(prep_signals)
        ssi_object.state = state
        if state[0]:  # SHankelMatrix
            ssi_object.hankel_matrix = in_dict['self.hankel_matrix']
            ssi_object.num_block_columns = int(
                in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:  # state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix = in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
        if state[2]:  # modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
            ssi_object.eigenvalues = in_dict['self.eigenvalues']

        return ssi_object

    @staticmethod
    def rescale_mode_shape(modeshape):
        # scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape


if __name__ == '__main__':
    pass
