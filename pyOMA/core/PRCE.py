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
        if not isinstance(num_corr_samples, int):
            raise TypeError(
                f"num_corr_samples must be int, got {type(num_corr_samples).__name__!r}"
            )

        self.num_corr_samples = num_corr_samples
        self.prep_signals.correlation(2 * num_corr_samples + 1)

        self.x_corr_Tensor = np.transpose(
            self.prep_signals.corr_matrix, [
                1, 0, 2])  # x_corr_Tensor
        self.state[0] = True


    def compute_modal_params(self, max_model_order):
        """Compute modal parameters for all model orders up to *max_model_order*."""
        if not isinstance(max_model_order, int):
            raise TypeError(
                f"max_model_order must be int, got {type(max_model_order).__name__!r}"
            )
        self.max_model_order = max_model_order
        if not self.state[0]:
            raise RuntimeError("Call build_corr_tensor() first.")

        logger.info('Computing modal parameters...')
        num_ch = self.prep_signals.num_analised_channels
        num_ref = self.prep_signals.num_ref_channels
        sr = self.prep_signals.sampling_rate
        n_cols = int(num_ref * max_model_order / 2)

        modal_frequencies = np.zeros((max_model_order, n_cols))
        modal_damping = np.zeros((max_model_order, n_cols))
        mode_shapes = np.ones((num_ch, n_cols, max_model_order), dtype=complex)

        printsteps = list(np.linspace(0, max_model_order, 100, dtype=int))
        for order in range(1, max_model_order + 1):
            while order in printsteps:
                del printsteps[0]
                print('.', end='', flush=True)
            self._compute_one_order(
                order, num_ref, num_ch, sr,
                modal_frequencies, modal_damping, mode_shapes)

        print('.', end='\n', flush=True)
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.state[1] = True

    def _compute_one_order(self, order, num_ref, num_ch, sr,
                           modal_freq, modal_damp, mode_shapes):
        """Run PRCE computation for a single *order* and store results in-place."""
        x_corr = self.x_corr_Tensor
        num_corr = self.num_corr_samples
        LHS, RHS = self._build_lhs_rhs(order, num_ref, num_ch, num_corr, x_corr)
        B_matrix = np.dot(np.dot(RHS, LHS.T), np.linalg.inv(np.dot(LHS, LHS.T)))
        companion = self._build_companion(order, num_ref, B_matrix)
        mu_vect, eigenvectors = np.linalg.eig(companion)
        W_matrix = eigenvectors[(order - 1) * num_ref:order * num_ref, :]
        W_Lambda = self._build_w_lambda(order, num_ref, W_matrix, mu_vect)
        H_j = self._build_h_j(order, num_ref, num_ch, x_corr)
        W_herm = np.conj(W_Lambda).T
        A_j1 = np.dot(np.dot(np.linalg.inv(np.dot(W_herm, W_Lambda)), W_herm), H_j)
        psi = self._build_psi(order, num_ref, num_ch, A_j1)
        eig_s, vec_s = self.remove_conjugates(mu_vect, psi)
        self._store_modes(order - 1, eig_s, vec_s, sr, modal_freq, modal_damp, mode_shapes)

    @staticmethod
    def _build_lhs_rhs(order, num_ref, num_ch, num_corr, x_corr):
        """Build left-hand-side and right-hand-side Hankel matrices."""
        rows = num_ref * order
        cols = num_ch * num_corr
        LHS = np.zeros((rows, cols))
        RHS = np.zeros((num_ref, cols))
        for jj in range(num_ch):
            for row_idx in range(order):
                block = x_corr[:, jj, row_idx:(row_idx + num_corr)]
                LHS[row_idx * num_ref:(row_idx + 1) * num_ref,
                    jj * num_corr:(jj + 1) * num_corr] = block
            rhs_block = x_corr[:, jj, order:(order + num_corr)]
            RHS[:, jj * num_corr:(jj + 1) * num_corr] = -rhs_block
        return LHS, RHS

    @staticmethod
    def _build_companion(order, num_ref, B_matrix):
        """Build the companion matrix from beta coefficients."""
        size = order * num_ref
        companion = np.zeros((size, size))
        for ii in range(order):
            beta = B_matrix[:, (order - (ii + 1)) * num_ref:(order - ii) * num_ref]
            companion[:num_ref, ii * num_ref:(ii + 1) * num_ref] = -beta
        if order > 1:
            companion[num_ref:size, :(order - 1) * num_ref] = np.identity((order - 1) * num_ref)
        return companion

    @staticmethod
    def _build_w_lambda(order, num_ref, W_matrix, mu_vect):
        """Build the W-Lambda Vandermonde-like matrix."""
        Lambda = np.diag(mu_vect)
        W_Lambda = np.zeros(((order + 1) * num_ref, order * num_ref), dtype=complex)
        for ii in range(order + 1):
            W_Lambda[ii * num_ref:(ii + 1) * num_ref, :] = np.dot(W_matrix, Lambda ** ii)
        return W_Lambda

    @staticmethod
    def _build_h_j(order, num_ref, num_ch, x_corr):
        """Build the H_j correlation matrix."""
        H_j = np.zeros(((order + 1) * num_ref, num_ch))
        for jj in range(num_ch):
            for ii in range(order + 1):
                H_j[ii * num_ref:(ii + 1) * num_ref, jj] = x_corr[:, jj, ii]
        return H_j

    @staticmethod
    def _build_psi(order, num_ref, num_ch, A_j1):
        """Compute the mode-shape matrix from residuals."""
        psi = np.zeros((num_ch, order * num_ref), dtype=complex)
        psi[0, :] = np.sqrt(A_j1[:, 0])
        other = A_j1[:, 1:num_ch].copy()
        for r in range(2 * order):
            other[r, :] = other[r, :] / psi[0, r]
        psi[1:num_ch, :] = other.T
        return psi

    @staticmethod
    def _store_modes(order_idx, eig_s, vec_s, sr, modal_freq, modal_damp, mode_shapes):
        """Store eigenvalues/vectors at *order_idx* into the output arrays."""
        for idx, k in enumerate(eig_s):
            lambda_k = np.log(complex(k)) * sr
            modal_freq[order_idx, idx] = np.abs(lambda_k) / (2 * np.pi)
            modal_damp[order_idx, idx] = np.real(lambda_k) / np.abs(lambda_k) * (-100)
            mode_shapes[:, idx, order_idx] = vec_s[:, idx]

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

        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be PreProcessSignals, got {type(prep_signals).__name__!r}"
            )
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
