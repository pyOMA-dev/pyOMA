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

Written by Volkmar Zabel 2016,
refactored by Simon Marwitz 2021,
improved, corrected and refactored by Simon Marwitz 2024

.. TODO::
     * Test functions should be added to the test package

'''

import numpy as np
import os
import scipy.signal
import scipy.linalg
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .Helpers import validate_array, simplePbar



class PLSCF(ModalBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = [False, False]

        self.begin_frequency = None
        self.end_frequency = None
        self.nperseg = None
        self.factor_a = None
        self.selected_omega_vector = None
        self.pos_half_spectra = None
        
        self._lower_residuals = None
        self._upper_residuals = None
        self._mode_shapes_raw = None
        self._participation_vectors = None
        self._eigenvalues = None
        
        self._half_spec_synth = None
        self.modal_contributions = None

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        assert os.path.exists(conf_file)
        assert isinstance(prep_signals, PreProcessSignals)

        with open(conf_file, 'r') as f:

            assert f.__next__().strip('\n').strip(' ') == 'Begin Frequency:'
            begin_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'End Frequency:'
            end_frequency = float(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Samples per time segment:'
            nperseg = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ') == 'Maximum Model Order:'
            max_model_order = int(f. __next__().strip('\n'))

        pLSCF_object = cls(prep_signals)
        pLSCF_object.build_half_spectra(nperseg, begin_frequency, end_frequency)
        pLSCF_object.compute_modal_params(max_model_order)

        return pLSCF_object

    def build_half_spectra(self, nperseg=None, 
                           begin_frequency=None, end_frequency=None, 
                           window_decay=0.001, **kwargs):
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
                
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to prep_signals.correlation
                
        '''

        logger.info('Constructing half-spectrum matrix ... ')
        if begin_frequency is None or begin_frequency < 0.0:
            begin_frequency = 0.0
        if isinstance(begin_frequency, int): 
            begin_frequency = float(begin_frequency)
        assert isinstance(begin_frequency, float)
        if end_frequency is None or end_frequency > self.prep_signals.sampling_rate / 2:
            end_frequency = self.prep_signals.sampling_rate / 2
        if isinstance(end_frequency, int): 
            end_frequency = float(end_frequency)
        assert isinstance(end_frequency, float)
        if nperseg is None:
            # check if correlations have already been computed (welch, blackman-tukey)
            if self.prep_signals.m_lags is not None :
                _nperseg = self.prep_signals.m_lags
            # alternatively try if psds might be available (welch)
            elif self.prep_signals.n_lines is not None:
                _nperseg = self.prep_signals.n_lines // 2 + 1
            else:
                raise RuntimeError('Argument nperseg or precomputed spectra/correlations must be provided.')
        else:
            assert isinstance(nperseg, int)

        if self.prep_signals._last_meth == 'welch':
            logger.info("The selected spectral estimation method (Welch) is not recommended (applied window introduces damping bias).")
        correlation_matrix = self.prep_signals.correlation(nperseg, **kwargs)
        # calling correlation(nperseg=None) ensured using precomputed correlations
        if nperseg is None:
            nperseg = _nperseg
        
        tau = - nperseg / np.log(window_decay)
        win = scipy.signal.windows.get_window(('exponential', 0, tau), nperseg, fftbins=True)
        factor_a = - 1 / tau
        psd_matrix = np.fft.rfft(correlation_matrix * win)
        
        sampling_rate = self.prep_signals.sampling_rate
        freqs = np.fft.rfftfreq(nperseg, 1 / sampling_rate)
        freq_inds = (freqs > begin_frequency) & (freqs < end_frequency)
        selected_omega_vector = freqs[freq_inds] * 2 * np.pi
        spectrum_tensor = psd_matrix[..., freq_inds]
        
        self.begin_frequency = begin_frequency
        self.end_frequency = end_frequency
        self.nperseg = nperseg
        
        self.selected_omega_vector = selected_omega_vector
        self.pos_half_spectra = spectrum_tensor
        self.factor_a = factor_a
        
        self.state[0] = True
    
    @property
    def num_omega(self):
        return self.selected_omega_vector.shape[0]
    
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
        if order>self.nperseg - 1:
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
            dtype=complex
        else:
            dtype=float
        
        RS_solutions = np.zeros((order + 1, (order + 1) * n_r, n_l), dtype=dtype)
        M = np.zeros(((order + 1) * n_r, (order + 1) * n_r), dtype=dtype)
        
        # Create matrices X_0 and Y_0, Peeters 2004: Sect. 2.2ff
        # for channel-dependent weights, this has to move into the loop below
        X_o = np.exp(1j * selected_omega_vector[:, np.newaxis] * 
                     Delta_t * np.arange(order + 1)[np.newaxis, :]) # (num_omega, (order + 1))
        X_o_H = np.conj(X_o.T) # ((order + 1), num_omega)
        R_o = X_o_H @ X_o# ((order + 1),(order + 1))
        if not complex_coefficients: R_o = R_o.real 
        
        Y_o = np.empty((num_omega, ((order + 1) * n_r)), dtype=complex)
        for i_l in range(n_l):
            for kk in range(num_omega):
                Y_o[kk, :] = np.kron(-X_o[kk, :], pos_half_spectra[i_l, :, kk].T)
            
            S_o = X_o_H @ Y_o # ((order + 1),(order + 1) * n_r)
            if not complex_coefficients: S_o = S_o.real
            
            T_o = np.conj(Y_o.T) @ Y_o# ((order + 1) * n_r,(order + 1) * n_r)
            if not complex_coefficients: T_o = T_o.real 
            
            RS_solution = np.linalg.solve(R_o, S_o)
            
            M = M + (T_o - np.conj(S_o).T @ RS_solution)
            M *= 2
            
            RS_solutions[:, :, i_l] = RS_solution
        
        # Compute alpha and beta coefficients: Cauberghe 2004. Sec. 5.2.1
        M_aa = M[:order * n_r, :order * n_r]
        M_ab = M[:order * n_r, -n_r:]
        alpha_b = - np.linalg.solve(M_aa, M_ab)
        alpha = np.concatenate((alpha_b, np.eye(n_r)), axis=0) # ((order + 1) * n_r, n_r)
        
        beta_l_i = np.zeros(((order + 1), n_r, n_l), dtype=dtype)
        for i_l in range(n_l):
            RS_solution = RS_solutions[:, :, i_l]
            beta_l = - RS_solution @ alpha
            
            beta_l_i[:, :, i_l] = beta_l
        
        return alpha, beta_l_i 
    
    
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
        A_p = alpha[-n_r:, :]
        B_p = beta_l_i[order, :, :].T
        
        A_c = np.zeros((order * n_r, order * n_r), dtype=alpha.dtype)
        C_c = np.zeros((n_l, order * n_r), dtype=alpha.dtype)
        
        for p_i in range(order):
            A_p_i = alpha[(order - p_i - 1) * n_r:(order - p_i) * n_r, :]
            
            this_A_c_block = - np.linalg.solve(A_p, A_p_i)
            A_c[:n_r, p_i * n_r:(p_i + 1) * n_r] = this_A_c_block
            
            B_p_i = beta_l_i[order - p_i - 1, :, :].T
            
            this_C_c_block = B_p_i + (B_p @ this_A_c_block)
            C_c[:, p_i * n_r:(p_i + 1) * n_r] = this_C_c_block
        
        A_c_rest = np.eye((order - 1) * n_r)
        A_c[n_r:, :(order - 1) * n_r] = A_c_rest
        
        
        eigvals, eigvecs_r = np.linalg.eig(A_c)
        
        conj_indices = self.remove_conjugates(eigvals, eigvecs_r, inds_only=True)
        n_modes = len(conj_indices)
        
        modal_frequencies = np.zeros((n_modes,))
        modal_damping = np.zeros((n_modes, ))
        mode_shapes = np.zeros((n_l, n_modes), dtype=complex)
        eigenvalues = np.zeros((n_modes), dtype=complex)
        
        Phi = C_c @ eigvecs_r
        
        for i, ind in enumerate(reversed(conj_indices)):
            
            lambda_i = np.log(eigvals[ind]) * sampling_rate
            freq_i = np.abs(lambda_i) / (2 * np.pi)
            
            # damping without correction if no exponential window was applied
            if factor_a is None:
                damping_i = np.real(lambda_i)/np.abs(lambda_i) * (-100)
            # damping with correction if exponential window was applied to
            else:
                damping_i = (np.real(lambda_i) / np.abs(lambda_i) - factor_a * (sampling_rate) / (freq_i * 2 * np.pi)) * (-100)
        
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
        
        return modal_frequencies[argsort], modal_damping[argsort], mode_shapes[:,argsort], eigenvalues[argsort]
    
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
        accel_channels = self.prep_signals.accel_channels
        velo_channels = self.prep_signals.velo_channels
    
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
        factor_a = self.factor_a
        sampling_rate = self.prep_signals.sampling_rate
        
        order = alpha.shape[0] // n_r - 1
        
        # Create companion matrix
        A_p = alpha[-n_r:, :]
        A_c = np.zeros((order * n_r, order * n_r), dtype=alpha.dtype)
        
        if np.issubdtype(alpha.dtype, complex):
            logger.warning('Residual-based modal analysis with complex coefficients has not been verified.')
        
        for p_i in range(order):
            A_p_i = alpha[(order - p_i - 1) * n_r:(order - p_i) * n_r, :]
    
            this_A_c_block = - np.linalg.solve(A_p, A_p_i)
            A_c[p_i * n_r:(p_i + 1) * n_r, :n_r] = this_A_c_block
            
        A_c_rest = np.eye((order - 1) * n_r)
        A_c[:-n_r, n_r:] = A_c_rest
        
        eigvals, eigvecs_l = scipy.linalg.eig(A_c, left=True, right=False)
        
        eigvals, eigvecs_l = self.remove_conjugates(eigvals, eigvecs_l)

        _eigenvalues = np.log(eigvals) * sampling_rate
        _modal_frequencies = np.abs(_eigenvalues) / (2 * np.pi)
        
        # remove all frequencies outside the spectral frequency band
        inds = np.where((_modal_frequencies >= self.begin_frequency) & (_modal_frequencies <= self.end_frequency))[0]
        n_modes = len(inds)
        
        modal_damping = np.zeros((n_modes, ))
        mode_shapes = np.zeros((n_l, n_modes), dtype=complex)
        participation_vectors = np.zeros((n_r, n_modes), dtype=complex)
        
        for i, ind in enumerate(inds):
            lambda_i = _eigenvalues[ind]
            freq_i = _modal_frequencies[ind]
    
            # damping without correction if no exponential window was applied
            if factor_a is None:
                damping_i = np.real(lambda_i)/np.abs(lambda_i) * (-100)
            # damping with correction if exponential window was applied to
            else:
                damping_i = (np.real(lambda_i) / np.abs(lambda_i) - factor_a * (sampling_rate) / (freq_i * 2 * np.pi)) * (-100)
        
            # _modal_frequencies[ind] = freq_i
            modal_damping[i] = damping_i
            
            part_vec = eigvecs_l[-n_r:, ind]
            #normalize
            part_vec /= part_vec[np.argmax(np.abs(part_vec))]
            participation_vectors[:,i] = part_vec
        
        modal_frequencies = _modal_frequencies[inds]
        eigenvalues = _eigenvalues[inds]
        
        argsort = np.argsort(modal_frequencies)
        
        A = np.zeros((self.num_omega * 2 * n_r, (2*n_modes + 4 *n_r)))
        h = np.zeros((self.num_omega * 2 * n_r, n_l))
        
        for i_omega, omega in enumerate(self.selected_omega_vector):
            Df1 = (1 / (1j*omega - eigenvalues))
            Df2 = (1 / (1j*omega - np.conj(eigenvalues)))
            LDf1 = participation_vectors * Df1[np.newaxis,:]
            LDf2 = np.conj(participation_vectors) * Df2[np.newaxis,:]
            
            A_f = np.zeros((2*n_r, (2*n_modes + 4* n_r)))
            
            A_f[:n_r, :n_modes] = np.real(LDf1) + np.real(LDf2)
            A_f[n_r:, :n_modes] = np.imag(LDf1) + np.imag(LDf2)
            
            A_f[:n_r, n_modes:2 * n_modes] = - np.imag(LDf1) + np.real(LDf2)
            A_f[n_r:, n_modes:2 * n_modes] = np.real(LDf1) - np.real(LDf2)
            
            A_f[:n_r, 2 * n_modes: 2 * n_modes + n_r] = np.eye(n_r)
            A_f[n_r:, 2 * n_modes + n_r : 2 * n_modes + 2 * n_r] = np.eye(n_r)
            
            A_f[:n_r, 2 * n_modes + 2 * n_r: 2 * n_modes + 3 * n_r] = np.eye(n_r) * omega ** 2
            A_f[n_r:, 2 * n_modes + 3 * n_r : 2 * n_modes + 4 * n_r] = np.eye(n_r) * omega ** 2
            
            A[i_omega*2*n_r:(i_omega+1)*2*n_r,:] = A_f
            h[i_omega * 2 * n_r:i_omega * 2 * n_r + n_r,:] = np.real(self.pos_half_spectra[:,:,i_omega]).T
            h[i_omega * 2 * n_r + n_r:i_omega * 2 * n_r + 2 * n_r,:] = np.imag(self.pos_half_spectra[:,:,i_omega]).T
        
        X = np.linalg.pinv(A) @ h
        
        mode_shapes_raw = X.T[:,:n_modes] + 1j * X.T[:,n_modes:2*n_modes]
        
        for ind in range(n_modes):
            mode_shape_i = mode_shapes_raw[:,ind]
            
            # scale modeshapes to modal displacements
            mode_shape_i = self.integrate_quantities(
                mode_shape_i, accel_channels, velo_channels, freq_i * 2 * np.pi)
            
            # rotate mode shape in complex plane
            mode_shape_i = self.rescale_mode_shape(mode_shape_i)
            
            mode_shapes[:, ind] = mode_shape_i
        
        self._lower_residuals = X.T[:,2*n_modes:2*n_modes + n_r] + 1j * X.T[:,2*n_modes + n_r:2*n_modes + 2*n_r]
        self._upper_residuals = X.T[:,2*n_modes + 2*n_r:2*n_modes + 3*n_r] + 1j * X.T[:,2*n_modes + 3*n_r:2*n_modes + 4*n_r]
        self._mode_shapes_raw = mode_shapes_raw[:,argsort]
        self._participation_vectors = participation_vectors[:,argsort]
        self._eigenvalues = eigenvalues[argsort]
        
        return  modal_frequencies[argsort], modal_damping[argsort], mode_shapes[:,argsort], eigenvalues[argsort]

    def synthesize_spectrum(self, alpha, beta_l_i, modal=True):
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
        pos_half_spectra = self.pos_half_spectra
        
        omega = self.selected_omega_vector
        num_omega = self.num_omega
        
        if modal:
            if self._lower_residuals is None:
                logger.warning('Residuals have not yet been estimated.')
                _,_,_,_ = self.modal_analysis_residuals(alpha)
    
            lower_residuals  = self._lower_residuals 
            upper_residuals = self._upper_residuals
            participation_vectors = self._participation_vectors 
            mode_shapes_raw = self._mode_shapes_raw
            eigenvalues = self._eigenvalues
    
            n_modes = mode_shapes_raw.shape[1]
            
            Sigma_data = np.zeros((n_l * n_r), dtype=complex)
            Sigma_synth = np.zeros((n_l * n_r), dtype=complex)
            Sigma_data_synth = np.zeros((n_l * n_r, n_modes), dtype=complex)
            
            modal_contributions = np.zeros((n_modes), dtype=complex)
            
            half_spec_modal = np.zeros((n_l, n_r, num_omega, n_modes), dtype=complex)
    
            # https://www.sciencedirect.com/science/article/pii/S0888327024008033#sec2.1.2
            for ind in range(n_modes):
    
                lamda_r = eigenvalues[ind]
    
                part_vec = participation_vectors[:,ind]
                mode_shape = mode_shapes_raw[:,ind]
    
                half_spec_modal[:,:,:,ind] = (part_vec[:, np.newaxis]  @         mode_shape[np.newaxis, :]).T [:, :, np.newaxis] / (1j * omega[np.newaxis, np.newaxis, :] -         lamda_r ) \
                                   + np.conj((part_vec[:, np.newaxis]) @ np.conj(mode_shape[np.newaxis, :])).T[:, :, np.newaxis] / (1j * omega[np.newaxis, np.newaxis, :] - np.conj(lamda_r))
    
            half_spec_synth = np.sum(half_spec_modal, axis=-1)
            half_spec_synth[:,:,:] += lower_residuals[:,:,np.newaxis]
            half_spec_synth[:,:,:] += upper_residuals[:,:,np.newaxis]*omega[np.newaxis,np.newaxis,:]**2
    
            self._half_spec_synth = half_spec_modal
            
            if logger.isEnabledFor(logging.DEBUG):
                Sigma_data_synthtot = np.zeros((n_l * n_r))
                
            for i_r in range(n_r):
                for i_l in range(n_l):
                    spec_data = pos_half_spectra[i_l, i_r, :]
                    spec_synth = half_spec_synth[i_l, i_r, :]
                    spec_synth = np.sum(half_spec_modal, axis=-1)[i_l, i_r, :]
                    
                    Sigma_data[i_r * n_l + i_l] = spec_data @ np.conj(spec_data.T)
                    Sigma_synth[i_r * n_l + i_l] = spec_synth @ np.conj(spec_synth.T)
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        Sigma_data_synthtot[i_r * n_l + i_l] = spec_data @ np.conj(spec_synth.T)
                    
                    for i in range(n_modes):
                        Sigma_data_synth[i_r * n_l + i_l, i] = spec_data @ np.conj(half_spec_modal[i_l, i_r, :, i])
                    
            for i in range(n_modes):
                rho = (Sigma_data_synth[:, i] / np.sqrt(Sigma_data * Sigma_synth))
                modal_contributions[i] = rho.mean()
                
            self._modal_contributions = modal_contributions
            
            return half_spec_modal, modal_contributions
    
        else:
            # Peeters 2004 Eqs. 4, 3 and 1
            order = alpha.shape[0] // n_r - 1
            r_vec = np.arange(order + 1)
            half_spec_synth = np.zeros_like(self.pos_half_spectra) # (n_l, n_r, num_omega)
            for i_omega in range(self.num_omega):
                Omega_r = np.exp(1j * omega[i_omega] / sampling_rate * r_vec)
    
                # alpha # ((order + 1) * n_r, n_r)
                A = np.zeros((n_r,n_r), dtype=complex)
                for i_ord in range(order+1):
                    A += alpha[i_ord * n_r:(i_ord + 1) * n_r,:] * Omega_r[i_ord]# (n_r, n_r)
                A_inv = np.linalg.inv(A)
    
                #beta_i_l# (order + 1, n_r, n_l)
                B_o = np.sum(Omega_r[:,np.newaxis, np.newaxis] * beta_l_i[:, :, :], axis=0) # ( 1, n_r)
                half_spec_synth[:,:,i_omega] = B_o.T @ A_inv
    
            self._half_spec_synth = half_spec_synth
    
            return half_spec_synth, None

    def compute_modal_params(self, max_model_order, complex_coefficients=False, 
                             algo='residuals', modal_contrib=None):
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
        '''
        assert max_model_order <= self.nperseg - 1
        assert algo in ['state-space', 'residuals']
        
        if modal_contrib is None:
            if algo == 'state-space':
                modal_contrib = False
            else:
                modal_contrib = True
                
        if modal_contrib and algo=='state-space':
            logger.warning('State space algorithm can not be used with spectral synthetization.')
            algo = 'residuals'
        
        
        n_l = self.prep_signals.num_analised_channels
        n_r = self.prep_signals.num_ref_channels
            
        assert self.state[0]

        logger.info('Computing modal parameters...')
        
        # Peeters 2004,p 400: "a pth order right matrix-fraction model yield pm poles"
        # that occur in complex conjugate poles?
        if complex_coefficients:
            max_modes = max_model_order * n_r 
        else:
            max_modes = max_model_order * n_r // 2
        modal_frequencies = np.zeros((max_model_order, max_modes))
        modal_damping = np.zeros((max_model_order, max_modes))
        mode_shapes = np.zeros((n_l, max_modes, max_model_order), dtype=complex)
        eigenvalues = np.zeros((max_model_order, max_modes), dtype=complex)
        
        if modal_contrib:
            modal_contributions = np.zeros((max_model_order, max_modes,), dtype=complex)
        else:
            # reset modal contributions in case of a subsequent run without modal_contrib
            modal_contributions = None
        
        pbar = simplePbar(max_model_order)
        for order in range(1,max_model_order):
            next(pbar)
            
            alpha, beta_l_i = self.estimate_model(order, complex_coefficients)
            
            if algo=='state-space':
                f, d, phi, lamda = self.modal_analysis_state_space(alpha, beta_l_i )
            elif algo=='residuals':
                f, d, phi, lamda = self.modal_analysis_residuals(alpha, beta_l_i)
            n_modes = len(f)
            
            if modal_contrib:
                _, delta = self.synthesize_spectrum(alpha, beta_l_i, True)
                modal_contributions[order, :n_modes] = delta

            modal_frequencies[order, :n_modes] = f
            modal_damping[order, :n_modes] = d
            eigenvalues[order,:n_modes] = lamda
            mode_shapes[:, :n_modes, order] = phi
            

        self.max_model_order = max_model_order
        
        self.eigenvalues = eigenvalues
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        self.modal_contributions = modal_contributions
            
        self.state[1] = True

    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, filename = os.path.split(fname)
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

        in_dict = np.load(fname, allow_pickle=True)
        #             0         1           2
        # self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        assert isinstance(prep_signals, PreProcessSignals)
        setup_name = str(in_dict['self.setup_name'].item())
        assert setup_name == prep_signals.setup_name
        start_time = prep_signals.start_time

        assert start_time == prep_signals.start_time
        
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
            pLSCF_object.modal_frequencies = in_dict['self.modal_frequencies']
            pLSCF_object.modal_damping = in_dict['self.modal_damping']
            pLSCF_object.mode_shapes = in_dict['self.mode_shapes']
            pLSCF_object.eigenvalues = in_dict['self.eigenvalues']
            pLSCF_object.modal_contributions = in_dict['self.modal_contributions']
            pLSCF_object.max_model_order = int(in_dict['self.max_model_order'])

        return pLSCF_object

def plot_spec_synth(modal_data, modelist=None, channel_inds=None, ref_channel_inds=None, axes=None):
    import matplotlib.pyplot as plt
    
    half_spec_synth = modal_data._half_spec_synth
    num_omega = modal_data.num_omega
    pos_half_spectra = modal_data.pos_half_spectra
    
    
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
    
    # Plot power spectral density functions for each channel combination and all modes
    num_plots =  len(i_l_i_r)
    
    fig2, axes = plt.subplots(num_plots, 1, sharex='col', sharey='col', squeeze=False)
    
    ft_freq = modal_data.selected_omega_vector / 2 / np.pi
    
    for j in range(num_plots):
        i_l, i_r = i_l_i_r[j,:]    
        
        ft_meas = pos_half_spectra[i_l, i_r, :]
        
        if j==0: label=f'Inp.'
        else: label=None
        
        axes[j, 0].plot(ft_freq, 10 * np.log10(np.abs(ft_meas)), ls='solid', color='k', label=label)
        
        for ip,i in enumerate(modelist):
            ft_synth = half_spec_synth[i_l, i_r, :, i]
            
            color = str(np.linspace(0, 1, len(modelist) + 2)[ip + 1])
            ls = ['-','--',':','-.'][i%4]
            if j==0: label=f'm={i+1}'
            else: label=None
            
            axes[j, 0].plot(ft_freq, 10*np.log10(np.abs(ft_synth)), color=color, ls=ls, label=label)
            axes[j,0].set_ylabel(f'{channel_headers[i_l]}\n $\leftrightarrow$ \n{channel_headers[ref_channels[i_r]]}', 
                                 rotation=0, labelpad=20, va='center', ha='center')

            
    axes[-1, 0].set_xlabel('$f$ [\si{\hertz}]')
    for ax in axes.flat:
        ax.set_yticks([])
        ax.set_xlim(0,1/2*sampling_rate)
        
        ax.set_ylim(ymin=-50)
    fig2.legend(title='Mode')
    fig2.subplots_adjust(left=None, bottom=None, right=0.97, top=0.97, wspace=None, hspace=0.1,)
    
    return fig2

def main():
    pass


if __name__ == '__main__':
    main()
