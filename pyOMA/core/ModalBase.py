# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Base class shared by all pyOMA system-identification methods."""

from .PreProcessingTools import PreProcessSignals
import numpy as np
from collections import deque
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ModalBase(object):
    """Base class from which all pyOMA system-identification classes inherit.

    Provides shared functionality (conjugate removal, mode-shape integration,
    rescaling, persistence) so that derived classes only implement the
    method-specific identification steps.  Post-processing tools (stabilization
    diagram, mode-shape plot) accept any :class:`ModalBase` subclass instance.

    Attributes
    ----------
    prep_signals : PreProcessSignals or None
        The signal object from which this analysis was created.
    setup_name : str
        Human-readable label for the measurement setup.
    start_time : datetime.datetime or None
        Timestamp of the measurement.
    num_analised_channels : int or None
        Total number of analysis channels.
    num_ref_channels : int or None
        Number of reference channels.
    max_model_order : int or None
        Maximum model order used in the identification.
    modal_frequencies : np.ndarray or None
        Identified natural frequencies (Hz), shape ``(max_model_order, n_modes)``.
    modal_damping : np.ndarray or None
        Identified modal damping ratios (%), same shape as ``modal_frequencies``.
    mode_shapes : np.ndarray or None
        Identified mode shapes, shape ``(n_channels, n_modes, max_model_order)``.
    eigenvalues : np.ndarray or None
        Identified (complex) eigenvalues.
    """

    def __init__(self, prep_signals=None):
        """
        Parameters
        ----------
        prep_signals : PreProcessSignals, optional
            Pre-processed signal object.  When ``None``, channel metadata
            attributes are initialised to ``None`` and must be set manually
            (e.g. when loading a saved state).
        """
        super().__init__()
        if prep_signals is not None:
            if not isinstance(prep_signals, PreProcessSignals):
                logger.warning(f'Argument prep_signals is wrong object type {type(prep_signals)}')
            self.setup_name = prep_signals.setup_name
            self.start_time = prep_signals.start_time
            self.num_analised_channels = prep_signals.num_analised_channels
            self.num_ref_channels = prep_signals.num_ref_channels
        else:
            self.setup_name = ''
            self.start_time = None
            self.num_analised_channels = None
            self.num_ref_channels = None

        self.prep_signals = prep_signals

        self.max_model_order = None

        self.eigenvalues = None
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None

    @staticmethod
    def remove_conjugates(eigval, eigvec_r=None, eigvec_l=None, inds_only=False):
        '''
        This method finds complex conjugate modes, and removes unstable and 
        overdamped poles. 
        
        A complex conjugate is defined as:
        :math:`\\lambda_i = \\overline{\\lambda_j} \\text{ for } i \\neq j`

        Unstable poles, i.e. negatively damped poles, are defined by:
        :math:`[\\ln(|\\lambda|)<0]: |\\lambda_i|> 1`

        Overdamped poles, are purely real poles:
        :math:`[\\operatorname{atan}(\\Im/\\Re)=0]: \\Im(\\lambda_i)=0`

        The method keeps the second occurance of a conjugate pair (usually the one
        with the negative imaginary part) and either returns a truncated set of 
        eigenvalues and eigenvectors or a list of (physical) poles that can be 
        iterated.
        
        Parameters
        ----------
            eigval: (order,) numpy.ndarray
                Complex array of all eigenvalues
            eigvec_r, eigvec_l: (order, n_channels) numpy.ndarray, optional
                Complex array(s) of all right (left) eigenvectors
            inds_only: bool, optional
                Whether to return a list of pole indices, or a reduced set of 
                eigenvalues and eigenvectors
        
        Returns
        -------
            conj_indices:  list
                list of (physical) pole indices
            eigval: (order,) numpy.ndarray
                Complex array of reduced (physical) eigenvalues
            eigvec_l, eigvec_r: (order, n_channels) numpy.ndarray, optional
                Complex array(s) of reduced (physical) left (right) eigenvectors
        '''

        num_val = len(eigval)
        conj_indices = deque()

        for i in range(num_val):
            this_val = eigval[i]
            this_conj_val = np.conj(this_val)
            # remove overdamped poles  i.e. real eigvals
            # use isclose instead of == to handle tiny floating-point imaginary residuals
            # produced by some LAPACK implementations (e.g. MKL on Windows)
            if np.isclose(this_val.imag, 0.0):
                conj_indices.append(i)
            # remove negatively damped poles i.e. unstable poles
            elif np.abs(this_val) > 1:
                conj_indices.append(i)
            # catches unordered conjugates but takes slightly longer
            for j in range(i + 1, num_val):
                if np.isclose(eigval[j] , this_conj_val):
                    conj_indices.append(j)
                    break

        conj_indices = list(set(range(num_val)).difference(conj_indices))

        if inds_only:
            return conj_indices

        if eigvec_l is None:

            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_r

        else:
            eigvec_l = eigvec_l[:, conj_indices]
            eigvec_r = eigvec_r[:, conj_indices]
            eigval = eigval[conj_indices]

            return eigval, eigvec_l, eigvec_r

    @classmethod
    def init_from_config(cls, conf_file, prep_signals):
        """Initialise a modal analysis object from a text configuration file.

        This is a stub that must be fully reimplemented by every derived class.
        Derived implementations typically read analysis parameters (e.g. model
        order, frequency range) from *conf_file*, call the relevant computation
        methods, and return the populated object.

        Parameters
        ----------
        conf_file : str
            Path to a tab-separated key-value configuration file compatible
            with :class:`~pyOMA.core.Helpers.ConfigFile`.
        prep_signals : PreProcessSignals
            Pre-processed signal object for this setup.

        Returns
        -------
        ModalBase
            Populated subclass instance.
        """

        assert os.path.exists(conf_file)
        assert isinstance(prep_signals, PreProcessSignals)

        with open(conf_file, 'r') as _:
            # read configuration parameters line by line
            pass

        modal_object = cls(prep_signals)

        return modal_object

    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
        '''
        Rescales mode shapes from modal accelerations / velocities to modal
        displacements, by multiplication of the relevant modal coordinates 
        (where accelerometers, or velocimeters were used, with 
        $-1 \omega^2$ or $i \omega$, respectively,
        
        Parameters
        ----------
            vector: (n_channels,) numpy.ndarray
                Complex modeshape for all n_channels
            accel_channels: list
                A list containing the channel numbers of all acceleration channels
            velo_channels: list
                A list containing the channel numbers of all velocity channels
            omega: float
                The circular frequency of the corresponding mode ($\omega = 2 \pi f$)
        
        Returns
        -------
            vector:  (n_channels,) numpy.ndarray
                Rescaled complex modeshape for all n_channels
        '''
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        #                     phase + 180; magn / omega^2
        vector = np.copy(vector)

        vector[accel_channels] *= -1 / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *= 1j / omega

        return vector

    @staticmethod
    def rescale_mode_shape(modeshape, rotate_only=False):
        '''
        Rescales and rotates modeshapes in the complex plane. Default behaviour 
        is to scale the larges component to unit modal displacement. If argument
        rotate_only is provided, the method given in Appendix C2 of Doehler 2013
        (doi:0.1016/j.ymssp.2012.11.011) is used to rotate but not rescale the 
        mode shape. Note: The scale of identified mode shapes is arbitrary in most 
        OMA methods.
        
        Parameters
        ----------
            modeshape: (n_channels,) numpy.ndarray
                Complex modeshape for all n_channels
            
            rotate_only: bool, optional
                Whether to rotate, but not rescale, the mode shape.
        
        Returns
        -------
            modeshape:  (n_channels,) numpy.ndarray
                Rescaled complex modeshape for all n_channels
        '''
        # scaling of mode shape
        if rotate_only:
            k = np.argmax(np.abs(modeshape))
            alpha = np.angle(modeshape[k])
            return modeshape * np.exp(-1j * alpha)
        else:
            modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
            return modeshape

    def save_state(self, fname):
        """Save the current computation state to a compressed NumPy archive.

        Must be fully reimplemented by every derived class.

        Parameters
        ----------
        fname : str
            Destination file path (without ``.npz`` extension).

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a derived class.
        """
        raise NotImplementedError(
            'save_state must be reimplemented by every derived class.')

    @classmethod
    def load_state(cls, fname, prep_signals):
        """Restore a modal-analysis object from a previously saved archive.

        Must be fully reimplemented by every derived class.

        Parameters
        ----------
        fname : str
            Path to the ``.npz`` archive written by :meth:`save_state`.
        prep_signals : PreProcessSignals
            Signal object for the same setup; used to validate the archive.

        Returns
        -------
        ModalBase
            Restored subclass instance.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a derived class.
        """
        raise NotImplementedError(
            'load_state must be reimplemented by every derived class.')
