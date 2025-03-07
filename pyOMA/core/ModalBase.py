'''
pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

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

Module that contains the basic class, of which all other OMA classes
should be inherited.

@author: womo1998
'''

from .PreProcessingTools import PreProcessSignals
import numpy as np
from collections import deque
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ModalBase(object):
    '''
    Base Class from which all other modal analysis classes should be inherited
        * provides commonly used functions s.t. these don't have to be copied to
          each class
        * object type checks in post-processing functions can check for
          modal base instead of each possible modal analysis class
    '''

    def __init__(self, prep_signals=None):
        super().__init__()
        if prep_signals is not None:
            assert isinstance(prep_signals, PreProcessSignals)
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
            if this_val == this_conj_val:
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
        '''
        A method for initializing a modal object from configuration data
        bypassing common operations in explicit code for semi-automated
        analyses

        This is a stub of the method that must be reimplemented by every
        derived class

        '''

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
        '''
        Saves the state of the object to a compressed numpy archive file
        This is only a stub for reimplementing the method in a derived class
        '''

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}
        out_dict['self.setup_name'] = self.setup_name

        raise NotImplementedError(
            'This method must be fully reimplemented by every derived class.')

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, prep_signals):
        '''
        Loads the state of the object from a compressed numpy archive file
        and returns the object
        This is only a stub for reimplementing the method in a derived class
        '''

        print('Now loading previous results from  {}'.format(fname))

        assert os.path.exists(fname)
        assert isinstance(prep_signals, PreProcessSignals)
        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['',
                                                    ]):
            if this_state:
                print(state_string)

        setup_name = str(in_dict['self.setup_name'].item())
        assert setup_name == prep_signals.setup_name

        modal_object = cls(prep_signals)
        modal_object.state = state

        raise NotImplementedError(
            'This method must be fully reimplemented by every derived class.')

        return modal_object
