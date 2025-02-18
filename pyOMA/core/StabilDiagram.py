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
Modified and Extended by Simon Marwitz 2015 ff.

..TODO::
 * scale markers right on every platform
 * frequency range as argument or from ssi params, sampling freq
 * add switch to choose between "unstable only in ..." or "stable in ..."
 * (select and merge several poles with a rectangular mouse selection)
 * distinguish beetween stabilization criteria and filtering criteria
 * add zoom and sliders (horizontal/vertical) for the main figure
 * distinguish between  "export results" and "save state"
 * rework mask logic (currently it is very difficult to understand)
 * Merge DataCursor and JupyterGUI.SnappingCursor 

'''

from .SSICovRef import PogerSSICovRef
from .ModalBase import ModalBase
from .Helpers import simplePbar, calculateMAC, calculateMPC, calculateMPD
import numpy as np

import scipy.cluster
import scipy.spatial
import scipy.stats

import os

import collections
from operator import itemgetter
from random import shuffle

# check if python is running in headless mode i.e. as a server script
# if 'DISPLAY' in os.environ:
#     matplotlib.use("Qt5Agg", force=True)
from matplotlib import rcParams
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.text import TextPath, FontProperties
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.widgets import Cursor
import matplotlib.cm
import matplotlib.pyplot as plot

plot.rc('figure', figsize=[8.5039399474194, 5.255723925793184], dpi=100,)
plot.rc('font', size=10)
plot.rc('legend', fontsize=10, labelspacing=0.1)
plot.rc('axes', linewidth=0.2)
plot.rc('xtick.major', width=0.2)
plot.rc('ytick.major', width=0.2)
# plot.ioff()

NoneType = type(None)


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class StabilCalc(object):

    def __init__(self, modal_data, prep_signals=None, **kwargs):

        super().__init__()
        #print(type(modal_data), file=sys.stderr)
        assert isinstance(modal_data, ModalBase)

        self.modal_data = modal_data

        self.extra_func = None

        self.setup_name = modal_data.setup_name
        self.start_time = modal_data.start_time

        if prep_signals is not None:
            logger.warning('Providing prep_signals is not required anymore. Ignoring argument!')
            # assert isinstance(prep_signals, PreProcessSignals)
        self.prep_signals = modal_data.prep_signals

        has_mode_shapes = self.modal_data.__dict__.get(
            'mode_shapes', None) is not None
        has_variances = self.modal_data.__dict__.get(
            'std_frequencies', None) is not None
        has_data = self.modal_data.prep_signals is not None
        has_MC = self.modal_data.__dict__.get(
            'modal_contributions', None) is not None
        has_ev = self.modal_data.__dict__.get(
            'eigenvalues', None) is not None

        self.capabilities = {'f': 1,
                             'd': 1,
                             'msh': has_mode_shapes,
                             'std': has_variances,
                             'ev': has_ev,
                             'mtn': 0,
                             'MC': has_MC,
                             'auto': isinstance(self, StabilCluster),
                             'data': has_data}

        if self.capabilities['ev']:
            self.masked_lambda = np.ma.array(
                self.modal_data.eigenvalues, fill_value=0)

        self.masked_frequencies = np.ma.array(
            self.modal_data.modal_frequencies, copy=True, fill_value=0)
        self.masked_frequencies[np.isnan(self.masked_frequencies)] = 0
        
        self.masked_damping = np.ma.array(
            self.modal_data.modal_damping, fill_value=0)

        max_model_order = self.modal_data.max_model_order
        self.num_solutions = self.modal_data.modal_frequencies.shape[1]

        self.order_dummy = np.ma.array(
            [[order] * self.num_solutions for order in range(max_model_order)], fill_value=0)

        # stable-in-... masks
        self.masks = {'mask_pre': None,  # some constraints (f>0.0, order_range, etc)
                      'mask_ad': None,  # absolute damping
                      'mask_stdf': None,  # uncertainty frequency
                      'mask_stdd': None,  # uncertainty damping
                      'mask_mpc': None,  # absolute modal phase collinearity
                      'mask_mpd': None,  # absolute mean phase deviation
                      'mask_mtn': None,  # absolute modal transfer norm
                      'mask_df': None,  # difference frequency
                      'mask_dd': None,  # difference damping
                      'mask_dmac': None,  # difference mac
                      'mask_dev': None,  # difference eigenvalue
                      'mask_dmtn': None,  # difference modal transfer norm
                      'mask_stable': None  # stable in all criteria
                      }

        # only-unstable-in-... masks
        self.nmasks = {'mask_ad': None,  # absolute damping
                       'mask_stdf': None,  # uncertainty frequency
                       'mask_stdd': None,  # uncertainty damping
                       # absolute modal phase collineratity
                       'mask_ampc': None,
                       'mask_ampd': None,  # absolute mean phase deviation
                       'mask_amtn': None,  # absolute modal transfer norm
                       'mask_df': None,  # difference frequency
                       'mask_dd': None,  # difference damping
                       'mask_dmac': None,  # difference mac
                       'mask_dev': None,  # difference eigenvalue
                       'mask_dmtn': None,  # difference modal transfer norm
                       }

        self.select_modes = []
        self.select_callback = None
        self.state = 0
        
        
        self.order_range = (0, 1, self.modal_data.max_model_order)
        self.d_range = (0, 100)    
        self.stdf_max = 100    
        self.stdd_max = 100    
        self.mpc_min = 0    
        self.mpd_max = 90    
        self.mtn_min = 0    
        self.df_max = 0.01    
        self.dd_max = 0.05    
        self.dmac_max = 0.02    
        self.dev_min = 0.02    
        self.dmtn_min = 0.02    
        self.MC_min = 0

        # print(self.capabilities)
        # self.calculate_soft_critera_matrices()
        
        self.callbacks = {'add_mode':[],
                  'remove_mode':[],}
        
    def add_callback(self, name, func):
        assert name in ['add_mode', 'remove_mode']
        self.callbacks[name].append(func)
        
    def calculate_soft_critera_matrices(self):
        logger.info('Checking stabilisation criteria...')

        # Direction 1: model order, Direction 2: current pole, Direction 3:
        # previous pole:
        max_model_order = self.modal_data.max_model_order
        num_solutions = self.num_solutions
        capabilities = self.capabilities

        lambda_diffs = np.ma.zeros(
            (max_model_order, num_solutions, num_solutions), fill_value=0)
        freq_diffs = np.ma.zeros(
            (max_model_order, num_solutions, num_solutions), fill_value=0)
        damp_diffs = np.ma.zeros(
            (max_model_order, num_solutions, num_solutions), fill_value=0)

        if capabilities['msh']:
            MAC_diffs = np.ma.zeros(
                (max_model_order, num_solutions, num_solutions), fill_value=0)
            MPD_diffs = np.ma.zeros(
                (max_model_order, num_solutions, num_solutions), fill_value=0)
            MP_diffs = np.ma.zeros(
                (max_model_order, num_solutions, num_solutions), fill_value=0)

            MPC_matrix = np.ma.zeros(
                (max_model_order, num_solutions), fill_value=0)
            MP_matrix = np.ma.zeros(
                (max_model_order, num_solutions), fill_value=0)
            MPD_matrix = np.ma.zeros(
                (max_model_order, num_solutions), fill_value=0)

        if capabilities['ev']:
            prev_lambda_row = self.masked_lambda.data[0, :]
        prev_freq_row = self.masked_frequencies[0, :]
        prev_damp_row = self.modal_data.modal_damping[0, :]
        if capabilities['msh']:
            prev_mode_shapes_row = self.modal_data.mode_shapes[:, :, 0]

        # tuple with array of indizes of non-zero frequencies
        if capabilities['ev']:
            # prev_non_zero_entries = np.nonzero(prev_lambda_row.imag)
            prev_non_zero_entries = np.where((~np.isnan(prev_lambda_row.imag)) & (prev_lambda_row.imag!=0))
        else:
            # prev_non_zero_entries = np.nonzero(prev_freq_row)
            prev_non_zero_entries = np.where((~np.isnan(prev_freq_row)) & (prev_freq_row!=0))
        prev_length = len(prev_non_zero_entries[0])
        if capabilities['ev']:
            prev_lambda = prev_lambda_row[prev_non_zero_entries]
        prev_freq = prev_freq_row[prev_non_zero_entries]
        prev_damp = prev_damp_row[prev_non_zero_entries]

        if capabilities['msh']:
            prev_mode_shapes = \
                prev_mode_shapes_row[:, prev_non_zero_entries[0]]

            prev_MPD, prev_MP_new = calculateMPD(prev_mode_shapes)
            prev_MP_new[prev_MP_new > 90] -= 180  # in range [-90,90]
        
        pbar = simplePbar(max_model_order - 1)
        for curr_order in range(1, max_model_order):
            next(pbar)

            if capabilities['ev']:
                curr_lambda_row = self.masked_lambda.data[(curr_order), :]

            curr_freq_row = self.masked_frequencies[(curr_order), :]

            curr_damp_row = self.modal_data.modal_damping[
                (curr_order), :]

            if capabilities['msh']:
                curr_mode_shapes_row = \
                    self.modal_data.mode_shapes[:, :, curr_order]

            # catches zeros and also real poles, which should have been removed
            # in remove conjugates already
            if capabilities['ev']:
                # curr_non_zero_entries = np.nonzero(curr_lambda_row.imag)
                curr_non_zero_entries = np.where((~np.isnan(curr_lambda_row.imag)) & (curr_lambda_row.imag!=0))
            else:
                # curr_non_zero_entries = np.nonzero(curr_freq_row)
                curr_non_zero_entries = np.where((~np.isnan(curr_freq_row)) & (curr_freq_row!=0))

            curr_length = len(curr_non_zero_entries[0])

            # print(curr_length)
            if not curr_length:
                continue
            if capabilities['ev']:
                curr_lambda = curr_lambda_row[curr_non_zero_entries]
            curr_freq = curr_freq_row[curr_non_zero_entries]
            curr_damp = curr_damp_row[curr_non_zero_entries]

            if capabilities['msh']:
                curr_mode_shapes = \
                    curr_mode_shapes_row[:, curr_non_zero_entries[0]]
            if capabilities['ev']:

                div_lambda = np.maximum(
                    np.repeat(np.expand_dims(np.ma.abs(prev_lambda), axis=1),
                              curr_lambda.shape[0], axis=1),
                    np.repeat(np.expand_dims(np.ma.abs(curr_lambda), axis=0),
                              prev_lambda.shape[0], axis=0))

            div_freq = np.maximum(
                np.repeat(np.expand_dims(np.abs(prev_freq), axis=1),
                          curr_freq.shape[0], axis=1),
                np.repeat(np.expand_dims(np.abs(curr_freq), axis=0),
                          prev_freq.shape[0], axis=0))

            div_damp = np.maximum(
                np.repeat(np.expand_dims(np.abs(prev_damp), axis=1),
                          curr_damp.shape[0], axis=1),
                np.repeat(np.expand_dims(np.abs(curr_damp), axis=0),
                          prev_damp.shape[0], axis=0))

            if capabilities['msh']:
                mac_diffs = np.transpose(1 - calculateMAC(
                    prev_mode_shapes[:, :prev_length], curr_mode_shapes[:, :curr_length]))
                # print(mac_diffs)
                MAC_diffs[curr_order,
                          curr_non_zero_entries[0],
                          :prev_length] = mac_diffs

                MPC_matrix[curr_order, curr_non_zero_entries[0]] = calculateMPC(
                    curr_mode_shapes[:, :curr_length])

                curr_MPD, curr_MP = calculateMPD(
                    curr_mode_shapes[:, :curr_length])
                MPD_matrix[curr_order, curr_non_zero_entries[0]], MP_matrix[
                    curr_order, curr_non_zero_entries[0]] = curr_MPD, curr_MP

            if capabilities['ev']:
                lambda_diffs[curr_order, curr_non_zero_entries[0], :len(prev_lambda)] = np.abs((np.repeat(
                    np.expand_dims(prev_lambda, axis=1), curr_lambda.shape[0], axis=1) - curr_lambda) / div_lambda).T
            freq_diffs[curr_order, curr_non_zero_entries[0], :len(prev_freq)] = np.abs((np.repeat(
                np.expand_dims(prev_freq, axis=1), curr_freq.shape[0], axis=1) - curr_freq) / div_freq).T
            damp_diffs[curr_order, curr_non_zero_entries[0], :len(prev_damp)] = np.abs((np.repeat(
                np.expand_dims(prev_damp, axis=1), curr_damp.shape[0], axis=1) - curr_damp) / div_damp).T
            
            if capabilities['msh']:

                div_MPD = np.maximum(
                    np.repeat(
                        np.expand_dims(
                            np.abs(prev_MPD),
                            axis=1),
                        curr_MPD.shape[0],
                        axis=1),
                    np.repeat(
                        np.expand_dims(
                            np.abs(curr_MPD),
                            axis=0),
                        prev_MPD.shape[0],
                        axis=0))

                MPD_diffs[curr_order, curr_non_zero_entries[0], :len(prev_MPD)] = np.abs((np.repeat(
                    np.expand_dims(prev_MPD, axis=1), curr_MPD.shape[0], axis=1) - curr_MPD) / div_MPD).T

                curr_MP_new = np.copy(curr_MP)  # in range [0,180]
                curr_MP_new[curr_MP_new > 90] -= 180  # in range [-90,90]

                div_MP = np.maximum(
                    np.repeat(
                        np.expand_dims(
                            np.abs(prev_MP_new),
                            axis=1),
                        curr_MP_new.shape[0],
                        axis=1),
                    np.repeat(
                        np.expand_dims(
                            np.abs(curr_MP_new),
                            axis=0),
                        prev_MP_new.shape[0],
                        axis=0))

                MP_diffs[curr_order, curr_non_zero_entries[0], :len(prev_MP_new)] = np.abs((np.repeat(
                    np.expand_dims(prev_MP_new, axis=1), curr_MP_new.shape[0], axis=1) - curr_MP_new) / div_MP).T

            if capabilities['ev']:
                prev_lambda = curr_lambda
            prev_freq = curr_freq
            prev_damp = curr_damp

            if capabilities['msh']:
                prev_mode_shapes = curr_mode_shapes
                prev_MPD = curr_MPD
                prev_MP_new = curr_MP_new

            prev_length = curr_length
            prev_non_zero_entries = curr_non_zero_entries
        if capabilities['ev']:
            self.lambda_diffs = lambda_diffs
        self.freq_diffs = freq_diffs
        self.damp_diffs = damp_diffs
        self.MAC_diffs = MAC_diffs
        self.MPD_diffs = MPD_diffs
        self.MP_diffs = MP_diffs

        self.MPD_matrix = MPD_matrix
        self.MP_matrix = MP_matrix
        self.MPC_matrix = MPC_matrix

        self.state = 1

    def export_results(self, fname, binary=False):

        selected_freq, selected_damp, selected_modes, _,\
            selected_order, selected_MC,\
            selected_MPC, selected_MP, selected_MPD,\
            selected_stdf, selected_stdd, selected_stdmsh = self.get_selected_modal_values()
        
        freq_str = ''
        damp_str = ''
        ord_str = ''

        if self.capabilities['msh']:
            msh_str = ''
            mpc_str = ''
            mp_str = ''
            mpd_str = ''
        if self.capabilities['std']:
            std_freq_str = ''
            std_damp_str = ''
            std_msh_str = ''
        if self.capabilities['MC']:
            MC_str = ''

        for col in range(len(selected_freq)):
            freq_str += '{:<3.3f}\t\t'.format(selected_freq[col])
            damp_str += '{:<3.3f}\t\t'.format(selected_damp[col])
            ord_str += '{:<6d}\t\t'.format(selected_order[col])

            if self.capabilities['msh']:
                mpc_str += '{:<3.3f}\t \t'.format(selected_MPC[col])
                mp_str += '{:<3.2f}\t\t'.format(selected_MP[col])
                mpd_str += '{:<3.2f}\t\t'.format(selected_MPD[col])

            if self.capabilities['std']:
                std_damp_str += '{:<3.3e}\t\t'.format(selected_stdd[col])
                std_freq_str += '{:<3.3e}\t\t'.format(selected_stdf[col])

            if self.capabilities['MC']:
                MC_str += '{:<3.3f}\t\t'.format(selected_MC[col])

        if self.capabilities['msh']:
            for row in range(selected_modes.shape[0]):
                if isinstance(self.modal_data, PogerSSICovRef):
                    chan_dofs = self.modal_data.merged_chan_dofs
                elif self.capabilities['data']:
                    chan_dofs = self.prep_signals.chan_dofs
                else:
                    chan_dofs = []
                for chan_dof in chan_dofs:
                    chan, node, az, elev = chan_dof[:4]
                    # print(chan, row, chan==row)
                    if chan==row:
                        msh_str += f'\n{node.ljust(10)}  ({az: <+3.2f}, {elev: >+3.2f})                  \t'
                        break
                else:
                    msh_str += '\n                                            '
                if self.capabilities['std']:
                    std_msh_str += '\n           \t\t'
                for col in range(selected_modes.shape[1]):
                    msh_str += '{:+<3.4f}\t'.format(selected_modes[row, col])
                    if self.capabilities['std']:
                        std_msh_str += '{:+<3.3e} \t'.format(
                            selected_stdmsh[row, col])

        export_modes      = 'MANUAL MODAL ANALYSIS\n'
        export_modes     += '=======================\n'
        export_modes     += 'Frequencies [Hz]:                           \t' + freq_str + '\n'
        if self.capabilities['std']:
            export_modes += 'Standard deviations of the Frequencies [Hz]:\t' + \
                std_freq_str + '\n'
        export_modes     += 'Damping [%]:                                \t' + damp_str + '\n'
        if self.capabilities['std']:
            export_modes += 'Standard deviations of the Damping [%]:     \t' + \
                std_damp_str + '\n'
        if self.capabilities['MC']:
            export_modes += 'Modal Contributions of the mode [-]:        \t' + \
                MC_str + '\n'
        if self.capabilities['msh']:
            export_modes += 'Node        (Azimuth, Elevation)            \tMode shapes:' + msh_str + '\n'
        if self.capabilities['std']:
            export_modes += 'Standard Deviations of the Mode shapes:     \t' + \
                std_msh_str + '\n'
        export_modes     += 'Model order:                                \t' + ord_str + '\n'
        if self.capabilities['msh']:
            export_modes += 'MPC [-]:                                    \t' + mpc_str + '\n'
            export_modes += 'MP  [\u00b0]:                                    \t' + mp_str + '\n'
            export_modes += 'MPD [-]:                                    \t' + mpd_str + '\n\n'
        #              + 'SSI parameters\n'
        #              + '=======================\n'\
        #              + 'Maximum order :\t\t'     + str(self.modal_data.max_model_order) + '\n'\
        #              + 'Block rows :\t\t'        + str(self.num_block_rows)     + '\n'\
        #              + 'Block columns :\t\t'     + str(self.num_block_columns)  + '\n'
        #              + 'Decimation :\t\t'        + str(dec_fact)       + '\n'\
        #              + 'Filtering :\t\t'         + str(filt_w)

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if binary:
            out_dict = {'selected_freq': selected_freq,
                        'selected_damp': selected_damp,
                        'selected_order': selected_order}

            if self.capabilities['msh']:
                out_dict['selected_MPC'] = selected_MPC
                out_dict['selected_MP'] = selected_MP
                out_dict['selected_MPD'] = selected_MPD
                out_dict['selected_modes'] = selected_modes

            if self.capabilities['std']:
                out_dict['selected_stdf'] = selected_stdf
                out_dict['selected_stdd'] = selected_stdd
                out_dict['selected_stdmsh'] = selected_stdmsh

            np.savez_compressed(fname, **out_dict)

        else:
            f = open(fname, 'w')
            f.write(export_modes)
            f.close()

    def calculate_stabilization_masks(
            self,
            order_range=None,
            d_range=None,
            stdf_max=None,
            stdd_max=None,
            mpc_min=None,
            mpd_max=None,
            mtn_min=None,
            df_max=None,
            dd_max=None,
            dmac_max=None,
            dev_min=None,
            dmtn_min=None,
            MC_min=None):
        if self.state < 1:
            self.calculate_soft_critera_matrices()

        if order_range is None:
            order_range = (0, 1, self.modal_data.max_model_order)
        if d_range is None:
            d_range = (0, 100)
        if stdf_max is None:
            stdf_max = 100
        if stdd_max is None:
            stdd_max = 100
        if mpc_min is None:
            mpc_min = 0
        if mpd_max is None:
            mpd_max = 90
        if mtn_min is None:
            mtn_min = 0
        if df_max is None:
            df_max = 0.01
        if dd_max is None:
            dd_max = 0.05
        if dmac_max is None:
            dmac_max = 0.02
        if dev_min is None:
            dev_min = 0.02
        if dmtn_min is None:
            dmtn_min = 0.02
        if MC_min is None:
            MC_min = 0

        self.state = 2

        self.update_stabilization_masks(
            order_range,
            d_range,
            stdf_max,
            stdd_max,
            mpc_min,
            mpd_max,
            mtn_min,
            df_max,
            dd_max,
            dmac_max,
            dev_min,
            dmtn_min,
            MC_min)

    def update_stabilization_masks(self, order_range=None, d_range=None,
                                   stdf_max=None, stdd_max=None,
                                   mpc_min=None, mpd_max=None, mtn_min=None,
                                   df_max=None, dd_max=None, dmac_max=None,
                                   dev_min=None, dmtn_min=None, MC_min=None):
        if self.state < 2:
            self.calculate_stabilization_masks()
        # update
        #print(order_range , d_range , stdf_max , stdd_max, mpc_min, mpd_max,  mtn_min,df_max, dd_max, dmac_max, dev_min, dmtn_min)
        if order_range is not None:
            self.order_range = order_range
        if d_range is not None:
            self.d_range = d_range
        if stdf_max is not None:
            self.stdf_max = stdf_max
        if stdd_max is not None:
            self.stdd_max = stdd_max
        if mpc_min is not None:
            self.mpc_min = mpc_min
        if mpd_max is not None:
            self.mpd_max = mpd_max
        if mtn_min is not None:
            self.mtn_min = mtn_min
        if df_max is not None:
            self.df_max = df_max
        if dd_max is not None:
            self.dd_max = dd_max
        if dmac_max is not None:
            self.dmac_max = dmac_max
        if dev_min is not None:
            self.dev_min = dev_min
        if dmtn_min is not None:
            self.dmtn_min = dmtn_min
        if MC_min is not None:
            self.MC_min = MC_min

        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        mask_pre = (~np.isnan(self.masked_frequencies)) & (self.masked_frequencies != 0)

        if order_range is not None:
            start, step, stop = order_range
            start = max(0, start)
            stop = min(stop, self.modal_data.max_model_order)
            mask_order = np.zeros_like(mask_pre)
            for order in range(start, stop, step):
                mask_order = np.logical_or(
                    mask_order, self.order_dummy == order)
            mask_pre = np.logical_and(mask_pre, mask_order)

        self.masks['mask_pre'] = mask_pre

        if d_range is not None:
            assert isinstance(d_range, (tuple, list))
            assert len(d_range) == 2
            mask = np.logical_and(
                mask_pre, self.modal_data.modal_damping >= d_range[0])
            mask = np.logical_and(
                mask, self.modal_data.modal_damping <= d_range[1])
            self.masks['mask_ad'] = mask

        if stdf_max is not None and self.capabilities['std']:
            mask = self.modal_data.std_frequencies <= stdf_max * \
                self.modal_data.modal_frequencies
            mask = np.logical_and(mask_pre, mask)
            self.masks['mask_stdf'] = mask

        if stdd_max is not None and self.capabilities['std']:
            mask = self.modal_data.std_damping <= stdd_max * \
                self.modal_data.modal_damping
            mask = np.logical_and(mask_pre, mask)
            self.masks['mask_stdd'] = mask

        if mpc_min is not None:
            mask = np.logical_and(mask_pre, self.MPC_matrix >= mpc_min)
            self.masks['mask_mpc'] = mask

        if mpd_max is not None:
            mask = np.logical_and(mask_pre, self.MPD_matrix <= mpd_max)
            self.masks['mask_mpd'] = mask

        # if mtn_min is not None:
        #     logger.warning('Modal Transfer Norm is not yet implemented! Ignoring')

        if MC_min is not None and self.capabilities['MC']:
            if np.issubdtype(self.modal_data.modal_contributions.dtype, complex):
                # account for complex modal contributions from spectra, e.g. pLSCF
                mask = np.logical_and(
                    mask_pre, np.abs(self.modal_data.modal_contributions) >= MC_min)
            else:
                mask = np.logical_and(
                    mask_pre, self.modal_data.modal_contributions >= MC_min)
            self.masks['mask_MC'] = mask

        full_masks = []
        if df_max is not None:
            # rel freq diffs for each pole with all previous poles,
            # for all poles and orders results in 3d array
            # compare those rel freq diffs with df_max
            # and reduce 3d array to 2d array, by applying logical_or
            # along each poles axis (diff with all previous)
            mask_sf_all = np.logical_and(
                self.freq_diffs != 0, self.freq_diffs <= df_max)
            mask_sf_red = np.any(mask_sf_all, axis=2)
            # print(np.any(mask_sf_all))
            self.masks['mask_df'] = np.logical_and(mask_pre, mask_sf_red)
            full_masks.append(mask_sf_all)

        if dd_max is not None:
            mask_sd_all = np.logical_and(
                self.damp_diffs != 0, self.damp_diffs <= dd_max)
            mask_sd_red = np.any(mask_sd_all, axis=2)
            # print(np.any(mask_sd_all))
            self.masks['mask_dd'] = np.logical_and(mask_pre, mask_sd_red)
            full_masks.append(mask_sd_all)

        if dmac_max is not None:
            mask_sv_all = np.logical_and(
                self.MAC_diffs != 0, self.MAC_diffs <= dmac_max)
            mask_sv_red = np.any(mask_sv_all, axis=2)
            # print(np.any(mask_sv_all))
            self.masks['mask_dmac'] = np.logical_and(mask_pre, mask_sv_red)
            full_masks.append(mask_sv_all)

        # if dev_min is not None:
        #     logger.warning(
        #         'Eigenvalues are not available/implemented! Ignoring')
        #
        # if dmtn_min is not None:
        #     logger.warning(
        #         'Modal Transfer Norm is not yet implemented! Ignoring')

        # check if all stability criteria are satisfied for all current poles
        if full_masks:
            stable_mask_full = np.ones_like(full_masks[0])
            for mask in full_masks:
                stable_mask_full = np.logical_and(stable_mask_full, mask)
            stable_mask = np.any(stable_mask_full, axis=2)
        else:
            stable_mask = mask_pre
        # print(np.any(stable_mask))
        self.masks['mask_stable'] = None
        for mask_name, mask in self.masks.items():
            if mask_name == 'mask_autosel':
                continue
            if mask_name == 'mask_autoclear':
                continue
            if mask is not None:
                stable_mask = np.logical_and(stable_mask, mask)
        self.masks['mask_stable'] = stable_mask

        # compute the only-unstable-in-... masks
        self.nmasks = {
            name: np.logical_not(stable_mask) for name,
            mask in self.masks.items() if mask is not None}

        for nname, nmask in self.nmasks.items():
            if nname in [
                'mask_pre',
                'mask_stable',
                'mask_autosel',
                    'mask_autoclear']:
                continue
            for name, mask in self.masks.items():
                if mask is None or name in [
                    'mask_pre',
                    'mask_stable',
                    'mask_autosel',
                        'mask_autoclear']:
                    continue
                if name == nname:
                    nmask = np.logical_and(nmask, np.logical_not(mask))
                else:
                    nmask = np.logical_and(nmask, mask)
            self.nmasks[nname] = nmask

        self.nmasks['mask_stable'] = stable_mask
        self.nmasks['mask_pre'] = self.masks['mask_pre']
        self.nmasks['mask_autoclear'] = np.logical_not(
            self.masks.get('mask_autoclear', None))
        self.nmasks['mask_autosel'] = np.logical_not(
            self.masks.get('mask_autosel', None))

    def get_stabilization_mask(self, name):
        # print(name)
        mask = self.nmasks.get(name)

        if mask is None:
            mask = self.nmasks['mask_pre']
            logger.debug('Pre Mask is empty')

        return np.logical_not(mask)

    def get_max_f(self):
        if self.prep_signals is not None:
            return self.prep_signals.sampling_rate / 2
        elif isinstance(self.modal_data, PogerSSICovRef):
            return self.modal_data.sampling_rate / 2
        else:
            return float(np.amax(self.masked_frequencies))
        
    def get_frequencies(self):
        '''
        Returns
        -------
            frequencies: list
                Identified frequencies of all currently selected modes.
        '''
        selected_indices = self.select_modes

        frequencies = sorted([self.masked_frequencies[index[0], index[1]]
                              for index in selected_indices])
        return frequencies
    
    def get_selected_modal_values(self):
        '''
        Returns
        -------
            frequencies: list
                Identified frequencies of all currently selected modes.
        '''
        if not self.select_modes:
            return [np.array([]) for _ in range(12)]

        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        select_modes = self.select_modes
        selected_freq = [self.masked_frequencies[index]
                         for index in self.select_modes]
        select_modes = [x for (_, x) in sorted(
            zip(selected_freq, select_modes), key=lambda pair: pair[0])]

        selected_freq = [self.masked_frequencies[index]
                         for index in select_modes]
        selected_damp = [self.modal_data.modal_damping[index]
                         for index in select_modes]
        selected_order = [self.order_dummy[index]
                          for index in select_modes]
        if self.capabilities['ev']:
            selected_lambda = [self.modal_data.eigenvalues[index]
                         for index in select_modes]
        else:
            selected_lambda = None
        
        if self.capabilities['msh']:
            selected_MPC = [self.MPC_matrix[index]
                            for index in select_modes]
            selected_MP = [self.MP_matrix[index]
                           for index in select_modes]
            selected_MPD = [self.MPD_matrix[index]
                            for index in select_modes]
        else:
            selected_MPC = None
            selected_MP = None
            selected_MPD = None

        if self.capabilities['std']:
            selected_stdf = [self.modal_data.std_frequencies[index]
                             for index in select_modes]
            selected_stdd = [self.modal_data.std_damping[index]
                             for index in select_modes]
            selected_stdmsh = np.zeros(
                (self.modal_data.mode_shapes.shape[0], len(select_modes)), dtype=complex)
        else:
            selected_stdf = None
            selected_stdd = None
            selected_stdmsh = None

        if self.capabilities['MC']:
            selected_MC = [self.modal_data.modal_contributions[index]
                           for index in select_modes]
        else:
            selected_MC = None

        if self.capabilities['msh']:
            selected_modes = np.zeros(
                (self.modal_data.mode_shapes.shape[0], len(select_modes)), dtype=complex)

            for num, ind in enumerate(select_modes):
                row_index = ind[0]
                col_index = ind[1]
                mode_tmp = self.modal_data.mode_shapes[
                    :, col_index, row_index]
                if self.capabilities['std']:
                    std_mode = self.modal_data.std_mode_shapes[
                        :, col_index, row_index]

                # scaling of mode shape
                abs_mode_tmp = np.abs(mode_tmp)
                index_max = np.argmax(abs_mode_tmp)
                this_max = mode_tmp[index_max]

                if not self.capabilities['std']:
                    mode_tmp = mode_tmp / this_max

                selected_modes[:, num] = mode_tmp

                if self.capabilities['std']:
                    selected_stdmsh[:, num] = std_mode
        else:
            selected_modes = None
                    
        
        return selected_freq, selected_damp, selected_modes, selected_lambda,\
            selected_order, selected_MC,\
            selected_MPC, selected_MP, selected_MPD,\
            selected_stdf, selected_stdd, selected_stdmsh
    
    def get_modal_values(self, i):
        # needed for gui
        assert isinstance(i, (list, tuple))
        assert len(i) == 2
        assert i[0] <= self.modal_data.max_model_order
        assert i[1] <= self.num_solutions

        n = self.order_dummy[i]
        f = self.masked_frequencies[i]
        d = self.modal_data.modal_damping[i]

        if self.capabilities['msh']:
            mpc = self.MPC_matrix[i]
            mp = self.MP_matrix[i]
            mpd = self.MPD_matrix[i]
            MP_diffs = self.MP_diffs[i]
            # print(np.nonzero(MP_diffs)[0])
            if len(np.nonzero(MP_diffs)[0]) >= 1:
                dmp = np.min(MP_diffs[np.nonzero(MP_diffs)])
            else:
                dmp = 0
            MPD_diffs = self.MPD_diffs[i]
            if len(np.nonzero(MPD_diffs)[0]) >= 1:
                dmpd = np.min(MPD_diffs[np.nonzero(MPD_diffs)])
            else:
                dmpd = 0
        else:
            mpc = np.nan
            mp = np.nan
            mpd = np.nan
            dmp = np.nan
            dmpd = np.nan

        if self.capabilities['std']:
            stdf = self.modal_data.std_frequencies[i]
        else:
            stdf = np.nan

        if self.capabilities['std']:
            stdd = self.modal_data.std_damping[i]
        else:
            stdd = np.nan

        if self.capabilities['mtn']:
            mtn = np.nan
        else:
            mtn = np.nan

        if self.capabilities['MC']:
            MC = self.modal_data.modal_contributions[i]
        else:
            MC = np.nan

        if self.extra_func is not None:
            ex_1, ex_2 = self.extra_func(self.modal_data, i, True)
        else:
            ex_1, ex_2 = np.nan, np.nan

        return n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2

    def get_mode_shape(self, i):
        assert isinstance(i, (list, tuple))
        assert len(i) == 2
        assert i[0] <= self.modal_data.max_model_order
        assert i[1] <= self.num_solutions
        return self.modal_data.mode_shapes[:, i[1], i[0]]
    
    def add_mode(self, mode_ind):
        if mode_ind not in self.select_modes:
            self.select_modes.append(mode_ind)
        for callback in self.callbacks['add_mode']:
            callback(mode_ind, len(self.select_modes) - 1)
                
        return self.select_modes.index(mode_ind)
    
    def remove_mode(self, mode_ind):
        if mode_ind in self.select_modes:
            list_ind = self.select_modes.index(mode_ind)
            del self.select_modes[list_ind]
            for callback in self.callbacks['remove_mode']:
                callback(mode_ind, list_ind)
            return list_ind
        else:
            logger.warning(f'{mode_ind} not in self.select_modes')
            return None
    
    def save_state(self, fname):

        logger.info('Saving results to  {}...'.format(fname))

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state': self.state}

        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time

        if self.state >= 1:
            if self.capabilities['ev']:
                out_dict['self.lambda_diffs'] = np.array(self.lambda_diffs)
            out_dict['self.freq_diffs'] = np.array(self.freq_diffs)
            out_dict['self.damp_diffs'] = np.array(self.damp_diffs)

            if self.capabilities['msh']:
                out_dict['self.MAC_diffs'] = np.array(self.MAC_diffs)
                out_dict['self.MPD_diffs'] = np.array(self.MPD_diffs)
                out_dict['self.MP_diffs'] = np.array(self.MP_diffs)

                out_dict['self.MPC_matrix'] = np.array(self.MPC_matrix)
                out_dict['self.MP_matrix'] = np.array(self.MP_matrix)
                out_dict['self.MPD_matrix'] = np.array(self.MPD_matrix)

        if self.state >= 2:
            out_dict['self.order_range'] = self.order_range
            out_dict['self.d_range'] = self.d_range
            if self.capabilities['std']:
                out_dict['self.stdf_max'] = self.stdf_max
                out_dict['self.stdd_max'] = self.stdd_max
            if self.capabilities['msh']:
                out_dict['self.mpc_min'] = self.mpc_min
                out_dict['self.mpd_max'] = self.mpd_max
                out_dict['self.mtn_min'] = self.mtn_min

            out_dict['self.df_max'] = self.df_max
            out_dict['self.dd_max'] = self.dd_max
            if self.capabilities['msh']:
                out_dict['self.dmac_max'] = self.dmac_max
            out_dict['self.dev_min'] = self.dev_min
            if self.capabilities['mtn']:
                out_dict['self.dmtn_min'] = self.dmtn_min
            if self.capabilities['MC']:
                out_dict['self.MC_min'] = self.MC_min
            out_dict['self.masks'] = self.masks
            out_dict['self.nmasks'] = self.nmasks

        if self.capabilities['auto']:
            if self.state >= 3:
                out_dict['self.num_iter'] = self.num_iter
                out_dict['self.threshold'] = self.threshold
                out_dict['self.clear_ctr'] = self.clear_ctr
            if self.state >= 4:
                out_dict['self.use_stabil'] = self.use_stabil
                out_dict['self.proximity_matrix_sq'] = self.proximity_matrix_sq
                out_dict['self.cluster_assignments'] = self.cluster_assignments
            if self.state >= 5:
                out_dict['self.select_clusters'] = self.select_clusters
                out_dict['self.nr_poles'] = self.nr_poles
                out_dict['self.selection_cut_off'] = self.selection_cut_off

        out_dict['self.select_modes'] = self.select_modes

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, modal_data):
        logger.info('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = float(in_dict['self.state'])
        else:
            return

        setup_name = str(in_dict['self.setup_name'].item())
        # start_time = in_dict['self.start_time'].item()

        assert setup_name == modal_data.setup_name
        #assert start_time == modal_data.start_time

        stabil_data = cls(modal_data)

        if state >= 1:
            if stabil_data.capabilities['ev']:
                stabil_data.lambda_diffs = np.ma.array(
                    in_dict['self.lambda_diffs'])
            stabil_data.freq_diffs = np.ma.array(in_dict['self.freq_diffs'])
            stabil_data.damp_diffs = np.ma.array(in_dict['self.damp_diffs'])

            if stabil_data.capabilities['msh']:
                stabil_data.MAC_diffs = np.ma.array(in_dict['self.MAC_diffs'])
                stabil_data.MPD_diffs = np.ma.array(in_dict['self.MPD_diffs'])
                stabil_data.MP_diffs = np.ma.array(in_dict['self.MP_diffs'])

                stabil_data.MPC_matrix = np.ma.array(
                    in_dict['self.MPC_matrix'])
                stabil_data.MP_matrix = np.ma.array(in_dict['self.MP_matrix'])
                stabil_data.MPD_matrix = np.ma.array(
                    in_dict['self.MPD_matrix'])

        if state >= 2:
            stabil_data.order_range = tuple(in_dict['self.order_range'])
            stabil_data.d_range = tuple(in_dict['self.d_range'])
            if stabil_data.capabilities['std']:
                stabil_data.stdf_max = float(in_dict['self.df_max'])
                stabil_data.stdd_max = float(in_dict['self.stdd_max'])
            if stabil_data.capabilities['msh']:
                stabil_data.mpc_min = float(in_dict['self.mpc_min'])
                stabil_data.mpd_max = float(in_dict['self.mpd_max'])
                stabil_data.mtn_min = float(in_dict['self.mtn_min'])

            stabil_data.df_max = float(in_dict['self.df_max'])
            stabil_data.dd_max = float(in_dict['self.dd_max'])
            if stabil_data.capabilities['msh']:
                stabil_data.dmac_max = float(in_dict['self.dmac_max'])
            stabil_data.dev_min = float(in_dict['self.dev_min'])
            if stabil_data.capabilities['mtn']:
                stabil_data.dmtn_min = float(in_dict['self.dmtn_min'])
            if stabil_data.capabilities['MC']:
                stabil_data.MC_min = float(in_dict['self.MC_min'])

            stabil_data.masks = in_dict['self.masks'].item()
            stabil_data.nmasks = in_dict['self.nmasks'].item()

        if stabil_data.capabilities['auto']:
            if state >= 3:
                stabil_data.num_iter = int(in_dict['self.num_iter'])
                stabil_data.threshold = float(in_dict['self.threshold'])
                stabil_data.clear_ctr = in_dict['self.clear_ctr']
            if state >= 4:
                stabil_data.use_stabil = bool(in_dict['self.use_stabil'])
                stabil_data.proximity_matrix_sq = in_dict['self.proximity_matrix_sq']
                stabil_data.cluster_assignments = in_dict['self.cluster_assignments']
            if state >= 5:
                stabil_data.select_clusters = list(
                    in_dict['self.select_clusters'])
                stabil_data.nr_poles = list(in_dict['self.nr_poles'])
                stabil_data.selection_cut_off = float(
                    in_dict['self.selection_cut_off'])

        select_modes = [tuple(a)
                        for a in in_dict['self.select_modes']]
        frequencies = [stabil_data.masked_frequencies[index[0], index[1]]
                       for index in select_modes]
        stabil_data.select_modes = [
            x for _, x in sorted(zip(frequencies, select_modes))]

        stabil_data.state = state
        return stabil_data


class StabilCluster(StabilCalc):
    """ The automatic modal analysis done in three stages clustering.
    1st stage: values sorted according to their soft and hard criteria by a 2-means partitioning algorithm
    2nd stage: hierarchical clustering with automatic or user defined intercluster distance
    the automatic distance is based on the 'df', 'dd' and 'MAC' values from the centroids obtained in the first stage
    :math:`d = weight*df + 1 - weight*MAC + weight*dd`
    3rd stage: 2-means partitioning of the physical and spurious poles.

    E. Neu et al.

    1. Identify mode candidates from a large number of system orders.
        -> OMA Algorithm with n_max sufficiently high, i.e. number of mathematical modes should exceed the number pf physical modes at n <= n_max

    2. Remove as many mathematical modes as possible.

    (a) Remove certainly mathematical modes using hard validation criteria.
        Re(\\lambda_n)>= 0 or Im(\\lambda_n)==0-> remove conjugates in OMA algorithm
    (b) Split modes into consistent and non-consistent sets using k-means clustering.
        p_i = [d_lambda, d_f, d_zeta, 1-MAC, dMPD]
        power transformation eq 11
        h_Ti = ln(p_i)
        normalize:
        h_Ni = (h_Ti - mean(h_Ti)) / std(h_Ti)
        initialize centroids with (+std(h_Ni), -std(h_Ni))

    3. Divide the remaining modes into homogeneous sets using hierarchical clustering.

    (a) Derive cutoff distance from the probability distribution of the consistent modes.
            np.percentile(a,95)
    (b) Cluster the mode candidates based on a complex distance measure.
            average linkage / single linkage
    (c) Remove all but one mode from a single system order in one cluster.
            walk over each cluster and ensure each model order exists only once in the cluster, else remove the mode with a higher distance to the cluster center

    4. Remove the small sets, which typically consist of mathematical modes.

    (a) Reject sets that are smaller than a threshold derived from the largest set size.
        no recommendations given in paper (threshold 50 %)
    (b) Use outlier rejection to remove natural frequency and damping outliers.
        skip
    (c) Select a single mode representative from the remaining modes in each cluster.
        "multivariate" median

    """

    def __init__(self, modal_data, prep_signals=None):
        '''
        stab_* in %
        '''
        super().__init__(modal_data, prep_signals)

        assert self.capabilities['ev']

        self.num_iter = 20000

        self.weight_f = 1
        self.weight_MAC = 1
        self.weight_d = 1
        self.weight_lambda = 1
        self.threshold = None
        self.use_stabil = False

    @staticmethod
    def decompress_flat_mask(compress_mask, flat_mask):
        # takes a flat mask generated on compressed data and restore it to its
        # decompressed form
        decompressed_mask = np.ma.copy(compress_mask.ravel())

        flat_index = 0
        for mask_index in range(decompressed_mask.shape[0]):
            if decompressed_mask[mask_index]:
                continue
            if flat_index >= len(flat_mask):
                decompressed_mask[mask_index] = True
            else:
                decompressed_mask[mask_index] = flat_mask[flat_index]
            flat_index += 1

        return decompressed_mask.reshape(compress_mask.shape)

    def plot_mask(self, mask, save_path=None):
        plot.figure(tight_layout=1)
        od_mask = np.copy(self.order_dummy.mask)
        mf_mask = np.copy(self.masked_frequencies.mask)
        self.order_dummy.mask = self.get_stabilization_mask('mask_pre')
        self.masked_frequencies.mask = self.get_stabilization_mask('mask_pre')
        plot.scatter(
            self.masked_frequencies.compressed(),
            self.order_dummy.compressed(),
            marker='o',
            facecolors='none',
            edgecolors='grey',
            s=10)
        self.order_dummy.mask = mask
        self.masked_frequencies.mask = mask
        plot.scatter(
            self.masked_frequencies.compressed(),
            self.order_dummy.compressed(),
            marker='o',
            facecolors='none',
            edgecolors='black',
            s=10)
        self.order_dummy.mask = od_mask
        self.masked_frequencies.mask = mf_mask
        plot.ylim((0, 200))
        plot.xlim((0, self.prep_signals.sampling_rate / 2))
        plot.xlabel('Frequency [Hz]')
        plot.ylabel('Model Order ')
        plot.tight_layout()
        if save_path:
            plot.savefig(save_path + 'mask.pdf')
        else:
            plot.show()
            plot.pause(0.001)

    def automatic_clearing(self, num_iter=None):
        if self.state < 2:
            self.calculate_soft_critera_matrices()
        logger.info('Clearing physical modes automatically...')
        # 2-means clustering of all poles by all available criteria
        # algorithm minimizes euclidian distances

        if num_iter is not None:
            assert isinstance(num_iter, int)
            assert num_iter > 0
            self.num_iter = num_iter

        # represent all the vectors by their soft criteria :
        # [index, i, d_lambda, d_f, d_xi, dMAC, dMPD]

        mask_pre = np.ma.array(self.get_stabilization_mask('mask_pre'))
        # in a second run mask_pre is itself masked

        self.freq_diffs.mask = np.ma.nomask
        self.damp_diffs.mask = np.ma.nomask
        self.lambda_diffs.mask = np.ma.nomask
        if self.capabilities['msh']:
            self.MAC_diffs.mask = np.ma.nomask
            #self.MPD_diffs.mask = np.ma.nomask
            self.MP_diffs.mask = np.ma.nomask

        # assuming there are no frequencies equal within given precision
        mask_pre_3d = self.freq_diffs == 0

        soft_criteria_matrices = []
        for matrix in [self.lambda_diffs, self.freq_diffs, self.damp_diffs]:
            matrix.mask = mask_pre_3d
            soft_criteria_matrices.append(matrix.min(axis=2))

        if self.capabilities['msh']:
            self.MAC_diffs.mask = mask_pre_3d
            soft_criteria_matrices.append(self.MAC_diffs.min(axis=2))

            #self.MPD_diffs.mask = mask_pre_3d
            # soft_criteria_matrices.append(self.MPD_diffs.min(axis=2))

            self.MP_diffs.mask = mask_pre_3d
            soft_criteria_matrices.append(self.MP_diffs.min(axis=2))

        for matrix in soft_criteria_matrices:
            matrix.mask = mask_pre

        # flatten unmasked values and remove first two model orders
        compressed_matrices = [matrix[2:, :].compressed()
                               for matrix in soft_criteria_matrices]

        # dlambda, df, dd, dMAC, dMPD, stacked as list of size (order,
        # num_modes)
        all_poles = np.vstack(compressed_matrices).T

        # transform distribution (weibull like) to logarithmic scale (resembles
        # normal distribution)
        all_poles = np.log(all_poles)

        # whitening (scale to unit variance) significantly improves
        # convergence rates of the kmeans algorithm
        mean_all_poles = np.mean(all_poles, axis=0)
        std_all_poles = np.std(all_poles, axis=0)
        all_poles -= mean_all_poles
        all_poles /= std_all_poles

        std_dev = np.std(all_poles, axis=0)
        ideal_physical_values = -std_dev
        ideal_spurious_values = std_dev

        # the k-means algorithm is sensitive to the initial starting
        # values in order to converge to a solution
        # therefore two starting attempts are introduced
        ctr_init = np.array([ideal_physical_values,
                             ideal_spurious_values])

        # masked arrays are not supported by scipy's kmeans algorithm
        # all_poles an M by N array where the rows are observation vectors
        self.clear_ctr, idx = scipy.cluster.vq.kmeans2(
            all_poles, ctr_init, self.num_iter)

        logger.info(
            'Possibly physical poles 1st stage: {0} Spurious poles 1st stage: {1}'.format(
                collections.Counter(idx)[0],
                collections.Counter(idx)[1]))

        # add unmasked values of the first two model orders that were
        # previously not considered
        mask_pre.mask = np.ma.nomask
        new_idx = np.hstack(
            (np.ones(np.sum(np.logical_not(mask_pre[:2, :]))), idx))
        mask_autoclear = self.decompress_flat_mask(mask_pre, new_idx)

        # re-apply mask_pre, should not be necessary
        mask_autoclear = np.logical_or(mask_autoclear, mask_pre)

        # apply the hard validation criteria
#         mask_autoclear = np.logical_or(
#             mask_autoclear, self.modal_data.modal_damping < 0.001)
#         mask_autoclear = np.logical_or(
#             mask_autoclear, self.modal_data.modal_damping > 20)

        # compute the threshold as the 95th percentile of
        # P(threshold > d_lambda + d_MAC) = 0.95
        soft_criteria_matrices[0].mask = np.ma.nomask
        soft_criteria_matrices[3].mask = np.ma.nomask
        distance_mat = soft_criteria_matrices[0] + soft_criteria_matrices[3]
        distance_mat.mask = mask_autoclear
        self.threshold = np.percentile(distance_mat.compressed(), q=95)

        self.masks['mask_autoclear'] = mask_autoclear
        self.update_stabilization_masks()

        self.state = 3

    def automatic_classification(self, threshold=None, use_stabil=False):
        if self.state < 3 and not use_stabil:
            self.automatic_clearing()
        logger.info('Classifying physical modes automatically...')

        if use_stabil:
            mask_autoclear = self.get_stabilization_mask('mask_stable')
            #self.masks['mask_autoclear'] = mask_autoclear
            # self.update_stabilization_masks()
            # print(123)
        else:
            mask_autoclear = self.get_stabilization_mask('mask_autoclear')

        self.use_stabil = use_stabil

        if threshold is not None:
            assert isinstance(threshold, int)
            self.threshold = threshold
        if self.threshold is None:
            self.freq_diffs.mask = np.ma.nomask
            mask_pre_3d = self.freq_diffs == 0
            self.lambda_diffs.mask = mask_pre_3d
            self.MAC_diffs.mask = mask_pre_3d
            distance_mat = self.lambda_diffs.min(axis=2)\
                + self.MAC_diffs.min(axis=2)
            distance_mat.mask = mask_autoclear
            self.threshold = np.percentile(distance_mat.compressed(), q=95)
        # print(self.threshold)

        length_mat = np.product(mask_autoclear.shape) \
            - np.sum(mask_autoclear)

        self.masked_lambda.mask = mask_autoclear
        lambda_compressed = self.masked_lambda.compressed()
        self.masked_lambda.mask = np.ma.nomask

        dim0, dim1 = mask_autoclear.shape
        mode_shapes_compressed = np.zeros(
            (self.modal_data.mode_shapes.shape[0], length_mat),
            dtype=np.complex128
        )
        
        n = 0
        for i in range(dim0):
            for j in range(dim1):
                if mask_autoclear[i, j]:
                    continue
                this_msh = self.modal_data.mode_shapes[:, j, i]
                mode_shapes_compressed[:, n] = this_msh
                n += 1

        l = len(lambda_compressed)
        # print(l)
        div_lambda = np.maximum(
            np.repeat(np.expand_dims(np.abs(lambda_compressed), axis=1),
                      lambda_compressed.shape[0], axis=1),
            np.repeat(np.expand_dims(np.abs(lambda_compressed), axis=0),
                      lambda_compressed.shape[0], axis=0)
        )

        lambda_proximity_matrix = np.abs(
            lambda_compressed - lambda_compressed.reshape((l, 1))) / div_lambda

        mac_proximity_matrix = 1 - \
            calculateMAC(mode_shapes_compressed, mode_shapes_compressed)
        print(lambda_proximity_matrix.shape, mac_proximity_matrix.shape, np.sum(mask_autoclear), n)
        proximity_matrix = self.weight_lambda * lambda_proximity_matrix \
            + self.weight_MAC * mac_proximity_matrix

        # correct round off errors
        proximity_matrix[
            proximity_matrix < np.finfo(proximity_matrix.dtype).eps] = 0

        self.proximity_matrix_sq = scipy.spatial.distance.squareform(
            proximity_matrix, checks=False)
        linkage_matrix = scipy.cluster.hierarchy.linkage(
            self.proximity_matrix_sq, method='average')
        self.cluster_assignments = scipy.cluster.hierarchy.fcluster(
            linkage_matrix, self.threshold, criterion='distance')

        for clusternr in range(1, max(self.cluster_assignments) + 1):
            flat_poles_ind = self.cluster_assignments != clusternr + 1
            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.order_dummy.mask = mask
            for order in range(self.modal_data.max_model_order):
                if np.sum(self.order_dummy == order) > 1:
                    logger.debug('Double Model Order: ', self.order_dummy[order, :])

        self.order_dummy.mask = np.ma.nomask

        logger.info('Number of classified clusters: {}'.format(
            max(self.cluster_assignments)))
        self.state = 4

    def automatic_selection(self, number=0):

        if self.state < 4:
            self.automatic_classification()

        # count clusters with more than a fraction of the number of elements of
        # the largest cluster and add that many zero size clusters
        poles = []

        for cluster_ in range(1, 1 + max(self.cluster_assignments)):
            poles.append(np.where(self.cluster_assignments == cluster_))

        nr_poles = [len(a[0]) for a in poles]
        max_nr = max(nr_poles)
        # enlarge list of poles to improve clustering
#         for nr in nr_poles:
#             if nr > max_nr * fraction:
#                 nr_poles.append(0)
#             elif nr < max_nr * fraction:
#                 continue
        nr_poles = np.array(nr_poles, dtype=np.float64)

        if number == 0:
            # 2-means clustering of the number-of-poles, return indices;
            # split into two clusters
            _, select_clusters = scipy.cluster.vq.kmeans2(
                np.array(nr_poles, dtype=np.float64),
                np.array([max_nr, 1e-12]), self.num_iter)
        else:
            meta_list = list(enumerate(nr_poles))

            sorted_meta_list = sorted(meta_list, key=itemgetter(1),
                                      reverse=True)
            select_clusters = [1 for _ in nr_poles]

            for i in range(number):
                ind = sorted_meta_list[i][0]
                select_clusters[ind] = 0

        logger.info('Number of physical modes: {0}'.format(
            collections.Counter(select_clusters)[0]))
        self.select_clusters = select_clusters
        self.nr_poles = nr_poles

        self.selection_cut_off = np.Infinity
        for i, b in zip(self.nr_poles, self.select_clusters):
            if not b:
                self.selection_cut_off = min(i - 1, self.selection_cut_off)
        logger.info('Minimum number of elements in retained clusters: {}'.format(
            self.selection_cut_off))

        if self.use_stabil:
            mask_autoclear = self.get_stabilization_mask('mask_stable')
            #mask_autoclear = self.masks['mask_autoclear']
        else:
            mask_autoclear = self.masks['mask_autoclear']

        self.MAC_diffs.mask = self.MAC_diffs == 0
        MAC_diffs = self.MAC_diffs.min(axis=2)
        self.MAC_diffs.mask = np.ma.nomask
        if 'mask_autosel' not in self.masks:
            self.masks['mask_autosel'] = []
        for clusternr, inout in enumerate(select_clusters):

            if inout:
                continue

            flat_poles_ind = self.cluster_assignments != clusternr + 1

            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.masks['mask_autosel'].append(np.ma.copy(mask))

            # remove outermost values in all criteria, until only the
            # "multi-variate median" is left this pole is selected as the
            # representative solution for this cluster
            num_poles_left = np.product(mask.shape) - np.sum(mask)

            #print(clusternr, num_poles_left)

            while num_poles_left > 1:
                ind = []
                for matrix, target in zip(
                    [self.masked_frequencies, self.masked_damping, ],
                        [np.ma.median, np.ma.median, ]):

                    matrix.mask = mask

                    val = target(matrix)
                    min_ = np.min(matrix)
                    max_ = np.max(matrix)

                    if val - min_ <= max_ - val:
                        ind.append(np.where(matrix == max_))
                    else:
                        ind.append(np.where(matrix == min_))

                for i in range(min(len(ind), num_poles_left - 1)):
                    this_ind = ind[i]
                    mask[this_ind] = True

                num_poles_left = np.product(mask.shape) - np.sum(mask)

                #print(clusternr, num_poles_left, len(ind))

            select_mode = np.where(np.logical_not(mask))
            # self.select_modes.append((select_mode[0][0], select_mode[1][0]))
            self.add_mode((select_mode[0][0], select_mode[1][0]))
            if self.select_callback is not None:
                #                 for matrix in [self.masked_frequencies, self.masked_damping,
                #                            MAC_diffs, self.MPC_matrix, self.MPD_matrix]:
                #
                #                     matrix.mask = np.ma.nomask
                self.select_callback(self.select_modes[-1])

        for matrix in [self.masked_frequencies, self.masked_damping,
                       MAC_diffs, self.MPC_matrix, self.MPD_matrix]:

            matrix.mask = np.ma.nomask

        self.state = 5

    def plot_clearing(self, save_path=None):

        mask_autoclear = self.masks['mask_autoclear']
        mask_pre = self.get_stabilization_mask('mask_pre')
        self.plot_mask(mask_autoclear, save_path)

        self.freq_diffs.mask = self.freq_diffs == 0
        freq_diffs = self.freq_diffs.min(axis=2)
        self.freq_diffs.mask = np.ma.nomask

        self.MAC_diffs.mask = self.MAC_diffs == 0
        MAC_diffs = self.MAC_diffs.min(axis=2)
        self.MAC_diffs.mask = np.ma.nomask

        self.damp_diffs.mask = self.damp_diffs == 0
        damp_diffs = self.damp_diffs.min(axis=2)
        self.damp_diffs.mask = np.ma.nomask

        crits = [freq_diffs, damp_diffs, MAC_diffs,
                 self.MPC_matrix, self.MPD_matrix]
        labels = ['df', 'dd', 'MAC', 'MPC', 'MPD']

        new_crits = []

        for j, b in enumerate(crits):
            new_crits.append(b)
            for i, a in enumerate(new_crits):
                if a is b:
                    continue
                plot.figure(tight_layout=1)

                a.mask = mask_autoclear
                b.mask = mask_autoclear

                plot.plot(a.compressed(), b.compressed(), ls='', marker=',')
                plot.plot(
                    np.mean(a), np.mean(b), ls='', marker='d', color='black')

                a.mask = mask_pre
                b.mask = mask_pre
                plot.plot(
                    a.compressed(),
                    b.compressed(),
                    ls='',
                    marker=',',
                    color='grey')
                plot.plot(
                    np.mean(a), np.mean(b), ls='', marker='d', color='grey')

                labela = labels[i]
                labelb = labels[j]

                plot.xlabel(labela)
                plot.ylabel(labelb)

                plot.xlim((0, 1))
                plot.ylim((0, 1))

                if save_path is not None:
                    plot.savefig(
                        save_path + 'clear_{}_{}.pdf'.format(labela, labelb))
                else:
                    # print('show')
                    plot.show()
                    plot.pause(0.01)
        for crit in crits:
            crit.mask = np.ma.nomask

    def plot_classification(self, save_path=None):
        rel_matrix = scipy.cluster.hierarchy.linkage(
            self.proximity_matrix_sq, method='average')
        lvs = scipy.cluster.hierarchy.leaves_list(rel_matrix)

        def _llf(_):
            if len(lvs) > 500:
                if (np.where(id == lvs)[0][0] % 100 == 0):
                    return str(np.where(id == lvs)[0][0])
                else:
                    return str('')
            else:
                if (np.where(id == lvs)[0][0] % 10 == 0):
                    return str(np.where(id == lvs)[0][0])
                else:
                    return str('')
        fig = plot.figure(tight_layout=1)
        ax = fig.add_subplot(111)
        scipy.cluster.hierarchy.dendrogram(
            rel_matrix,
            leaf_label_func=_llf,
            color_threshold=self.threshold,
            leaf_font_size=16,
            leaf_rotation=40)
        # ax = plot.gca()
        ax.set_xlabel('Mode number [-]')
        ax.set_ylabel('Distance [-]')
        ax.axhline(self.threshold, c='r', ls='--', linewidth=3)
        plot.tight_layout()
        if save_path is not None:
            plot.savefig(save_path + 'dendrogram.pdf')
        else:
            # print('show')
            plot.show()
            plot.pause(0.001)

    def plot_selection(self, save_path=None):
        """ Plot relevant results of the clustering."""

        plot.figure(tight_layout=1)

        in_poles = list(self.nr_poles[self.nr_poles >= self.selection_cut_off])
        in_poles.sort(reverse=True)
        out_poles = self.nr_poles[self.nr_poles < self.selection_cut_off]
        out_poles = list(out_poles[out_poles > 0])
        out_poles.sort(reverse=True)
        plot.bar(range(len(in_poles)), in_poles, facecolor='red',
                 edgecolor='none', align='center',)
        # print(list(range(len(in_poles),len(self.nr_poles))),out_poles)
        plot.bar(
            range(
                len(in_poles),
                len(in_poles) +
                len(out_poles)),
            out_poles,
            facecolor='blue',
            edgecolor='none',
            align='center',
        )

        plot.xlim((0, len(self.nr_poles)))
        plot.tight_layout()

        if save_path is not None:
            plot.savefig(save_path + 'cluster_sizes.pdf')
        else:
            plot.show()
            plot.pause(0.001)

        fig = plot.figure(tight_layout=1)
        ax1 = fig.add_subplot(211)

        mask_autoclear = self.masks['mask_autoclear']
        mask_pre = self.get_stabilization_mask('mask_pre')
        mask_pre_ = np.logical_not(
            np.logical_and(np.logical_not(mask_pre), mask_autoclear))

        self.order_dummy.mask = mask_pre_
        self.masked_frequencies.mask = mask_pre_
        ax1.scatter(
            self.masked_frequencies.compressed(),
            self.order_dummy.compressed(),
            marker='o',
            facecolors='none',
            edgecolors='grey',
            s=10,
            label='pole')

        self.order_dummy.mask = mask_autoclear
        self.masked_frequencies.mask = mask_autoclear
        ax1.scatter(
            self.masked_frequencies.compressed(),
            self.order_dummy.compressed(),
            marker='o',
            facecolors='none',
            edgecolors='black',
            s=10,
            label='stable pole')

        self.order_dummy.mask = np.ma.nomask
        self.masked_frequencies.mask = np.ma.nomask

        ax1.autoscale_view(tight=True)
        #ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Model order [-]')
        ax1.set_title('Stabilization Diagram')
        ax1.set_ylim((0, 200))

        for mask in self.masks['mask_autosel']:
            self.masked_frequencies.mask = mask
            plot.axvspan(
                self.masked_frequencies.min(),
                self.masked_frequencies.max(),
                facecolor='blue',
                alpha=.3,
                edgecolor='none')
            #print(self.masked_frequencies.min(), self.masked_frequencies.max(),np.ma.mean(self.masked_frequencies))

        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        for mode in self.select_modes:
            f = self.modal_data.modal_frequencies[mode]
            n = self.order_dummy[mode]
            ax1.scatter(
                f, n, facecolors='none', marker='o', edgecolors='red', s=10)

        num_poles = []
        fpoles = []
        for clusternr in range(1, 1 + max(self.cluster_assignments)):

            flat_poles_ind = self.cluster_assignments != clusternr
            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.masked_frequencies.mask = mask
            num_poles.append(np.product(mask.shape) - np.sum(mask))
            # print(np.ma.mean(self.masked_frequencies))
            fpoles.append(np.ma.mean(self.masked_frequencies))

        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.bar(fpoles, num_poles, width=0.01,
                align='center', edgecolor='none')

        ax2.axhline(self.selection_cut_off, c='r', ls='--', linewidth=2)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Nr. of elements')
        ax2.set_title('Clusters')
        plot.tight_layout()
        plot.xlim((0, self.prep_signals.sampling_rate / 2))
        #plot.savefig('Main_plot_clusters_' + self.timestamp + '.' + self.format_plot, format=self.format_plot)
        if save_path is not None:
            plot.savefig(save_path + 'select_clusters.pdf')
        else:
            # print('show')
            plot.show()
            plot.pause(0.001)
        # plot.show(block=False)

    def return_results(self):

        all_f = []
        all_d = []
        all_n = []
        all_std_f = []
        all_std_d = []
        all_MPC = []
        all_MPD = []
        all_MP = []
        all_msh = []
        all_MC = []

        # for select_mode, mask in zip(self.select_modes,
        # self.masks['mask_autosel']):
        for select_mode in self.select_modes:

            n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2 = self.get_modal_values(
                select_mode)
            msh = self.get_mode_shape(select_mode)

            all_n.append(n)
            all_f.append(f)
            all_std_f.append(stdf)
            all_d.append(d)
            all_std_d.append(stdd)
            all_MPC.append(mpc)
            all_MP.append(mp)
            all_MPD.append(mpd)
            all_MC.append(MC)
            all_msh.append(msh)

            continue

#             self.masked_frequencies.mask = np.ma.nomask
#             select_f = self.masked_frequencies[select_mode]
#
#             self.masked_damping.mask = np.ma.nomask
#             select_d = self.masked_damping[select_mode]
#
#             select_order = self.order_dummy[select_mode]
#
#             if self.capabilities['msh']:
#                 self.MPC_matrix.mask = np.ma.nomask
#                 select_MPC = self.MPC_matrix[select_mode]
#
# #                 self.MP_matrix.mask = np.ma.nomask
# #                 select_MP = self.MP_matrix[select_mode]
#
#                 self.MPD_matrix.mask = np.ma.nomask
#                 select_MPD = self.MPD_matrix[select_mode]
#
#                 select_msh = self.modal_data.mode_shapes[
#                     :, select_mode[1], select_mode[0]]
#
#                 # scaling of mode shape
#                 #select_msh /= select_msh[np.argmax(np.abs(select_msh))][0]
#             if self.capabilities['MC']:
#                 self.modal_data.modal_contributions[select_mode]
#
#             all_f.append(float(select_f))
#             all_d.append(float(select_d))
#             all_n.append(float(select_order))
#             all_MPC.append(float(select_MPC))
# #             all_MP.append(float(select_MP))
#             all_MPD.append(float(select_MPD))
#             all_msh.append(select_msh)

        return np.array(all_n), np.array(all_f), np.array(all_std_f), np.array(all_d), np.array(
            all_std_d), np.array(all_MPC), np.array(all_MP), np.array(all_MPD), np.array(all_MC), np.array(all_msh),


class StabilPlot(object):

    def __init__(self, stabil_calc, fig=None):
        '''
        stab_* in %
        '''
        super().__init__()

        assert isinstance(stabil_calc, StabilCalc)
        self.stabil_calc = stabil_calc
        if fig is None:
            self.fig = Figure(facecolor='white')#, dpi=100, figsize=(16, 12))
            self.fig.set_tight_layout(True)
            # canvas = FigureCanvasBase(self.fig)
        else:
            self.fig = fig
            
        self.ax = self.fig.add_subplot(111)

        # self.ax2 = self.ax.twinx()
        # self.ax2.set_navigate(False)


        # if self.fig.canvas:
        if False:
            self.init_cursor()
        else:
            self.cursor = None
        marker_obj_1 = MarkerStyle('o')
        path_1 = marker_obj_1.get_path().transformed(
            marker_obj_1.get_transform())
        marker_obj_2 = MarkerStyle('+')
        path_2 = marker_obj_2.get_path().transformed(
            marker_obj_2.get_transform())
        path_stab = Path.make_compound_path(path_1, path_2)

        marker_obj_2 = MarkerStyle('x')
        path_2 = marker_obj_2.get_path().transformed(
            marker_obj_2.get_transform())
        path_auto = Path.make_compound_path(path_1, path_2)

        fp = FontProperties(family='monospace', weight=0, size='large')

        self.psd_plot = []

        self.stable_plot = {
            'plot_pre': None,
            # 'plot_ad':    None,
            # 'plot_df':    None,
            # 'plot_dd':    None,
            'plot_stable': None,
        }

        self.colors = {
            'plot_pre': 'grey',
            # 'plot_ad':    'grey',
            # 'plot_df':    'black',
            # 'plot_dd':    'black',
            'plot_stable': 'black',
        }

        self.markers = {
            'plot_pre': 'o',
            # 'plot_ad':    TextPath((-2, -4), '\u00b7 d', prop=fp, size=10),
            # 'plot_df':    TextPath((-2, -4), '\u00b7 f', prop=fp, size=10),
            # 'plot_dd':    TextPath((-2, -4), '\u00b7 d', prop=fp, size=10),
            'plot_stable': path_stab,
            #
        }

        self.labels = {
            'plot_pre': 'all poles',
            # 'plot_ad':    'damping criterion',
            # 'plot_df':    'unstable in frequency',
            # 'plot_dd':    'unstable in damping',
            'plot_stable': 'stable poles',

        }

        if self.stabil_calc.capabilities['std']:
            self.stable_plot['plot_stdf'] = None  # uncertainty frequency
            self.stable_plot['plot_stdd'] = None  # uncertainty damping

            self.colors['plot_stdf'] = 'grey'
            self.colors['plot_stdd'] = 'grey'

            self.labels['plot_stdf'] = 'uncertainty bounds frequency criterion'
            self.labels['plot_stdd'] = 'uncertainty bounds damping criterion'

            self.markers['plot_stdf'] = 'd'
            self.markers['plot_stdd'] = 'd'
        if self.stabil_calc.capabilities['msh']:
            # absolute modal phase collineratity
            self.stable_plot['plot_mpc'] = None
            # absolute mean phase deviation
            self.stable_plot['plot_mpd'] = None
            self.stable_plot['plot_dmac'] = None  # difference mac

            #self.colors['plot_mpc'] = 'grey'
            #self.colors['plot_mpd']=  'grey'
            #self.colors['plot_dmac']=  'black'

            #self.labels['plot_mpc']=   'modal phase collinearity criterion'
            #self.labels['plot_mpd']=   'mean phase deviation criterion'
            #self.labels['plot_dmac']=  'unstable in mac'

            #self.markers['plot_mpc']=   TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            #self.markers['plot_mpd']=   TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            #self.markers['plot_dmac']=  TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)

        if self.stabil_calc.capabilities['auto']:
            # auto clearing by 2Means Algorithm
            self.stable_plot['plot_autoclear'] = None
            # autoselection by 2 stage hierarchical clustering
            self.stable_plot['plot_autosel'] = None

            self.colors['plot_autoclear'] = 'black'
            self.colors['plot_autosel'] = 'rainbow'

            self.labels['plot_autoclear'] = 'autoclear poles'
            self.labels['plot_autosel'] = 'autoselect poles'

            self.markers['plot_autoclear'] = path_auto
            self.markers['plot_autosel'] = 'o'

        if self.stabil_calc.capabilities['MC']:
            # absolute modal error contribution
            self.stable_plot['plot_MC'] = None
            self.colors['plot_MC'] = 'grey'

            self.labels['plot_MC'] = 'modal error contribution criterion'

            self.markers['plot_MC'] = 'x'

        if self.stabil_calc.capabilities['mtn']:
            # difference modal transfer norm
            self.stable_plot['plot_dmtn'] = None
            self.stable_plot['plot_mtn'] = None  # absolute modal transfer norm

            self.colors['plot_dmtn'] = 'black'
            self.colors['plot_mtn'] = 'grey'

            self.labels['plot_mtn'] = 'modal transfer norm criterion'
            self.labels['plot_dmtn'] = 'unstable in modal transfer norm'

            self.markers['plot_mtn'] = '>'
            self.markers['plot_dmtn'] = '>'
        if False:
            self.stable_plot['plot_dev'] = None  # difference eigenvalue

            self.colors['plot_dev'] = 'grey'

            self.labels['plot_dev'] = 'unstable in eigenvalue'

            self.markers['plot_dev'] = TextPath(
                (-2, -4), '\u00b7 \u03bb', prop=fp, size=10),

        self.zorders = {key: key != 'plot_pre' for key in self.labels.keys()}
        self.zorders['plot_autosel'] = 2
        self.sizes = {key: 30 for key in self.labels.keys()}

        self.prepare_diagram()
        
        
        
        # that list should eventually be replaced by a matplotlib.collections
        # collection
        self.scatter_objs = [None for _ in self.stabil_calc.select_modes]
        
        self.stabil_calc.add_callback('add_mode', self.add_mode)
        self.stabil_calc.add_callback('remove_mode', self.remove_mode)
        
        if stabil_calc.select_modes:
            for mode in stabil_calc.select_modes:
                list_ind = self.stabil_calc.select_modes.index(mode)
                self.add_mode(mode,list_ind)

    def init_cursor(self, visible=True):
        
        self.cursor = DataCursor(
            ax=self.ax,
            horizOn=visible, vertOn=visible,
            order_data=self.stabil_calc.order_dummy,
            f_data=self.stabil_calc.masked_frequencies,
            datalist=self.stabil_calc.select_modes,
            color='black', useblit=True)
        
        self.fig.canvas.mpl_connect(
            'button_press_event', self.mode_selected)
        self.fig.canvas.mpl_connect(
            'resize_event', self.cursor.fig_resized)
        
        return self.cursor

    def prepare_diagram(self):

        self.ax.set_ylim((0, self.stabil_calc.modal_data.max_model_order))
        self.ax.locator_params(
            'y',
            tight=True,
            nbins=self.stabil_calc.modal_data.max_model_order //
            5)
        x_lims = (0, self.stabil_calc.get_max_f())
        self.ax.set_xlim(x_lims)
        self.ax.autoscale_view(tight=True)
        self.ax.set_xlabel('Frequency [Hz]')
        self.ax.set_ylabel('Model Order')

    def update_stabilization(self, **criteria):

        # print(criteria)
        # recalculate stabilization masks
        self.stabil_calc.update_stabilization_masks(**criteria)

        # update stabil plots if values have changed
        # if 'd_range' in criteria:
        #    self.plot_stabil('plot_ad')
        if 'stdf_max' in criteria and self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdf')
        if 'stdd_max' in criteria and self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdd')
        # if 'mpc_min' in criteria and self.stabil_calc.capabilities['msh']:
        #    self.plot_stabil('plot_mpc')
        # if 'mpd_max' in criteria and self.stabil_calc.capabilities['msh']:
        #    self.plot_stabil('plot_mpd')
        if 'mtn_min' in criteria and self.stabil_calc.capabilities['mtn']:
            self.plot_stabil('plot_mtn')
        # if 'df_max' in criteria:
        #    self.plot_stabil('plot_df')
        # if 'dd_max' in criteria:
        #    self.plot_stabil('plot_dd')
        # if 'dmac_max' in criteria and self.stabil_calc.capabilities['msh']:
        #    self.plot_stabil('plot_dmac')
        if 'dev_min' in criteria and False:
            self.plot_stabil('plot_dev')
        if 'dmtn_min' in criteria and self.stabil_calc.capabilities['mtn']:
            self.plot_stabil('plot_dmtn')
        if 'MC_min' in criteria and self.stabil_calc.capabilities['MC']:
            self.plot_stabil('plot_MC')

        self.plot_stabil('plot_pre')
        self.plot_stabil('plot_stable')

        if self.stabil_calc.capabilities['auto']:
            if self.stabil_calc.state >= 3 and not self.stabil_calc.use_stabil:
                self.plot_stabil('plot_autoclear')
            if self.stabil_calc.state >= 5:
                self.plot_stabil('plot_autosel')

        if self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdf')
        # update the cursors snap mask
        if self.cursor:
            cursor_name_mask = self.cursor.name_mask
            cursor_mask = self.stabil_calc.get_stabilization_mask(
                cursor_name_mask)
            self.cursor.set_mask(cursor_mask, cursor_name_mask)

    def plot_stabil(self, name):
        # print(name)
        color = self.colors[name]
        marker = self.markers[name]
        #print(marker, name)
        zorder = self.zorders[name]
        size = self.sizes[name]
        label = self.labels[name]

        #print(name, color)

        if name == 'plot_autosel':
            if self.stable_plot[name] is not None:
                for plot in self.stable_plot[name]:
                    plot.remove()

            visibility = True
            masks = self.stabil_calc.masks['mask_autosel']
            # @UndefinedVariable
            colors = list(
                matplotlib.cm.gist_rainbow(
                    np.linspace(
                        0, 1, len(masks))))  # @UndefinedVariable
            shuffle(colors)
            self.stable_plot[name] = []
            for color, mask in zip(colors, masks):
                self.stabil_calc.masked_frequencies.mask = mask
                self.stabil_calc.order_dummy.mask = mask
                self.stable_plot[name].append(
                    self.ax.scatter(
                        self.stabil_calc.masked_frequencies.compressed(),
                        self.stabil_calc.order_dummy.compressed(),
                        zorder=zorder,
                        facecolors=color,
                        edgecolors='none',
                        marker=marker,
                        alpha=0.4,
                        s=size,
                        label=label,
                        visible=visibility))

        elif name == 'plot_stdf':

            if self.stable_plot[name] is not None:
                try:
                    visibility = self.stable_plot[name][1][0].get_visible()
                    self.stable_plot[name][1][0].remove()
                    self.stable_plot[name][1][1].remove()
                    self.stable_plot[name][2][0].remove()
                except IndexError:
                    visibility = True

            else:
                visibility = True
            mask = self.stabil_calc.get_stabilization_mask('mask_stable')
            self.stabil_calc.masked_frequencies.mask = mask
            self.stabil_calc.order_dummy.mask = mask
            if self.stabil_calc.capabilities['std']:
                std_frequencies = np.ma.array(
                    self.stabil_calc.modal_data.std_frequencies)
                std_frequencies.mask = mask

                # standard error
                num_blocks = self.stabil_calc.modal_data.num_blocks
                std_error = std_frequencies.compressed() / np.sqrt(num_blocks)

                # 95 % confidence interval -> student t (tabulated percentage
                # points) * std_error (approx 2* std_error)
                self.stable_plot[name] = self.ax.errorbar(
                    self.stabil_calc.masked_frequencies.compressed(),
                    self.stabil_calc.order_dummy.compressed(),
                    xerr=scipy.stats.t.ppf(
                        0.975,
                        num_blocks) * std_error,
                    zorder=zorder,
                    fmt='none',
                    ecolor=color,
                    label=label,
                    visible=visibility)

        else:
            if self.stable_plot[name] is not None:
                visibility = self.stable_plot[name].get_visible()
                self.stable_plot[name].remove()
            else:
                visibility = True
            mask = self.stabil_calc.get_stabilization_mask(
                name.replace('plot', 'mask'))

            self.stabil_calc.masked_frequencies.mask = mask
            self.stabil_calc.order_dummy.mask = mask

            self.stable_plot[name] = self.ax.scatter(
                self.stabil_calc.masked_frequencies.compressed(),
                self.stabil_calc.order_dummy.compressed(),
                zorder=zorder,
                facecolors='none',
                edgecolors=color,
                marker=marker,
                s=size,
                label=label,
                visible=visibility)

        # self.ax.set_yticks([])

        mask_stable = self.stabil_calc.get_stabilization_mask('mask_pre')
        self.stabil_calc.masked_frequencies.mask = mask_stable
        self.stabil_calc.order_dummy.mask = mask_stable

        self.fig.canvas.draw_idle()

    def show_MC(self, b=False):

        if b:
            if not self.stabil_calc.capabilities['MC']:
                logger.warning('Modal contributions are not computed and cannot be displayed.')
                return
            ylim = self.fig.axes[0].get_ylim()
            if len(self.fig.axes) < 2:
                self.fig.add_subplot(1, 2, 2, sharey=self.fig.axes[0])
                gs = matplotlib.gridspec.GridSpec(
                    1, 2, width_ratios=(6, 1), wspace=0.01, hspace=0)
                self.fig.axes[0].set_subplotspec(gs[0])
                self.fig.axes[1].set_subplotspec(gs[1])
            ax = self.fig.axes[1]
            MCs = np.zeros((self.stabil_calc.modal_data.max_model_order))
            for order in range(self.stabil_calc.modal_data.max_model_order):
                # abs used for complex modal contributions (pLSCF)
                MCs[order] = np.abs(np.sum(
                    self.stabil_calc.modal_data.modal_contributions[order, :]))
            ax.plot(
                MCs,
                list(
                    range(
                        self.stabil_calc.modal_data.max_model_order)),
                marker='o',
                fillstyle='full',
                markerfacecolor='white',
                markeredgecolor='grey',
                color='darkgrey',
                markersize=4)
            ax.grid(True, axis='x')
            ax.set_ylim(ylim)
            ax.yaxis.tick_right()
            ax.set_xlim([0, 1])
            ax.set_xticks([ 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels([ '0.25', '0.5', '0.75', '1'])
        else:
            if len(self.fig.axes) < 2:
                return
            ax = self.fig.axes[1]
            self.fig.delaxes(ax)
            gs = matplotlib.gridspec.GridSpec(1, 1, wspace=0, hspace=0)
            self.fig.axes[0].set_subplotspec(gs[0])

        self.fig.canvas.draw_idle()

    def plot_sv_psd(self, b, NFFT=None):
        '''
         .. TODO::
             * add GUI for choosing PSD parameters
         '''    
        
        #cases:
        # create new plot with defaults (True, None)
        # hide plot (False, ...)
        #     show last plot (True, NFFT==n_lines)
        #     recreate plot with other NFFT (True, NFFT!=n_lines)
        #
        # check if something was drawn already
        #     check if it should be hidden
        #         hide it
        #     check if parameters match
        
        if self.psd_plot and not b:
            for channel in self.psd_plot:
                for line in channel:
                    line._visible = b
            self.fig.canvas.draw_idle()
            return
        elif self.psd_plot and NFFT == self.stabil_calc.prep_signals.n_lines:
            for channel in self.psd_plot:
                for line in channel:
                    line._visible = b
            self.fig.canvas.draw_idle()
            return
        elif self.psd_plot:
            for channel in self.psd_plot:
                for line in channel:
                    line.remove()
            self.psd_plot = []

        if self.stabil_calc.prep_signals is None:
            raise RuntimeError('Measurement Data was not provided!')
        if not b:
            return
        
        sv_psd = self.stabil_calc.prep_signals.sv_psd(NFFT)
        freq_psd = self.stabil_calc.prep_signals.freqs

        # sv_psd -= np.min(sv_psd)
        # sv_psd /= (np.max(sv_psd)) * 0.5 * \
        #     self.stabil_calc.modal_data.max_model_order
        sv_psd_db_scaled = 10 * np.log10(sv_psd)
        sv_psd_db_scaled -= np.min(sv_psd_db_scaled)
        sv_psd_db_scaled /= 2 * np.max(sv_psd_db_scaled)
        n_channels = sv_psd.shape[0]
        for channel in range(n_channels):
            self.psd_plot.append(self.ax.plot(freq_psd,
                                     sv_psd_db_scaled[channel, :], color='grey', 
                                     alpha = (n_channels - channel) / n_channels,
                                     linestyle='solid', visible=b,
                                     zorder=-1, transform=self.ax.get_xaxis_transform()))
        self.fig.canvas.draw_idle()

    def update_xlim(self, xlim):
        self.ax.set_xlim(xlim)
        self.fig.canvas.draw_idle()

    def update_ylim(self, ylim):
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()

#     #@pyqtSlot(bool)
#     def snap_frequency(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_df')
#             self.cursor.set_mask(mask, 'mask_df')
#
#     #@pyqtSlot(bool)
#     def snap_damping(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_dd')
#             self.cursor.set_mask(mask, 'mask_dd')
#
#     #@pyqtSlot(bool)
#     def snap_vector(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_dmac')
#             self.cursor.set_mask(mask, 'mask_dmac')
#
#     #@pyqtSlot(bool)
#     def snap_stable(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_stable')
#             self.cursor.set_mask(mask, 'mask_stable')
#
#     #@pyqtSlot(bool)
#     def snap_all(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_pre')
#             self.cursor.set_mask(mask, 'mask_pre')
#
#     #@pyqtSlot(bool)
#     def snap_clear(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_autoclear')
#             self.cursor.set_mask(mask, 'mask_autoclear')
#
#     #@pyqtSlot(bool)
#     def snap_select(self, b=True):
#         if b:
#             mask = self.stabil_calc.get_stabilization_mask('mask_autoselect')
#             self.cursor.set_mask(mask, 'mask_autoselect')

    # @pyqtSlot(int)
    def toggle_df(self, b):
        plot_obj = self.stable_plot['plot_df']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_stdf(self, b):
        plot_obj = self.stable_plot['plot_stdf']
        if plot_obj is None:
            return
        for obj in plot_obj:
            if obj is None:
                continue
            for nobj in obj:
                nobj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_stdd(self, b):
        plot_obj = self.stable_plot['plot_stdd']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_ad(self, b):
        plot_obj = self.stable_plot['plot_ad']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_dd(self, b):
        plot_obj = self.stable_plot['plot_dd']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_dmac(self, b):
        plot_obj = self.stable_plot['plot_dmac']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_mpc(self, b):
        plot_obj = self.stable_plot['plot_mpc']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_mpd(self, b):
        plot_obj = self.stable_plot['plot_dmac']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_mtn(self, b):
        plot_obj = self.stable_plot['plot_mtn']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_dev(self, b):
        plot_obj = self.stable_plot['plot_dev']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_dmtn(self, b):
        plot_obj = self.stable_plot['plot_dmtn']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_stable(self, b):
        # print('plot_stable',b)
        plot_obj = self.stable_plot['plot_stable']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_clear(self, b):
        # print('plot_autoclear',b)
        plot_obj = self.stable_plot['plot_autoclear']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_select(self, b):
        plot_obj = self.stable_plot['plot_autosel']
        if plot_obj is None:
            return
        for plot_obj_ in plot_obj:
            plot_obj_.set_visible(b)
        self.fig.canvas.draw_idle()

    # @pyqtSlot(bool)
    # @pyqtSlot(int)
    def toggle_all(self, b):
        plot_obj = self.stable_plot['plot_pre']
        if plot_obj is None:
            #print('plot_pre not found')
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()
           
    
    def save_figure(self, fname=None):

        startpath = rcParams.get('savefig.directory', '')
        startpath = os.path.expanduser(startpath)
        # start = os.path.join(startpath, self.fig.canvas.get_default_filename())

        if fname:
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                rcParams['savefig.directory'] = os.path.dirname(str(fname))
            try:
                # scatter_objs = []
                # ord_mask = self.stabil_calc.order_dummy.mask
                # self.stabil_calc.order_dummy.mask = np.ma.nomask
                # f_mask = self.stabil_calc.masked_frequencies.mask
                # self.stabil_calc.masked_frequencies.mask = np.ma.nomask
                #
                # for mode in self.stabil_calc.select_modes:
                #     mode = tuple(mode)
                #     y, x = self.stabil_calc.order_dummy[
                #         mode], self.stabil_calc.masked_frequencies[mode]
                #     # print(x,y)
                #     scatter_objs.append(
                #         self.ax.scatter(
                #             x,
                #             y,
                #             facecolors='none',
                #             edgecolors='red',
                #             s=200,
                #             visible=True))
                #
                # self.stabil_calc.order_dummy.mask = ord_mask
                # self.stabil_calc.masked_frequencies.mask = f_mask
                #
                # text = self.ax.annotate(str(self.stabil_calc.start_time), xy=(
                #     0.85, 0.99), xycoords='figure fraction')

                self.fig.canvas.print_figure(str(fname))

                # text.remove()
                #
                # for scatter_obj in scatter_objs:
                #     scatter_obj.remove()
                # del scatter_objs

            except Exception as e:
                import traceback
                traceback.print_exc()
    
    def mode_selected(self, event):
        '''
        connect this function to the button press event of the canvas
        
        '''

        if event.name == "button_press_event" and event.inaxes == self.ax:        
            
            # Check if in zooming or panning mode; credit: https://stackoverflow.com/questions/48446351/
            zooming_panning = False
            try: # Qt Backend
                zooming_panning = ( self.fig.canvas.cursor().shape() != 0 ) # 0 is the arrow, which means we are not zooming or panning.
            except: pass
            try: # nbAgg Backend
                zooming_panning = str(self.fig.canvas.toolbar.cursor) != 'Cursors.POINTER'
            except: pass
            if zooming_panning:
                logger.debug('In zooming or panning mode')
                return
            
            ind = self.cursor.i
            if ind is None:
                logger.warning('Empty mode index for the button_press_event. Ensure cursor is working.')
                return
            if ind not in self.stabil_calc.select_modes:
                self.stabil_calc.add_mode(ind)
            else:
                self.stabil_calc.remove_mode(ind)
        
    def toggle_mode(self, datapoint):
        datapoint = tuple(datapoint)
        if datapoint in self.stabil_calc.select_modes:
            self.stabil_calc.remove_mode(datapoint)
        else:
            self.stabil_calc.add_mode(datapoint)

    def add_mode(self, datapoint, list_ind):
        # datapoint = tuple(datapoint)
        # list_ind = self.stabil_calc.add_mode(datapoint)

        if len(self.scatter_objs)<= list_ind:
            self.scatter_objs.append(None)
        if self.scatter_objs[list_ind] is not None:
            self.scatter_objs[list_ind].remove()
        
        x = self.stabil_calc.masked_frequencies[datapoint]
        y = self.stabil_calc.order_dummy[datapoint]
        
        # x, y = self.x[datapoint], self.y[datapoint]
        self.scatter_objs[list_ind] = self.ax.scatter(
            x, y, facecolors='none', edgecolors='red', s=200, visible=True, zorder=3)
        
        # TODO:: improve Performance by blitting the scatter_objs
        if False: 
        #if self.useblit:
            if self.background is not None:
                self.fig.canvas.restore_region(self.background)
            for scatter in self.scatter_objs:
                scatter.set_visible(True)
                self.ax.draw_artist(scatter)
                scatter.set_visible(False)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.fig.canvas.blit(self.ax.bbox)
        else:
            # for scatter in self.scatter_objs:
            #     scatter.set_visible(True)
            self.fig.canvas.draw()
        
        
    # def add_modes(self, datalist):
    #     # convenience function for add_datapoint
    #     for datapoint in datalist:
    #         self.add_mode(datapoint)

    def remove_mode(self, datapoint, list_ind):
        # datapoint = tuple(datapoint)
        # list_ind = self.stabil_calc.remove_mode(datapoint)
        
        if list_ind is not None:            
            self.scatter_objs[list_ind].remove()
            del self.scatter_objs[list_ind]
            self.fig.canvas.draw()
            
    # def remove_modes(self, datalist):
    #     # convenience function for remove_datapoint
    #     for datapoint in datalist:
    #         self.remove_mode(datapoint)

class DataCursor(Cursor):
    # create and edit an instance of the matplotlib default Cursor widget

    #show_current_info = pyqtSignal(tuple)
    #mode_selected = pyqtSignal(tuple)
    #mode_deselected = pyqtSignal(tuple)

    def __init__(
            self,
            ax,
            order_data,
            f_data,
            mask=None,
            useblit=True,
            datalist=[],
            **lineprops):

        Cursor.__init__(self, ax, useblit=useblit, **lineprops)
        #QObject.__init__(self)
        self.callbacks = {'show_current_info':lambda *args,**kwargs: None, 
                          'mode_selected':lambda *args,**kwargs: None,
                          'mode_deselected':lambda *args,**kwargs: None,}
        self.ax = ax

        self.y = order_data
        self.y.mask = np.ma.nomask

        self.x = f_data
        self.x.mask = np.ma.nomask

        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ma.nomask

        self.name_mask = 'mask_stable'
        self.i = None

        # that list should eventually be replaced by a matplotlib.collections
        # collection
        # self.scatter_objs = []
        #
        # self.datalist = datalist
        # if datalist:
        #     self.add_datapoints(datalist)

        self.fig_resized()
        
        
    def add_callback(self, name, func):
        assert name in ['show_current_info','mode_selected','mode_deselected']
        self.callbacks[name] = func
        
    # def add_datapoint(self, datapoint):
    #     datapoint = tuple(datapoint)
    #     if datapoint not in self.datalist:
    #         self.datalist.append(datapoint)
    #     x, y = self.x[datapoint], self.y[datapoint]
    #     # print(x,y)
    #     self.scatter_objs.append(self.ax.scatter(
    #         x, y, facecolors='none', edgecolors='red', s=200, visible=False))
    #     self.callbacks['mode_selected'](datapoint)
    #
    # def add_datapoints(self, datalist):
    #     # convenience function for add_datapoint
    #     for datapoint in datalist:
    #         self.add_datapoint(datapoint)
    #
    # def remove_datapoint(self, datapoint):
    #     datapoint = tuple(datapoint)
    #     if datapoint in self.datalist:
    #         ind = self.datalist.index(datapoint)
    #         self.scatter_objs[ind].remove()
    #         del self.scatter_objs[ind]
    #         self.datalist.remove(datapoint)
    #         self.callbacks['mode_deselected'](datapoint)
    #     else:
    #         print(datapoint, 'not in self.datalist')
    #
    # def remove_datapoints(self, datalist):
    #     # convenience function for remove_datapoint
    #     for datapoint in datalist:
    #         self.remove_datapoint(datapoint)

    def set_mask(self, mask, name):
        self.mask = mask
        self.fig_resized()
        self.name_mask = name

    def fig_resized(self, event=None):
        #self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)

        # if event is not None:
        #     self.width, self.height = event.width, event.height
        # else:
        #     self.width, self.height = self.ax.get_figure(
        #     ).canvas.get_width_height()

        self.xpix, self.ypix = self.ax.transData.transform(
            np.vstack([self.x.flatten(), self.y.flatten()]).T).T

        self.xpix.shape = self.x.shape
        self.xpix.mask = self.mask

        self.ypix.shape = self.y.shape
        self.ypix.mask = self.mask

    def onmove(self, event):
        
        if self.ignore(event):
            return
        '''
        1. Override event.data to force it to snap-to nearest data item
        2. On a mouse-click, select the data item and append it to a list of selected items
        3. The second mouse-click on a previously selected item, removes it from the list
        '''
        if (self.xpix.mask).all():  # i.e. no stable poles
            return

        if event.name == "motion_notify_event":

            # get cursor coordinates
            xdata = event.xdata
            ydata = event.ydata

            if xdata is None or ydata is None:
                return

            xData_yData_pixels = self.ax.transData.transform(
                np.vstack([xdata, ydata]).T)

            xdata_pix, ydata_pix = xData_yData_pixels.T

            self.fig_resized()

            self.i = self.findIndexNearestXY(xdata_pix[0], ydata_pix[0])
            xnew, ynew = self.x[self.i], self.y[self.i]

            if xdata == xnew and ydata == ynew:
                return

            # set the cursor and draw
            event.xdata = xnew
            event.ydata = ynew

            self.callbacks['show_current_info'](self.i)

        # select item by mouse-click only if the cursor is active and in the
        # main plot
        # if event.name == "button_press_event" and event.inaxes == self.ax and self.i is not None:
        #
        #     '''
        #     we have the index already from the last motion notify event
        #     stabil_plot hold the scatter plot objects
        #     stabil_calc holds the selected modes indices
        #     both lists must be inline
        #     cursor decides if a modes is selected/deselected?
        #     '''
        #
        #
        #     if self.i not in self.stabil_plot.stabil_calc.select_modes:
        #         self.stabil_plot.add_mode(self.i)
        #         # self.linev.set_visible(False)
        #         # self.lineh.set_visible(False)
        #         #self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)
        #         #self.datalist.append(self.i)
        #         # self.ax.hold(True) # overlay plots
        #         # plot a circle where clicked
        #         #self.scatter_objs.append(self.ax.scatter(self.x[self.i], self.y[
        #         #                         self.i], facecolors='none', edgecolors='red', s=200, visible=False))
        #         self.callbacks['mode_selected'](self.i)
        #         # self.ax.draw_artist(self.scatter_objs[-1])
        #
        #     else:
        #         self.stabil_plot.remove_mode(self.i)
        #         # ind = self.datalist.index(self.i)
        #         # self.scatter_objs[ind].remove()
        #         # del self.scatter_objs[ind]
        #         # self.datalist.remove(self.i)
        #         self.callbacks['mode_deselected'](self.i)
        #
        #     # self.ax.figure.canvas.restore_region(self.background)
        #     # self.ax.figure.canvas.blit(self.ax.figure.bbox)
        #
        #     self.i = None

        Cursor.onmove(self, event)
        #for scatter in self.scatter_objs: scatter.set_visible(False)

    def _update(self):
        
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            # for scatter in self.scatter_objs:
            #     scatter.set_visible(True)
            #     self.ax.draw_artist(scatter)
            #     scatter.set_visible(False)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:
            if self.horizOn or self.vertOn:
                # for scatter in self.scatter_objs:
                #     scatter.set_visible(True)
                self.canvas.draw_idle()

        return False

    def findIndexNearestXY(self, x_point, y_point):
        '''
        Finds the nearest neighbour

        .. TODO::
            currently a very inefficient brute force implementation
            should be replaced by e.g. a k-d-tree nearest neighbour search
            `https://en.wikipedia.org/wiki/K-d_tree`

        '''

        distance = np.square(
            self.ypix - y_point) + np.square(self.xpix - x_point)
        index = np.argmin(distance)
        index = np.unravel_index(index, distance.shape)
        return index
    
    
def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10**sig_fig) == int(b * 10**sig_fig)
            )



if __name__ == '__main__':
    pass
