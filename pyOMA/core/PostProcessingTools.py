# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Post-processing tools including PoSER multi-setup merging (MergePoSER)."""

import numpy as np
import datetime
from .PreProcessingTools import PreProcessSignals
from .ModalBase import ModalBase
from .StabilDiagram import StabilCalc
from .Helpers import calculateMAC, calculateMPC, calculateMPD
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class MergePoSER(object):
    """Post-Separate Estimation and Re-scaling (PoSER) multi-setup merger.

    Combines modal results from multiple measurement setups that share a common
    set of reference channels.  Each setup is added via :meth:`add_setup`; the
    merged frequencies, damping, and mode shapes are computed by
    :meth:`merge_mode_shapes`.

    The resulting object has the same interface expected by
    :class:`~pyOMA.core.PlotMSH.ModeShapePlot` for multi-setup results.

    Notes
    -----
    For each setup the following objects must be provided:

    * ``prep_signals`` — :class:`~pyOMA.core.PreProcessingTools.PreProcessSignals`
      with ``chan_dofs`` and ``ref_channels`` defined.
    * ``modal_data`` — any :class:`~pyOMA.core.ModalBase.ModalBase` subclass
      with ``modal_frequencies``, ``modal_damping``, and ``mode_shapes``.
    * ``stabil_data`` — :class:`~pyOMA.core.StabilDiagram.StabilCalc` with
      ``select_modes`` set.
    """

    def __init__(self,):
        """Initialise an empty merger; add setups with :meth:`add_setup`."""
        self.setups = []

        self.merged_chan_dofs = []
        self.merged_num_channels = None

        self.mean_frequencies = None
        self.mean_damping = None
        self.merged_mode_shapes = None

        self.std_frequencies = None
        self.std_damping = None

        self.setup_name = 'merged_poser'
        self.start_time = datetime.datetime.now()
        self.state = [False, False]

    def _validate_add_setup_args(self, prep_signals, modal_data, stabil_data):
        """Validate types and name consistency for :meth:`add_setup` arguments."""
        if not isinstance(prep_signals, PreProcessSignals):
            raise TypeError(
                f"prep_signals must be a PreProcessSignals instance, got {type(prep_signals)}")
        if not isinstance(modal_data, ModalBase):
            raise TypeError(
                f"modal_data must be a ModalBase instance, got {type(modal_data)}")
        if not isinstance(stabil_data, StabilCalc):
            raise TypeError(
                f"stabil_data must be a StabilCalc instance, got {type(stabil_data)}")

        if prep_signals.setup_name != modal_data.setup_name:
            raise ValueError(
                f"prep_signals and modal_data belong to different setups: "
                f"{prep_signals.setup_name!r} vs {modal_data.setup_name!r}")
        if modal_data.setup_name != stabil_data.setup_name:
            raise ValueError(
                f"modal_data and stabil_data belong to different setups: "
                f"{modal_data.setup_name!r} vs {stabil_data.setup_name!r}")

        if not prep_signals.chan_dofs:
            raise ValueError(
                "prep_signals.chan_dofs must be assigned before merging setups.")

        if not stabil_data.select_modes:
            raise ValueError(
                "No modes selected in stabil_data. Call select_modes before adding the setup.")

    def add_setup(
            self,
            prep_signals,
            modal_data,
            stabil_data,
            override_ref_channels=None):
        # does not check, if same method was used for each setup, also anaylsis
        # parameters should be similar
        if override_ref_channels:
            raise RuntimeWarning('This function is not implemented yet!')

        self._validate_add_setup_args(prep_signals, modal_data, stabil_data)

        # extract needed information and store them in a dictionary
        self.setups.append({'setup_name': prep_signals.setup_name,
                            'chan_dofs': prep_signals.chan_dofs,
                            'num_channels': prep_signals.num_analised_channels,
                            'ref_channels': prep_signals.ref_channels,
                            'modal_frequencies': [modal_data.modal_frequencies[index] for index in stabil_data.select_modes],
                            'modal_damping': [modal_data.modal_damping[index] for index in stabil_data.select_modes],
                            'mode_shapes': [modal_data.mode_shapes[:, index[1], index[0]] for index in stabil_data.select_modes]
                            })
        self.start_time = min(self.start_time, prep_signals.start_time)

        logger.info('Added setup "%s" with %d channels and %d selected modes.',
                    prep_signals.setup_name, prep_signals.num_analised_channels,
                    len(stabil_data.select_modes))

        self.state[0] = True

    @staticmethod
    def _pair_modes_inner(frequencies_1, frequencies_2):
        """Pair modes between two setups by minimal relative frequency difference."""
        delta_matrix = np.ma.array(
            np.zeros((len(frequencies_1), len(frequencies_2))))
        for index, frequency in enumerate(frequencies_1):
            delta_matrix[index, :] = np.abs(
                (frequencies_2 - frequency) / frequency)
        mode_pairs = []
        while True:
            row, col = np.unravel_index(
                np.argmin(delta_matrix), delta_matrix.shape)
            delta_matrix[row, :] = np.ma.masked
            delta_matrix[:, col] = np.ma.masked
            mode_pairs.append((row, col))
            if len(mode_pairs) == len(frequencies_1):
                break
            if len(mode_pairs) == len(frequencies_2):
                break
        return mode_pairs

    @staticmethod
    def _pair_channels(chan_dofs_base, chan_dofs_this):
        """Match reference channels between two setups by node and direction."""
        pairs = []
        for chan_dof_base in chan_dofs_base:
            chan_base, node_base, az_base, elev_base = chan_dof_base[0:4]
            for chan_dof_this in chan_dofs_this:
                chan_this, node_this, az_this, elev_this = chan_dof_this[0:4]
                if node_this == node_base and az_this == az_base and elev_this == elev_base:
                    pairs.append((chan_base, chan_this))
        return pairs

    def _build_channel_and_mode_pairing(self, setups, chan_dofs_base,
                                        frequencies_base, auto_pairing):
        """Build channel and (optionally) mode pairings for all non-base setups."""
        total_dofs = len(chan_dofs_base)
        channel_pairing = []
        mode_pairing = [] if auto_pairing else None

        for setup in setups:
            these_pairs = self._pair_channels(chan_dofs_base, setup['chan_dofs'])
            channel_pairing.append(these_pairs)
            total_dofs += setup['num_channels'] - len(these_pairs)
            if auto_pairing:
                mode_pairs = self._pair_modes_inner(
                    np.array(frequencies_base), np.array(setup['modal_frequencies']))
                mode_pairing.append(mode_pairs)

        return channel_pairing, mode_pairing, total_dofs

    @staticmethod
    def _mode_in_all_setups(mode_pairing, mode_num):
        """Return True if *mode_num* appears in every setup's pairing list."""
        for mode_pairs in mode_pairing:
            for mode_pair in mode_pairs:
                if mode_pair[0] == mode_num:
                    break
            else:
                return False
        return True

    @staticmethod
    def _remove_mode_from_pairing(mode_pairing, mode_num):
        """Remove all pairs with *mode_num* as the base index from *mode_pairing*."""
        for mode_pairs in mode_pairing:
            while True:
                for index, mode_pair in enumerate(mode_pairs):
                    if mode_pair[0] == mode_num:
                        del mode_pairs[index]
                        break
                else:
                    break

    def _filter_common_modes(self, mode_pairing, frequencies_base):
        """Remove modes not common to all setups from *mode_pairing* in-place."""
        for mode_num in range(len(frequencies_base) - 1, -1, -1):
            if not self._mode_in_all_setups(mode_pairing, mode_num):
                self._remove_mode_from_pairing(mode_pairing, mode_num)

    def _copy_base_modes(self, mode_shapes, f_list, d_list, mode_pairing,
                         base_data, new_mode_nums, num_channels_base):
        """Copy modal data from the base setup into the output arrays."""
        mode_shapes_base, frequencies_base, damping_base = base_data
        for mode_num_base, _ in mode_pairing[0]:
            mode_index = new_mode_nums.index(mode_num_base)
            mode_base = mode_shapes_base[mode_num_base]
            mode_shapes[0:num_channels_base, 0, mode_index] = mode_base
            f_list[0, mode_index] = frequencies_base[mode_num_base]
            d_list[0, mode_index] = damping_base[mode_num_base]

    @staticmethod
    def _process_roving_channels(ref_channels_this, num_channels_this,
                                split_mat_refs_this, split_mat_rovs_this,
                                chan_dofs_this, start_dof, chan_dofs_base):
        """Fill selection matrices and accumulate roving chan_dofs."""
        row_ref = row_rov = 0
        for channel in range(num_channels_this):
            if channel in ref_channels_this:
                split_mat_refs_this[row_ref, channel] = 1
                row_ref += 1
            else:
                split_mat_rovs_this[row_rov, channel] = 1
                for chan_dof_this in chan_dofs_this:
                    chan, node, az, elev = chan_dof_this[0:4]
                    if chan == channel:
                        chan = int(start_dof + row_rov)
                        chan_dofs_base.append([chan, node, az, elev])
                        row_rov += 1

    def _build_split_matrices(self, these_pairs, num_channels_base,
                              num_channels_this, ref_counts,
                              chan_dofs_this, start_dof, chan_dofs_base):
        """Build selection matrices for reference and roving channels."""
        num_ref_channels, num_remain_channels = ref_counts
        ref_channels_base = [pair[0] for pair in these_pairs]
        ref_channels_this = [pair[1] for pair in these_pairs]
        logger.debug('Next Instance: %s %s', ref_channels_base, ref_channels_this)

        split_mat_refs_base = np.zeros((num_ref_channels, num_channels_base))
        split_mat_refs_this = np.zeros((num_ref_channels, num_channels_this))
        split_mat_rovs_this = np.zeros((num_remain_channels, num_channels_this))

        row_ref = 0
        for channel in range(num_channels_base):
            if channel in ref_channels_base:
                split_mat_refs_base[row_ref, channel] = 1
                row_ref += 1

        self._process_roving_channels(ref_channels_this, num_channels_this,
                                     split_mat_refs_this, split_mat_rovs_this,
                                     chan_dofs_this, start_dof, chan_dofs_base)

        return split_mat_refs_base, split_mat_refs_this, split_mat_rovs_this

    def _rescale_and_merge_modes(self, setup, setup_num, mode_pairing,
                                 split_matrices, base_ctx, output_arrays, roving_ctx):
        """Rescale and store mode contributions from one non-base setup."""
        split_mat_refs_base, split_mat_refs_this, split_mat_rovs_this = split_matrices
        mode_shapes_base, new_mode_nums = base_ctx
        mode_shapes, f_list, d_list, scale_factors = output_arrays
        start_dof, num_remain_channels = roving_ctx
        for mode_num_base, mode_num_this in mode_pairing[setup_num]:
            mode_index = new_mode_nums.index(mode_num_base)
            mode_base = mode_shapes_base[mode_num_base]
            mode_refs_base = np.dot(split_mat_refs_base, mode_base)
            mode_this = setup['mode_shapes'][mode_num_this]
            mode_refs_this = np.dot(split_mat_refs_this, mode_this)
            mode_rovs_this = np.dot(split_mat_rovs_this, mode_this)

            numer = np.dot(np.transpose(np.conjugate(mode_refs_this)), mode_refs_base)
            denom = np.dot(np.transpose(np.conjugate(mode_refs_this)), mode_refs_this)
            scale_fact = numer / denom
            scale_factors[setup_num, mode_index] = scale_fact
            mode_shapes[start_dof:start_dof + num_remain_channels, 0, mode_index] = (
                scale_fact * mode_rovs_this)
            f_list[setup_num + 1, mode_index] = setup['modal_frequencies'][mode_num_this]
            d_list[setup_num + 1, mode_index] = setup['modal_damping'][mode_num_this]

    def _normalise_and_compute_stats(self, mode_shapes, f_list, d_list,
                                     mode_pairing, new_mode_nums, common_modes):
        """Normalise merged mode shapes and compute frequency/damping statistics.

        Parameters
        ----------
        mode_shapes : np.ndarray
            Merged mode-shape array (modified in place).
        f_list, d_list : np.ndarray
            Per-setup frequency and damping arrays.
        mode_pairing : list of list
            Mode pairings.
        new_mode_nums : list of int
            Ordered mode indices.
        common_modes : int
            Number of common modes.

        Returns
        -------
        mean_frequencies, std_frequencies, mean_damping, std_damping : np.ndarray
            Statistical summaries over the merged setups.
        """
        mean_frequencies = np.zeros((common_modes,))
        std_frequencies = np.zeros((common_modes,))
        mean_damping = np.zeros((common_modes,))
        std_damping = np.zeros((common_modes,))

        for mode_num_base, _ in mode_pairing[0]:
            mode_index = new_mode_nums.index(mode_num_base)
            mode_tmp = mode_shapes[:, 0, mode_index]
            this_max = mode_tmp[np.argmax(np.abs(mode_tmp))]
            mode_shapes[:, 0, mode_index] = mode_tmp / this_max
            mean_frequencies[mode_index] = np.mean(f_list[:, mode_index], axis=0)
            std_frequencies[mode_index] = np.std(f_list[:, mode_index], axis=0)
            mean_damping[mode_index] = np.mean(d_list[:, mode_index], axis=0)
            std_damping[mode_index] = np.std(d_list[:, mode_index], axis=0)

        return mean_frequencies, std_frequencies, mean_damping, std_damping

    def _init_merge_outputs(self, setups, num_channels_base, common_modes, channel_pairing):
        """Compute total DOFs and allocate output arrays for :meth:`merge`."""
        total_dofs = num_channels_base
        for i, setup in enumerate(setups):
            total_dofs += setup['num_channels'] - len(channel_pairing[i])
        mode_shapes = np.zeros((total_dofs, 1, common_modes), dtype=complex)
        f_list = np.zeros((len(setups) + 1, common_modes))
        d_list = np.zeros((len(setups) + 1, common_modes))
        scale_factors = np.zeros((len(setups), common_modes), dtype=complex)
        return total_dofs, mode_shapes, f_list, d_list, scale_factors

    def _store_merge_results(self, chan_dofs_base, total_dofs, mode_shapes,
                             mean_frequencies, std_frequencies,
                             mean_damping, std_damping):
        """Persist merged results to instance attributes."""
        self.merged_chan_dofs = chan_dofs_base
        self.merged_num_channels = total_dofs
        self.merged_mode_shapes = mode_shapes
        self.mean_frequencies = np.expand_dims(mean_frequencies, axis=1)
        self.std_frequencies = np.expand_dims(std_frequencies, axis=1)
        self.mean_damping = np.expand_dims(mean_damping, axis=1)
        self.std_damping = np.expand_dims(std_damping, axis=1)
        self.state[1] = True

    def merge(self, base_setup_num=0, mode_pairing=None):
        '''
        generate new_chan_dofs
        assign modes from each setup

        ::
            for each mode:
                for each setup:
                    rescale
                    merge

        .. TODO::
             * rescale w.r.t to the average solution from all setups rather than specifying a base setup
             * compute scaling factors for each setup with each setup and average them for each setup before rescaling
             * corresponding standard deviations can be used to asses the quality of fit
        '''
        setups = self.setups
        chan_dofs_base = setups[base_setup_num]['chan_dofs']
        num_channels_base = setups[base_setup_num]['num_channels']
        mode_shapes_base = setups[base_setup_num]['mode_shapes']
        frequencies_base = setups[base_setup_num]['modal_frequencies']
        damping_base = setups[base_setup_num]['modal_damping']
        del setups[base_setup_num]

        if mode_pairing is None:
            auto_pairing = True
            mode_pairing = []
        else:
            auto_pairing = False
            logger.info('The provided mode pairs will be applied without any further checks.')

        channel_pairing, auto_mode_pairing, _ = self._build_channel_and_mode_pairing(
            setups, chan_dofs_base, frequencies_base, auto_pairing)

        if auto_pairing:
            mode_pairing = auto_mode_pairing

        self._filter_common_modes(mode_pairing, frequencies_base)

        lengths = [len(mode_pairs) for mode_pairs in mode_pairing]
        common_modes = min(lengths)
        new_mode_nums = [mode_num[0] for mode_num in mode_pairing[0]]

        total_dofs, mode_shapes, f_list, d_list, scale_factors = self._init_merge_outputs(
            setups, num_channels_base, common_modes, channel_pairing)

        self._copy_base_modes(mode_shapes, f_list, d_list, mode_pairing,
                              (mode_shapes_base, frequencies_base, damping_base),
                              new_mode_nums, num_channels_base)

        start_dof = num_channels_base
        for setup_num, setup in enumerate(setups):
            these_pairs = channel_pairing[setup_num]
            num_ref_channels = len(these_pairs)
            num_remain_channels = setup['num_channels'] - num_ref_channels
            split_matrices = self._build_split_matrices(
                these_pairs, num_channels_base, setup['num_channels'],
                (num_ref_channels, num_remain_channels),
                setup['chan_dofs'], start_dof, chan_dofs_base)
            self._rescale_and_merge_modes(
                setup, setup_num, mode_pairing,
                split_matrices,
                (mode_shapes_base, new_mode_nums),
                (mode_shapes, f_list, d_list, scale_factors),
                (start_dof, num_remain_channels))
            start_dof += num_remain_channels

        mean_frequencies, std_frequencies, mean_damping, std_damping = \
            self._normalise_and_compute_stats(
                mode_shapes, f_list, d_list, mode_pairing, new_mode_nums, common_modes)

        self._store_merge_results(chan_dofs_base, total_dofs, mode_shapes,
                                  mean_frequencies, std_frequencies,
                                  mean_damping, std_damping)

    def save_state(self, fname):

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {}

        out_dict['self.state'] = self.state

        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time

        if self.state[0]:
            out_dict['self.setups'] = self.setups

        if self.state[1]:
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.merged_mode_shapes'] = self.merged_mode_shapes
            out_dict['self.mean_frequencies'] = self.mean_frequencies
            out_dict['self.std_frequencies'] = self.std_frequencies
            out_dict['self.mean_damping'] = self.mean_damping
            out_dict['self.std_damping'] = self.std_damping

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname):

        logger.info('Loading results from %s', fname)

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' in in_dict:
            state = list(in_dict['self.state'])
        else:
            return

        for this_state, state_string in zip(state, ['Setups added',
                                                    'Setups merged',
                                                    ]):
            if this_state:
                logger.info(state_string)
        postprocessor = cls()

        setup_name = str(in_dict['self.setup_name'].item())
        start_time = in_dict['self.start_time'].item()

        postprocessor.setup_name = setup_name
        postprocessor.start_time = start_time

        if state[0]:
            postprocessor.setups = list(in_dict['self.setups'])

        if state[1]:
            postprocessor.merged_chan_dofs = [[int(float(chan_dof[0])),
                                               str(chan_dof[1]),
                                               float(chan_dof[2]),
                                               float(chan_dof[3]),
                                               str(chan_dof[-1])] for chan_dof in in_dict['self.merged_chan_dofs']]

            postprocessor.merged_num_channels = in_dict['self.merged_num_channels']
            postprocessor.merged_mode_shapes = in_dict['self.merged_mode_shapes']
            postprocessor.mean_frequencies = in_dict['self.mean_frequencies']
            postprocessor.std_frequencies = in_dict['self.std_frequencies']
            postprocessor.mean_damping = in_dict['self.mean_damping']
            postprocessor.std_damping = in_dict['self.std_damping']

        return postprocessor

    def _build_export_text(self, selected_freq, selected_damp, selected_stdf,
                           selected_stdd, mode_metrics, selected_modes, num_modes):
        """Build the plain-text export string for modal results."""
        selected_MPC, selected_MP, selected_MPD = mode_metrics
        freq_str = ''
        damp_str = ''
        ord_str = ''
        msh_str = ''
        mpc_str = ''
        mp_str = ''
        mpd_str = ''
        std_freq_str = ''
        std_damp_str = ''

        for col in range(num_modes):
            freq_str += '{:3.3f} \t\t'.format(selected_freq[col, 0])
            damp_str += '{:3.3f} \t\t'.format(selected_damp[col, 0])
            mpc_str += '{:3.3f}\t \t'.format(selected_MPC[col])
            mp_str += '{:3.2f} \t\t'.format(selected_MP[col])
            mpd_str += '{:3.2f} \t\t'.format(selected_MPD[col])
            std_damp_str += '{:3.3e} \t\t'.format(selected_stdd[col, 0])
            std_freq_str += '{:3.3e} \t\t'.format(selected_stdf[col, 0])

        for row in range(num_modes):
            msh_str += '\n           \t\t'
            for col in range(self.merged_num_channels):
                msh_str += '{:+3.4f} \t'.format(selected_modes[col, 0, row])

        export_modes = 'MANUAL MODAL ANALYSIS\n'
        export_modes += '=======================\n'
        export_modes += 'Frequencies [Hz]:\t' + freq_str + '\n'
        export_modes += 'Standard deviations of the Frequencies [Hz]:\t' + std_freq_str + '\n'
        export_modes += 'Damping [%]:\t\t' + damp_str + '\n'
        export_modes += 'Standard deviations of the Damping [%]:\t' + std_damp_str + '\n'
        export_modes += 'Mode shapes:\t\t' + msh_str + '\n'
        export_modes += 'Model order:\t\t' + ord_str + '\n'
        export_modes += 'MPC [-]:\t\t' + mpc_str + '\n'
        export_modes += 'MP  [°]:\t\t' + mp_str + '\n'
        export_modes += 'MPD [-]:\t\t' + mpd_str + '\n\n'
        return export_modes

    def _export_binary(self, fname, selected_freq, selected_damp, selected_stdf,
                       selected_stdd, mode_metrics, selected_modes):
        """Save modal results in compressed NumPy binary format."""
        selected_MPC, selected_MP, selected_MPD = mode_metrics
        out_dict = {
            'selected_freq': selected_freq,
            'selected_damp': selected_damp,
            'selected_MPC': selected_MPC,
            'selected_MP': selected_MP,
            'selected_MPD': selected_MPD,
            'selected_modes': selected_modes,
            'selected_stdf': selected_stdf,
            'selected_stdd': selected_stdd,
        }
        np.savez_compressed(fname, **out_dict)

    def export_results(self, fname, binary=False):

        selected_freq = self.mean_frequencies
        selected_damp = self.mean_damping
        num_modes = len(selected_freq)

        selected_MPC = calculateMPC(self.merged_mode_shapes[:, 0, :])
        selected_MP, selected_MPD = calculateMPD(self.merged_mode_shapes[:, 0, :])

        selected_stdf = self.std_frequencies
        selected_stdd = self.std_damping
        selected_modes = self.merged_mode_shapes

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        mode_metrics = (selected_MPC, selected_MP, selected_MPD)
        if binary:
            self._export_binary(fname, selected_freq, selected_damp,
                                selected_stdf, selected_stdd,
                                mode_metrics, selected_modes)
        else:
            export_modes = self._build_export_text(
                selected_freq, selected_damp, selected_stdf, selected_stdd,
                mode_metrics, selected_modes, num_modes)
            with open(fname, 'w') as f:
                f.write(export_modes)


def _resolve_candidate(row, col, row_ind, col_ind, del_row, del_col,
                       mac_matrix, debug_str):
    """Resolve the best candidate when a mode has multiple close matches."""
    if del_row and del_col:
        return row, col, debug_str
    best = np.nanargmax([mac_matrix[row_ind, col],
                         mac_matrix[row, col_ind],
                         mac_matrix[row, col]])
    if best == 0:
        return row_ind, col, debug_str + f'Chose alternative match for "Mode A" at {row_ind}, '
    if best == 1:
        return row, col_ind, debug_str + f'Chose alternative match for "Mode B" at {col_ind}, '
    if not del_row:
        return row, col, debug_str + f'Reject alternative match for "Mode A" at {row_ind}, '
    return row, col, debug_str + f'Reject alternative match for "Mode B" at {col_ind}, '


def _check_threshold_debug(row, col, delta_matrix, mac_matrix,
                           freq_thresh, mac_thresh, debug_str):
    """Append threshold-check information to *debug_str* when DEBUG logging is active.

    Parameters
    ----------
    row, col : int
        Current candidate pair indices.
    delta_matrix : np.ma.MaskedArray
        Relative frequency-difference matrix.
    mac_matrix : np.ndarray
        MAC matrix.
    freq_thresh, mac_thresh : float
        Acceptance thresholds.
    debug_str : str
        Running debug string.

    Returns
    -------
    debug_str : str
        Updated debug string.
    """
    if delta_matrix[row, col] < freq_thresh or mac_matrix[row, col] > mac_thresh:
        debug_str += "Thresholds are within limits for: "
        if delta_matrix[row, col] < freq_thresh:
            debug_str += "freq, "
        if mac_matrix[row, col] > mac_thresh:
            debug_str += "mac, "
    else:
        debug_str += "Thresholds are out of limits, "
    return debug_str


def _build_freq_mac_matrices(freq_a, freq_b, shapes_a, shapes_b):
    """Build relative-frequency-difference and MAC matrices for mode pairing."""
    shape = (len(freq_a), len(freq_b))
    delta_matrix = np.ma.array(np.zeros(shape), mask=np.zeros(shape))
    for index, frequency in enumerate(freq_a):
        delta_matrix[index, :] = np.abs(
            (freq_b - frequency) / (0.5 * (freq_b + frequency)))
    delta_matrix.mask = np.isnan(delta_matrix)
    return delta_matrix, calculateMAC(shapes_a, shapes_b)


def _find_col_ambiguity(delta_matrix, row, col):
    """Check if *col* is the unambiguous argmin-row for any other column."""
    for col_ind in range(delta_matrix.shape[1]):
        if col_ind == col:
            continue
        if delta_matrix[:, col_ind].mask.all():
            continue
        if np.nanargmin(delta_matrix[:, col_ind]) == row:
            return col_ind, False
    return col, True


def _find_row_ambiguity(delta_matrix, row, col):
    """Check if *row* is the unambiguous argmin-col for any other row."""
    for row_ind in range(delta_matrix.shape[0]):
        if row_ind == row:
            continue
        if delta_matrix[row_ind, :].mask.all():
            continue
        if np.nanargmin(delta_matrix[row_ind, :]) == col:
            return row_ind, False
    return row, True


def pair_modes(freq_a, freq_b,
               shapes_a, shapes_b,
               freq_thresh=0.2, mac_thresh=0.8):
    '''
    A function to pair two sets of modes (here: a and b) based on frequency
    differences and mode shape similarity. The number of modes in both sets may
    be different and relative complements of both arrays may be non-empty.

    The threshold where pairing stops is based on normalized frequency differences
    AND modal assurance criteria.

    Parameters
    ----------

        f_a, f_b: np.ndarray
            Arrays holding the natural frequencies of both sets of modes. The
            dimension (number of modes) of both sets can be different.
        d_a, d_b: np.ndarray
            Arrays holding the damping ratios of both sets of modes. The dimension
            (number of modes) of both sets can be different.
        phi_a, phi_b: np.ndarray
            Arrays holding the mode shapes of both sets of modes. The first
            dimension is the number of channels, that must match in both arrays.

    Other Parameters
    ----------------
        kwargs :
            Additional kwargs are passed to pair_modes

    Returns
    -------
        inds_a, inds_b: np.ndarray,
            Arrays holding the indices of paired modes sorted by ascending
            frequencies (set a). Length represents the number of common modes.

        unp_a, unp_b: np.ndarray
            Arrays holding the indices of modes that could not be paired
    '''
    delta_matrix, mac_matrix = _build_freq_mac_matrices(freq_a, freq_b, shapes_a, shapes_b)

    indices_a = []
    indices_b = []
    delta_values = []
    mac_values = []

    while ~np.all(delta_matrix.mask):
        row, col = np.unravel_index(np.nanargmin(delta_matrix), delta_matrix.shape)
        col_ind, del_col = _find_col_ambiguity(delta_matrix, row, col)
        row_ind, del_row = _find_row_ambiguity(delta_matrix, row, col)

        debug_str = f"Current Minimum at {row}:{col}, "
        row, col, debug_str = _resolve_candidate(
            row, col, row_ind, col_ind, del_row, del_col, mac_matrix, debug_str)
        debug_str = _check_threshold_debug(
            row, col, delta_matrix, mac_matrix, freq_thresh, mac_thresh, debug_str)

        if delta_matrix[row, col] < freq_thresh and mac_matrix[row, col] > mac_thresh:
            delta_values.append(delta_matrix[row, col])
            mac_values.append(mac_matrix[row, col])
            indices_a.append(row)
            indices_b.append(col)
            debug_str += "Selecting candidate."
        else:
            debug_str += "Rejecting candidate."

        delta_matrix[row, :] = np.ma.masked
        delta_matrix[:, col] = np.ma.masked
        logger.debug(debug_str)

    sort_inds = np.argsort(freq_a[indices_a])
    indices_a = np.array(indices_a)[sort_inds]
    indices_b = np.array(indices_b)[sort_inds]

    unp_a = list(np.setdiff1d(np.arange(len(freq_a)), indices_a))
    unp_b = list(np.setdiff1d(np.arange(len(freq_b)), indices_b))

    return indices_a, indices_b, unp_a, unp_b


def compare_modes(f_a, d_a, phi_a, f_b, d_b, phi_b, **kwargs):
    '''
    Compares two sets of modes (set a and set b)  by first pairing them and then displaying
    statistics on the identified pairs and a full MAC matrix for manual assessment.

    Parameters
    ----------

        f_a, f_b: np.ndarray
            Arrays holding the natural frequencies of both sets of modes. The
            dimension (number of modes) of both sets can be different.
        d_a, d_b: np.ndarray
            Arrays holding the damping ratios in percent of both sets of modes. The dimension
            (number of modes) of both sets can be different.
        phi_a, phi_b: np.ndarray
            Arrays holding the mode shapes of both sets of modes. The first
            dimension is the number of channels, that must match in both arrays.

    Other Parameters
    ----------------
        kwargs :
            Additional kwargs are passed to pair_modes

    Returns
    -------
        inds_a, inds_b: np.ndarray
            Arrays holding the indices of paired modes

        unp_a, unp_b: np.ndarray
            Arrays holding the indices of modes that could not be paired

    '''
    import matplotlib.pyplot as plt

    inds_a, inds_b, unp_a, unp_b = pair_modes(f_a, f_b, phi_a, phi_b, **kwargs)
    if inds_a.shape[0] == 0:
        logger.warning('Could not match any modes. Consider raising freq_thresh or lowering mac_thresh.')
        return inds_a, inds_b, unp_a, unp_b
    if np.max(d_a) <= 1:
        logger.warning('First set damping values do not seem to be given in percent.')
    if np.max(d_b) <= 1:
        logger.warning('Second set damping values do not seem to be given in percent.')

    all_inds_b = np.concatenate((inds_b, unp_b))
    corr_inds_a = np.ma.concatenate([np.ma.array(inds_a, mask=np.zeros_like(inds_a, dtype=bool)), np.ma.array(np.zeros_like(unp_b), mask=np.ones_like(unp_b, dtype=bool), dtype=int)])

    # indices of "modes 1" in the order of "modes 2" (for each mode 2 the index of mode 1 of nan)
    corr_inds_a_sort = corr_inds_a[np.argsort(all_inds_b)]

    freqs_a_corr = np.ma.array(f_a[corr_inds_a_sort],
                             mask=corr_inds_a_sort.mask,
                             fill_value=np.nan
                            ).filled()
    damps_a_corr = np.ma.array(d_a[corr_inds_a_sort],
                                 mask=corr_inds_a_sort.mask,
                                 fill_value=np.nan
                                ).filled()
    msh_a_corr = np.ma.array(phi_a[:, corr_inds_a_sort],
                               mask=np.repeat(np.ma.getmaskarray(corr_inds_a_sort)[np.newaxis, :], phi_a.shape[0], axis=0),
                               fill_value=np.nan
                              ).filled()

    freq_diffs = freqs_a_corr - f_b
    damp_diffs = damps_a_corr - d_b
    mac_matrix = calculateMAC(msh_a_corr, phi_b)

    macs = np.diag(mac_matrix)

    # create the alpha mask: put 0.5 into every row corresponding to unp 1 and every column corresponding to unp 2
    mac_matrix = calculateMAC(phi_a, phi_b)
    alpha_mask = np.ones_like(mac_matrix)
    alpha_mask[unp_a, :] = 0.25
    alpha_mask[:, unp_b] = 0.25

    plt.matshow(mac_matrix, alpha=alpha_mask, cmap='viridis_r', vmin=0, vmax=1)
    plt.yticks(ticks=np.arange(f_a.shape[0]), labels=[f"{v:1.2f} Hz" for v in f_a])
    plt.xticks(ticks=np.arange(f_b.shape[0]), labels=[f"{v:1.2f} Hz" for v in f_b], rotation=90)
    plt.scatter(inds_b, inds_a, color='r', marker='+')

    logger.info(f'''Statistics on identification:
Δf = {np.nanmean(freq_diffs):1.3f}± {np.nanstd(freq_diffs):1.3f},
Δd = {np.nanmean(damp_diffs):1.3f}± {np.nanstd(damp_diffs):1.3f},
MAC: mean = {np.nanmean(macs):1.3f}, min= {np.nanmin(macs):1.3f},
Number of unmatched modes: "a" {len(unp_a)}, "b" {len(unp_b)}''')

    plt.figure()
    plt.plot(f_a, d_a, marker='x', color='black', ls='none')
    plt.plot(f_b, d_b, marker='+', color='black', ls='none')
    for ind_a, ind_b in zip(inds_a, inds_b):
        fs = (f_a[ind_a], f_b[ind_b])
        ds = (d_a[ind_a], d_b[ind_b])
        plt.plot(fs, ds, color='red')
    plt.annotate(f'''Statistics on identification:
Δf = {np.nanmean(freq_diffs):1.3f}± {np.nanstd(freq_diffs):1.3f},
Δd = {np.nanmean(damp_diffs):1.3f}± {np.nanstd(damp_diffs):1.3f},
MAC: mean = {np.nanmean(macs):1.3f}, min= {np.nanmin(macs):1.3f},
Number of unmatched modes: "a" {len(unp_a)}, "b" {len(unp_b)}''',
                 (0.55, 0.7), xycoords='figure fraction')

    return inds_a, inds_b, unp_a, unp_b


def main():
    pass


if __name__ == '__main__':
    main()
