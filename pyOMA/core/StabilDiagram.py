# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Stabilization diagram computation (StabilCalc, StabilCluster) and static plot (StabilPlot)."""

from .SSICovRef import PogerSSICovRef
from .ModalBase import ModalBase
from .Helpers import simplePbar, calculateMAC, calculateMPC, calculateMPD
import numpy as np

import scipy.cluster
import scipy.spatial
import scipy.stats

import os
import warnings
import dataclasses
from typing import Optional, Tuple

import collections
from operator import itemgetter
from random import shuffle

# check if python is running in headless mode i.e. as a server script
# if 'DISPLAY' in os.environ:
#     matplotlib.use("Qt5Agg", force=True)
from matplotlib import rcParams
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

# Namedtuples used to group related arrays and reduce parameter counts
_ScalarDiffs = collections.namedtuple('_ScalarDiffs', ['lambda_diffs', 'freq_diffs', 'damp_diffs'])
_ModalDiffs = collections.namedtuple(
    '_ModalDiffs',
    ['MAC_diffs', 'MPC_matrix', 'MPD_matrix', 'MP_matrix', 'MPD_diffs', 'MP_diffs'])
_SelectedResults = collections.namedtuple(
    '_SelectedResults',
    ['freq', 'damp', 'order', 'modes', 'MC', 'MPC', 'MP', 'MPD', 'stdf', 'stdd', 'stdmsh'])

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


@dataclasses.dataclass
class StabCriteria:
    """Grouping of all stabilization threshold parameters.

    Pass an instance to :meth:`StabilCalc.calculate_stabilization_masks` or
    :meth:`StabilCalc.update_stabilization_masks` instead of individual keyword
    arguments.

    Parameters
    ----------
    order_range : tuple, optional
        ``(start, step, stop)`` for the model-order range.
    d_range : tuple, optional
        ``(min, max)`` damping ratio [%] band.
    stdf_max : float, optional
        Maximum relative standard deviation of frequency [%].
    stdd_max : float, optional
        Maximum relative standard deviation of damping [%].
    mpc_min : float, optional
        Minimum Modal Phase Collinearity value.
    mpd_max : float, optional
        Maximum Mean Phase Deviation [°].
    mtn_min : float, optional
        Minimum Modal Transfer Norm.
    df_max : float, optional
        Maximum relative frequency difference for stabilization.
    dd_max : float, optional
        Maximum relative damping difference for stabilization.
    dmac_max : float, optional
        Maximum MAC difference for stabilization.
    dev_min : float, optional
        Minimum relative eigenvalue difference.
    dmtn_min : float, optional
        Minimum Modal Transfer Norm difference.
    MC_min : float, optional
        Minimum Modal Contribution.
    """

    order_range: Optional[Tuple] = None
    d_range: Optional[Tuple] = None
    stdf_max: Optional[float] = None
    stdd_max: Optional[float] = None
    mpc_min: Optional[float] = None
    mpd_max: Optional[float] = None
    mtn_min: Optional[float] = None
    df_max: Optional[float] = None
    dd_max: Optional[float] = None
    dmac_max: Optional[float] = None
    dev_min: Optional[float] = None
    dmtn_min: Optional[float] = None
    MC_min: Optional[float] = None


class StabilCalc(object):
    """Stabilisation diagram computation and pole selection.

    Computes stabilisation masks by comparing modal parameters between
    successive model orders, applies physical criteria (frequency, damping,
    MPC, MPD, MAC), and manages pole selection for export.  Optionally
    delegates automatic clearing and clustering to
    :class:`StabilCluster`.

    Parameters
    ----------
    modal_data : ModalBase
        Any pyOMA system-identification result object (must be a subclass of
        :class:`~pyOMA.core.ModalBase.ModalBase`).
    prep_signals : PreProcessSignals, optional
        Deprecated — ignored; ``modal_data.prep_signals`` is used instead.

    .. TODO::
        * scale markers right on every platform
        * frequency range as argument or from ssi params, sampling freq
        * add switch to choose between "unstable only in ..." or "stable in ..."
        * distinguish between stabilization criteria and filtering criteria
        * rework mask logic (currently it is very difficult to understand)
        * Merge DataCursor and JupyterGUI.SnappingCursor
    """

    def __init__(self, modal_data, prep_signals=None, **kwargs):

        super().__init__()
        if not isinstance(modal_data, ModalBase):
            raise TypeError(f"Expected ModalBase for 'modal_data', got {type(modal_data).__name__!r}.")

        self.modal_data = modal_data
        self.extra_func = None
        self.setup_name = modal_data.setup_name
        self.start_time = modal_data.start_time

        if prep_signals is not None:
            logger.warning('Providing prep_signals is not required anymore. Ignoring argument!')
        self.prep_signals = modal_data.prep_signals

        self.capabilities = self._build_capabilities()
        self._init_masked_arrays()
        self._init_masks()
        self._init_thresholds()

        self.select_modes = []
        self.select_callback = None
        self.state = 0
        self.callbacks = {'add_mode': [], 'remove_mode': []}

    def _build_capabilities(self):
        """Build and return the capabilities dict from modal_data attributes."""
        md = self.modal_data
        return {
            'f': 1,
            'd': 1,
            'msh': md.__dict__.get('mode_shapes') is not None,
            'std': md.__dict__.get('std_frequencies') is not None,
            'ev': md.__dict__.get('eigenvalues') is not None,
            'mtn': 0,
            'MC': md.__dict__.get('modal_contributions') is not None,
            'auto': isinstance(self, StabilCluster),
            'data': md.prep_signals is not None,
        }

    def _init_masked_arrays(self):
        """Allocate masked frequency, damping, eigenvalue and order arrays."""
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
            [[order] * self.num_solutions for order in range(max_model_order)],
            fill_value=0)

    def _init_masks(self):
        """Initialise stable-in and only-unstable-in mask dicts to None."""
        self.masks = {
            'mask_pre': None,   # some constraints (f>0.0, order_range, etc)
            'mask_ad': None,    # absolute damping
            'mask_stdf': None,  # uncertainty frequency
            'mask_stdd': None,  # uncertainty damping
            'mask_mpc': None,   # absolute modal phase collinearity
            'mask_mpd': None,   # absolute mean phase deviation
            'mask_mtn': None,   # absolute modal transfer norm
            'mask_df': None,    # difference frequency
            'mask_dd': None,    # difference damping
            'mask_dmac': None,  # difference mac
            'mask_dev': None,   # difference eigenvalue
            'mask_dmtn': None,  # difference modal transfer norm
            'mask_stable': None,  # stable in all criteria
        }
        self.nmasks = {
            'mask_ad': None,    # absolute damping
            'mask_stdf': None,  # uncertainty frequency
            'mask_stdd': None,  # uncertainty damping
            'mask_ampc': None,  # absolute modal phase collinearity
            'mask_ampd': None,  # absolute mean phase deviation
            'mask_amtn': None,  # absolute modal transfer norm
            'mask_df': None,    # difference frequency
            'mask_dd': None,    # difference damping
            'mask_dmac': None,  # difference mac
            'mask_dev': None,   # difference eigenvalue
            'mask_dmtn': None,  # difference modal transfer norm
        }

    def _init_thresholds(self):
        """Set default stabilization threshold attributes."""
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

    def add_callback(self, name, func):
        if name not in ['add_mode', 'remove_mode']:
            raise ValueError(f"'name' must be one of {['add_mode', 'remove_mode']}, got {name!r}.")
        self.callbacks[name].append(func)

    def calculate_soft_critera_matrices(self):
        logger.info('Checking stabilisation criteria...')

        # Direction 1: model order, Direction 2: current pole, Direction 3:
        # previous pole:
        max_model_order = self.modal_data.max_model_order
        num_solutions = self.num_solutions
        capabilities = self.capabilities

        scalar_diffs, modal_diffs = self._init_criteria_matrices(
            max_model_order, num_solutions, capabilities)

        # Initialise previous-order state
        prev_state = self._get_initial_order_state(capabilities, 0)

        pbar = simplePbar(max_model_order - 1)
        for curr_order in range(1, max_model_order):
            next(pbar)

            curr_non_zero_entries, curr_length = self._get_non_zero_entries(
                capabilities, curr_order)

            # print(curr_length)
            if not curr_length:
                continue

            curr_state = self._get_curr_order_state(
                capabilities, curr_order, curr_non_zero_entries)

            self._update_scalar_diff_matrices(
                capabilities, curr_order, curr_non_zero_entries,
                prev_state, curr_state, scalar_diffs)

            if capabilities['msh']:
                self._update_mac_mpc_matrices(
                    curr_order, curr_non_zero_entries, curr_length,
                    prev_state, curr_state, modal_diffs)

            prev_state = curr_state

        if capabilities['ev']:
            self.lambda_diffs = scalar_diffs.lambda_diffs
        self.freq_diffs = scalar_diffs.freq_diffs
        self.damp_diffs = scalar_diffs.damp_diffs
        self.MAC_diffs = modal_diffs.MAC_diffs
        self.MPD_diffs = modal_diffs.MPD_diffs
        self.MP_diffs = modal_diffs.MP_diffs

        self.MPD_matrix = modal_diffs.MPD_matrix
        self.MP_matrix = modal_diffs.MP_matrix
        self.MPC_matrix = modal_diffs.MPC_matrix

        self.state = 1

    # ------------------------------------------------------------------
    # Private helpers for calculate_soft_critera_matrices
    # ------------------------------------------------------------------

    def _init_criteria_matrices(self, max_model_order, num_solutions, capabilities):
        """Allocate all zero-filled difference and absolute matrices.

        Returns
        -------
        scalar_diffs : _ScalarDiffs
        modal_diffs : _ModalDiffs
        """
        shape3d = (max_model_order, num_solutions, num_solutions)
        shape2d = (max_model_order, num_solutions)

        scalar_diffs = _ScalarDiffs(
            lambda_diffs=np.ma.zeros(shape3d, fill_value=0),
            freq_diffs=np.ma.zeros(shape3d, fill_value=0),
            damp_diffs=np.ma.zeros(shape3d, fill_value=0),
        )

        if capabilities['msh']:
            modal_diffs = _ModalDiffs(
                MAC_diffs=np.ma.zeros(shape3d, fill_value=0),
                MPC_matrix=np.ma.zeros(shape2d, fill_value=0),
                MPD_matrix=np.ma.zeros(shape2d, fill_value=0),
                MP_matrix=np.ma.zeros(shape2d, fill_value=0),
                MPD_diffs=np.ma.zeros(shape3d, fill_value=0),
                MP_diffs=np.ma.zeros(shape3d, fill_value=0),
            )
        else:
            modal_diffs = _ModalDiffs(
                MAC_diffs=None, MPC_matrix=None, MPD_matrix=None,
                MP_matrix=None, MPD_diffs=None, MP_diffs=None,
            )

        return scalar_diffs, modal_diffs

    def _get_initial_order_state(self, capabilities, order):
        """Build the previous-order state dict for order 0."""
        if capabilities['ev']:
            prev_lambda_row = self.masked_lambda.data[order, :]
        prev_freq_row = self.masked_frequencies[order, :]
        prev_damp_row = self.modal_data.modal_damping[order, :]

        if capabilities['ev']:
            nze = np.where(
                (~np.isnan(prev_lambda_row.imag)) & (prev_lambda_row.imag != 0))
        else:
            nze = np.where(
                (~np.isnan(prev_freq_row)) & (prev_freq_row != 0))

        length = len(nze[0])
        freq = prev_freq_row[nze]
        damp = prev_damp_row[nze]
        state = {'nze': nze, 'length': length, 'freq': freq, 'damp': damp}

        if capabilities['ev']:
            state['lambda'] = prev_lambda_row[nze]

        if capabilities['msh']:
            prev_msh_row = self.modal_data.mode_shapes[:, :, order]
            msh = prev_msh_row[:, nze[0]]
            mpd, mp_new = calculateMPD(msh)
            mp_new[mp_new > 90] -= 180  # in range [-90,90]
            state['msh'] = msh
            state['MPD'] = mpd
            state['MP_new'] = mp_new

        return state

    def _get_non_zero_entries(self, capabilities, curr_order):
        """Return (non_zero_entries_tuple, count) for the current order."""
        if capabilities['ev']:
            row = self.masked_lambda.data[curr_order, :]
            nze = np.where((~np.isnan(row.imag)) & (row.imag != 0))
        else:
            row = self.masked_frequencies[curr_order, :]
            nze = np.where((~np.isnan(row)) & (row != 0))
        return nze, len(nze[0])

    def _get_curr_order_state(self, capabilities, curr_order, nze):
        """Build the current-order state dict from raw data."""
        freq_row = self.masked_frequencies[curr_order, :]
        damp_row = self.modal_data.modal_damping[curr_order, :]
        state = {
            'nze': nze,
            'length': len(nze[0]),
            'freq': freq_row[nze],
            'damp': damp_row[nze],
        }
        if capabilities['ev']:
            state['lambda'] = self.masked_lambda.data[curr_order, :][nze]
        if capabilities['msh']:
            msh_row = self.modal_data.mode_shapes[:, :, curr_order]
            msh = msh_row[:, nze[0]]
            mpd, mp = calculateMPD(msh[:, :state['length']])
            mp_new = np.copy(mp)
            mp_new[mp_new > 90] -= 180
            state['msh'] = msh
            state['MPD'] = mpd
            state['MP'] = mp
            state['MP_new'] = mp_new
        return state

    def _rel_diff_matrix(self, prev_vals, curr_vals):
        """Compute element-wise relative difference matrix (prev x curr)."""
        div = np.maximum(
            np.repeat(np.expand_dims(np.abs(prev_vals), axis=1),
                      curr_vals.shape[0], axis=1),
            np.repeat(np.expand_dims(np.abs(curr_vals), axis=0),
                      prev_vals.shape[0], axis=0))
        return np.abs((
            np.repeat(np.expand_dims(prev_vals, axis=1),
                      curr_vals.shape[0], axis=1)
            - curr_vals) / div).T

    def _update_scalar_diff_matrices(
            self, capabilities, curr_order, nze,
            prev, curr, scalar_diffs):
        """Fill lambda/freq/damp relative-difference slices."""
        prev_length = prev['length']
        if capabilities['ev']:
            div_lambda = np.maximum(
                np.repeat(np.expand_dims(np.ma.abs(prev['lambda']), axis=1),
                          curr['lambda'].shape[0], axis=1),
                np.repeat(np.expand_dims(np.ma.abs(curr['lambda']), axis=0),
                          prev['lambda'].shape[0], axis=0))
            scalar_diffs.lambda_diffs[curr_order, nze[0], :prev_length] = np.abs(
                (np.repeat(np.expand_dims(prev['lambda'], axis=1),
                           curr['lambda'].shape[0], axis=1)
                 - curr['lambda']) / div_lambda).T

        scalar_diffs.freq_diffs[curr_order, nze[0], :prev_length] = self._rel_diff_matrix(
            prev['freq'], curr['freq'])
        scalar_diffs.damp_diffs[curr_order, nze[0], :prev_length] = self._rel_diff_matrix(
            prev['damp'], curr['damp'])

    def _update_mac_mpc_matrices(
            self, curr_order, nze, curr_length,
            prev, curr, modal_diffs):
        """Fill MAC, MPC, MPD and MP difference slices."""
        prev_length = prev['length']

        mac_diffs = np.transpose(
            1 - calculateMAC(
                prev['msh'][:, :prev_length],
                curr['msh'][:, :curr_length]))
        modal_diffs.MAC_diffs[curr_order, nze[0], :prev_length] = mac_diffs

        modal_diffs.MPC_matrix[curr_order, nze[0]] = calculateMPC(
            curr['msh'][:, :curr_length])

        modal_diffs.MPD_matrix[curr_order, nze[0]] = curr['MPD']
        modal_diffs.MP_matrix[curr_order, nze[0]] = curr['MP']

        modal_diffs.MPD_diffs[curr_order, nze[0], :len(prev['MPD'])] = (
            self._rel_diff_matrix(prev['MPD'], curr['MPD']))

        modal_diffs.MP_diffs[curr_order, nze[0], :len(prev['MP_new'])] = (
            self._rel_diff_matrix(prev['MP_new'], curr['MP_new']))

    def export_results(self, fname, binary=False):

        (selected_freq, selected_damp, selected_modes, _,
         selected_order, selected_MC,
         selected_MPC, selected_MP, selected_MPD,
         selected_stdf, selected_stdd, selected_stdmsh) = self.get_selected_modal_values()

        results = _SelectedResults(
            freq=selected_freq, damp=selected_damp, order=selected_order,
            modes=selected_modes, MC=selected_MC,
            MPC=selected_MPC, MP=selected_MP, MPD=selected_MPD,
            stdf=selected_stdf, stdd=selected_stdd, stdmsh=selected_stdmsh,
        )

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if binary:
            self._export_binary(fname, results)
        else:
            export_modes = self._build_text_export(results)
            with open(fname, 'w') as fh:
                fh.write(export_modes)

    # ------------------------------------------------------------------
    # Private helpers for export_results
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_field(fmt, vals):
        """Join all values in *vals* formatted by *fmt*."""
        return ''.join(fmt.format(v) for v in vals)

    def _build_export_scalar_strings(self, results):
        """Return per-column string accumulators for scalar fields."""
        has_msh = self.capabilities['msh']
        has_std = self.capabilities['std']
        has_mc = self.capabilities['MC']
        fmt = self._fmt_field
        return (
            fmt('{:<3.3f}\t\t', results.freq),
            fmt('{:<3.3f}\t\t', results.damp),
            fmt('{:<6d}\t\t', results.order),
            fmt('{:<3.3f}\t \t', results.MPC) if has_msh else None,
            fmt('{:<3.2f}\t\t', results.MP) if has_msh else None,
            fmt('{:<3.2f}\t\t', results.MPD) if has_msh else None,
            fmt('{:<3.3e}\t\t', results.stdf) if has_std else None,
            fmt('{:<3.3e}\t\t', results.stdd) if has_std else None,
            fmt('{:<3.3f}\t\t', results.MC) if has_mc else None,
        )

    def _get_chan_dofs(self):
        """Return the channel-DOF list appropriate for the current modal data."""
        if isinstance(self.modal_data, PogerSSICovRef):
            return self.modal_data.merged_chan_dofs
        if self.capabilities['data']:
            return self.prep_signals.chan_dofs
        return []

    def _row_label(self, row, chan_dofs):
        """Return the mode-shape row label string for *row*."""
        for chan_dof in chan_dofs:
            chan, node, az, elev = chan_dof[:4]
            if chan == row:
                return (
                    f'\n{node.ljust(10)}  ({az: <+3.2f}, {elev: >+3.2f})'
                    '                  \t')
        return '\n                                            '

    def _build_mode_shape_string(self, selected_modes, selected_stdmsh):
        """Return (msh_str, std_msh_str) for mode shape rows."""
        has_std = self.capabilities['std']
        chan_dofs = self._get_chan_dofs()
        msh_parts = []
        std_parts = [] if has_std else None

        for row in range(selected_modes.shape[0]):
            msh_parts.append(self._row_label(row, chan_dofs))
            if has_std:
                std_parts.append('\n           \t\t')
            for col in range(selected_modes.shape[1]):
                msh_parts.append('{:+<3.4f}\t'.format(selected_modes[row, col]))
                if has_std:
                    std_parts.append('{:+<3.3e} \t'.format(
                        selected_stdmsh[row, col]))

        msh_str = ''.join(msh_parts)
        std_msh_str = ''.join(std_parts) if has_std else None
        return msh_str, std_msh_str

    def _build_text_export(self, results):
        """Compose the human-readable text export string."""
        (freq_str, damp_str, ord_str,
         mpc_str, mp_str, mpd_str,
         std_freq_str, std_damp_str, MC_str) = self._build_export_scalar_strings(results)

        if self.capabilities['msh']:
            msh_str, std_msh_str = self._build_mode_shape_string(
                results.modes, results.stdmsh)
        else:
            msh_str = None
            std_msh_str = None

        lines = ['MANUAL MODAL ANALYSIS\n', '=======================\n']
        lines.append('Frequencies [Hz]:                           \t' + freq_str + '\n')
        if self.capabilities['std']:
            lines.append('Standard deviations of the Frequencies [Hz]:\t' + std_freq_str + '\n')
        lines.append('Damping [%]:                                \t' + damp_str + '\n')
        if self.capabilities['std']:
            lines.append('Standard deviations of the Damping [%]:     \t' + std_damp_str + '\n')
        if self.capabilities['MC']:
            lines.append('Modal Contributions of the mode [-]:        \t' + MC_str + '\n')
        if self.capabilities['msh']:
            lines.append('Node        (Azimuth, Elevation)            \tMode shapes:' + msh_str + '\n')
        if self.capabilities['std']:
            lines.append('Standard Deviations of the Mode shapes:     \t' + std_msh_str + '\n')
        lines.append('Model order:                                \t' + ord_str + '\n')
        if self.capabilities['msh']:
            lines.append('MPC [-]:                                    \t' + mpc_str + '\n')
            lines.append('MP  [\u00b0]:                                    \t' + mp_str + '\n')
            lines.append('MPD [-]:                                    \t' + mpd_str + '\n\n')
        return ''.join(lines)

    def _export_binary(self, fname, results):
        """Save selected results as a compressed NumPy archive."""
        out_dict = {
            'selected_freq': results.freq,
            'selected_damp': results.damp,
            'selected_order': results.order,
        }
        if self.capabilities['msh']:
            out_dict['selected_MPC'] = results.MPC
            out_dict['selected_MP'] = results.MP
            out_dict['selected_MPD'] = results.MPD
            out_dict['selected_modes'] = results.modes
        if self.capabilities['std']:
            out_dict['selected_stdf'] = results.stdf
            out_dict['selected_stdd'] = results.stdd
            out_dict['selected_stdmsh'] = results.stdmsh
        np.savez_compressed(fname, **out_dict)

    _CRITERIA_FLAT_KEYS = frozenset({
        'order_range', 'd_range', 'stdf_max', 'stdd_max',
        'mpc_min', 'mpd_max', 'mtn_min', 'df_max', 'dd_max',
        'dmac_max', 'dev_min', 'dmtn_min', 'MC_min',
    })

    @staticmethod
    def _resolve_criteria(criteria, kwargs):
        """Return a :class:`StabCriteria` from *criteria* or flat *kwargs*.

        When flat keyword arguments are passed instead of a ``StabCriteria``
        object, a :class:`DeprecationWarning` is emitted and a temporary
        ``StabCriteria`` is built from the kwargs so the rest of the code path
        stays the same.
        """
        if criteria is not None:
            if not isinstance(criteria, StabCriteria):
                raise TypeError(
                    f'criteria must be a StabCriteria instance, got '
                    f'{type(criteria).__name__!r}')
            return criteria
        return StabilCalc._build_criteria_from_kwargs(kwargs)

    @staticmethod
    def _build_criteria_from_kwargs(kwargs):
        """Build a :class:`StabCriteria` from flat keyword arguments."""
        flat_keys = StabilCalc._CRITERIA_FLAT_KEYS
        used_flat = {k: v for k, v in kwargs.items() if k in flat_keys and v is not None}
        if used_flat:
            warnings.warn(
                'Pass a StabCriteria object instead of individual threshold '
                'arguments',
                DeprecationWarning,
                stacklevel=4)
        return StabCriteria(**{k: v for k, v in kwargs.items() if k in flat_keys})

    def calculate_stabilization_masks(self, criteria=None, **kwargs):
        """Compute all stabilization masks from scratch.

        Parameters
        ----------
        criteria : StabCriteria, optional
            Threshold bundle.  Pass this *or* the individual keyword arguments
            below (old call style, deprecated).
        **kwargs : optional
            Individual threshold values (deprecated — use *criteria* instead).
            Accepted keys: order_range, d_range, stdf_max, stdd_max, mpc_min,
            mpd_max, mtn_min, df_max, dd_max, dmac_max, dev_min, dmtn_min,
            MC_min.
        """
        if self.state < 1:
            self.calculate_soft_critera_matrices()

        c = self._resolve_criteria(criteria, kwargs)

        # Fill in defaults when not provided by the caller
        defaults = StabCriteria(
            order_range=(0, 1, self.modal_data.max_model_order),
            d_range=(0, 100),
            stdf_max=100, stdd_max=100,
            mpc_min=0, mpd_max=90, mtn_min=0,
            df_max=0.01, dd_max=0.05,
            dmac_max=0.02, dev_min=0.02, dmtn_min=0.02, MC_min=0,
        )
        merged = StabCriteria(
            **{
                field.name: (
                    getattr(c, field.name)
                    if getattr(c, field.name) is not None
                    else getattr(defaults, field.name)
                )
                for field in dataclasses.fields(StabCriteria)
            }
        )

        self.state = 2
        self.update_stabilization_masks(merged)

    def update_stabilization_masks(self, criteria=None, **kwargs):
        """Update the stabilization masks with new threshold values.

        Parameters
        ----------
        criteria : StabCriteria, optional
            Threshold bundle.  Pass this *or* the individual keyword arguments
            below (old call style, deprecated).
        **kwargs : optional
            Individual threshold values (deprecated — use *criteria* instead).
            Accepted keys: order_range, d_range, stdf_max, stdd_max, mpc_min,
            mpd_max, mtn_min, df_max, dd_max, dmac_max, dev_min, dmtn_min,
            MC_min.
        """
        if self.state < 2:
            self.calculate_stabilization_masks()

        c = self._resolve_criteria(criteria, kwargs)

        # Merge non-None fields from criteria into instance attributes
        self._apply_criteria(c)

        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        mask_pre = self._compute_mask_pre(c.order_range)
        self.masks['mask_pre'] = mask_pre

        self._compute_absolute_masks(mask_pre, c)
        self._compute_diff_masks(mask_pre, c)
        self._finalize_stable_mask(mask_pre)

    # ------------------------------------------------------------------
    # Private helpers for update_stabilization_masks
    # ------------------------------------------------------------------

    def _apply_criteria(self, c):
        """Copy non-None fields of *c* onto instance attributes."""
        for field in dataclasses.fields(StabCriteria):
            val = getattr(c, field.name)
            if val is not None:
                setattr(self, field.name, val)

    def _compute_mask_pre(self, order_range):
        """Return the base pre-mask (valid frequencies and selected orders)."""
        mask_pre = (
            (~np.isnan(self.masked_frequencies)) & (self.masked_frequencies != 0))

        if order_range is not None:
            start, step, stop = order_range
            start = max(0, start)
            stop = min(stop, self.modal_data.max_model_order)
            mask_order = np.zeros_like(mask_pre)
            for order in range(start, stop, step):
                mask_order = np.logical_or(
                    mask_order, self.order_dummy == order)
            mask_pre = np.logical_and(mask_pre, mask_order)

        return mask_pre

    def _compute_absolute_masks(self, mask_pre, c):
        """Compute absolute-criterion masks (damping, std, MPC, MPD, MC)."""
        self._apply_damping_mask(mask_pre, c)
        self._apply_std_masks(mask_pre, c)
        self._apply_mpc_mpd_masks(mask_pre, c)
        self._apply_mc_mask(mask_pre, c)

    def _apply_damping_mask(self, mask_pre, c):
        """Compute and store the absolute-damping mask."""
        if c.d_range is None:
            return
        if not isinstance(c.d_range, (tuple, list)):
            raise TypeError(
                f"d_range must be a tuple or list, got "
                f"{type(c.d_range).__name__!r}")
        if len(c.d_range) != 2:
            raise ValueError(
                f"d_range must have length 2, got {len(c.d_range)}")
        mask = np.logical_and(
            mask_pre, self.modal_data.modal_damping >= c.d_range[0])
        self.masks['mask_ad'] = np.logical_and(
            mask, self.modal_data.modal_damping <= c.d_range[1])

    def _apply_std_masks(self, mask_pre, c):
        """Compute and store standard-deviation masks for frequency and damping."""
        if not self.capabilities['std']:
            return
        num_blocks = self.modal_data.num_blocks
        t_factor = scipy.stats.t.ppf(0.95, num_blocks)
        sqrt_nb = np.sqrt(num_blocks)
        if c.stdf_max is not None:
            mask = t_factor * self.modal_data.std_frequencies / sqrt_nb <= c.stdf_max
            self.masks['mask_stdf'] = np.logical_and(mask_pre, mask)
        if c.stdd_max is not None:
            mask = t_factor * self.modal_data.std_damping / sqrt_nb <= c.stdd_max
            self.masks['mask_stdd'] = np.logical_and(mask_pre, mask)

    def _apply_mpc_mpd_masks(self, mask_pre, c):
        """Compute and store MPC and MPD masks."""
        if c.mpc_min is not None:
            self.masks['mask_mpc'] = np.logical_and(
                mask_pre, self.MPC_matrix >= c.mpc_min)
        if c.mpd_max is not None:
            self.masks['mask_mpd'] = np.logical_and(
                mask_pre, self.MPD_matrix <= c.mpd_max)

    def _apply_mc_mask(self, mask_pre, c):
        """Compute and store the modal-contribution mask."""
        if c.MC_min is None or not self.capabilities['MC']:
            return
        mc = self.modal_data.modal_contributions
        if np.issubdtype(mc.dtype, complex):
            contrib_mask = np.abs(mc) >= c.MC_min
        else:
            contrib_mask = mc >= c.MC_min
        self.masks['mask_MC'] = np.logical_and(mask_pre, contrib_mask)

    def _compute_diff_masks(self, mask_pre, c):
        """Compute difference-criterion masks (df, dd, dMAC) and collect them."""
        full_masks = []

        if c.df_max is not None:
            # rel freq diffs for each pole with all previous poles,
            # for all poles and orders results in 3d array
            # compare those rel freq diffs with df_max
            # and reduce 3d array to 2d array, by applying logical_or
            # along each poles axis (diff with all previous)
            mask_sf_all = np.logical_and(
                self.freq_diffs != 0, self.freq_diffs <= c.df_max)
            mask_sf_red = np.any(mask_sf_all, axis=2)
            self.masks['mask_df'] = np.logical_and(mask_pre, mask_sf_red)
            full_masks.append(mask_sf_all)

        if c.dd_max is not None:
            mask_sd_all = np.logical_and(
                self.damp_diffs != 0, self.damp_diffs <= c.dd_max)
            mask_sd_red = np.any(mask_sd_all, axis=2)
            self.masks['mask_dd'] = np.logical_and(mask_pre, mask_sd_red)
            full_masks.append(mask_sd_all)

        if c.dmac_max is not None:
            mask_sv_all = np.logical_and(
                self.MAC_diffs != 0, self.MAC_diffs <= c.dmac_max)
            mask_sv_red = np.any(mask_sv_all, axis=2)
            self.masks['mask_dmac'] = np.logical_and(mask_pre, mask_sv_red)
            full_masks.append(mask_sv_all)

        return full_masks

    def _finalize_stable_mask(self, mask_pre):
        """Combine all individual masks into mask_stable and nmasks."""
        stable_mask = self._combine_into_stable_mask(mask_pre)
        self.masks['mask_stable'] = stable_mask
        self.nmasks = self._build_nmasks(stable_mask)

    def _combine_into_stable_mask(self, mask_pre):
        """Return the combined stable mask from diff and absolute masks."""
        c = StabCriteria(df_max=self.df_max, dd_max=self.dd_max, dmac_max=self.dmac_max)
        full_masks = self._compute_diff_masks(mask_pre, c)

        if full_masks:
            stable_mask_full = np.ones_like(full_masks[0])
            for mask in full_masks:
                stable_mask_full = np.logical_and(stable_mask_full, mask)
            stable_mask = np.any(stable_mask_full, axis=2)
        else:
            stable_mask = mask_pre

        skip = {'mask_stable', 'mask_autosel', 'mask_autoclear'}
        for mask_name, mask in self.masks.items():
            if mask_name not in skip and mask is not None:
                stable_mask = np.logical_and(stable_mask, mask)
        return stable_mask

    def _build_nmasks(self, stable_mask):
        """Build and return the only-unstable-in-... nmasks dict."""
        _skip = {'mask_pre', 'mask_stable', 'mask_autosel', 'mask_autoclear'}
        nmasks = {
            name: np.logical_not(stable_mask)
            for name, mask in self.masks.items() if mask is not None}

        for nname, nmask in nmasks.items():
            if nname in _skip:
                continue
            for name, mask in self.masks.items():
                if mask is None or name in _skip:
                    continue
                if name == nname:
                    nmask = np.logical_and(nmask, np.logical_not(mask))
                else:
                    nmask = np.logical_and(nmask, mask)
            nmasks[nname] = nmask

        nmasks['mask_stable'] = stable_mask
        nmasks['mask_pre'] = self.masks['mask_pre']
        nmasks['mask_autoclear'] = np.logical_not(
            self.masks.get('mask_autoclear', None))
        nmasks['mask_autosel'] = np.logical_not(
            self.masks.get('mask_autosel', None))
        return nmasks

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

        selected_lambda = self._get_selected_lambda(select_modes)
        selected_MPC, selected_MP, selected_MPD = self._get_selected_mpc_mpd(select_modes)
        selected_stdf, selected_stdd, selected_stdmsh = self._get_selected_std(select_modes)
        selected_MC = self._get_selected_mc(select_modes)
        selected_modes, selected_stdmsh = self._get_selected_mode_shapes(
            select_modes, selected_stdmsh)

        return selected_freq, selected_damp, selected_modes, selected_lambda, \
            selected_order, selected_MC, \
            selected_MPC, selected_MP, selected_MPD, \
            selected_stdf, selected_stdd, selected_stdmsh

    # ------------------------------------------------------------------
    # Private helpers for get_selected_modal_values
    # ------------------------------------------------------------------

    def _get_selected_lambda(self, select_modes):
        """Return list of eigenvalues for selected modes, or None."""
        if self.capabilities['ev']:
            return [self.modal_data.eigenvalues[index] for index in select_modes]
        return None

    def _get_selected_mpc_mpd(self, select_modes):
        """Return (MPC, MP, MPD) lists for selected modes, or (None, None, None)."""
        if self.capabilities['msh']:
            selected_MPC = [self.MPC_matrix[index] for index in select_modes]
            selected_MP = [self.MP_matrix[index] for index in select_modes]
            selected_MPD = [self.MPD_matrix[index] for index in select_modes]
            return selected_MPC, selected_MP, selected_MPD
        return None, None, None

    def _get_selected_std(self, select_modes):
        """Return (stdf, stdd, stdmsh) for selected modes, or (None, None, None)."""
        if self.capabilities['std']:
            selected_stdf = [self.modal_data.std_frequencies[index]
                             for index in select_modes]
            selected_stdd = [self.modal_data.std_damping[index]
                             for index in select_modes]
            selected_stdmsh = np.zeros(
                (self.modal_data.mode_shapes.shape[0], len(select_modes)),
                dtype=complex)
            return selected_stdf, selected_stdd, selected_stdmsh
        return None, None, None

    def _get_selected_mc(self, select_modes):
        """Return list of modal contributions for selected modes, or None."""
        if self.capabilities['MC']:
            return [self.modal_data.modal_contributions[index]
                    for index in select_modes]
        return None

    def _get_selected_mode_shapes(self, select_modes, selected_stdmsh):
        """Return (mode_shape_matrix, updated_stdmsh), or (None, selected_stdmsh)."""
        if not self.capabilities['msh']:
            return None, selected_stdmsh

        n_channels = self.modal_data.mode_shapes.shape[0]
        selected_modes = np.zeros(
            (n_channels, len(select_modes)), dtype=complex)

        for num, ind in enumerate(select_modes):
            row_index = ind[0]
            col_index = ind[1]
            mode_tmp = self.modal_data.mode_shapes[:, col_index, row_index]

            if self.capabilities['std']:
                std_mode = self.modal_data.std_mode_shapes[:, col_index, row_index]
                selected_stdmsh[:, num] = std_mode
            else:
                # scaling of mode shape
                abs_mode_tmp = np.abs(mode_tmp)
                this_max = mode_tmp[np.argmax(abs_mode_tmp)]
                mode_tmp = mode_tmp / this_max

            selected_modes[:, num] = mode_tmp

        return selected_modes, selected_stdmsh

    def _validate_index(self, i):
        """Raise if *i* is not a valid (order, solution) index pair."""
        if not isinstance(i, (list, tuple)):
            raise TypeError(f"Expected list or tuple for 'i', got {type(i).__name__!r}.")
        if len(i) != 2:
            raise ValueError(f"'i' must have length 2, got {len(i)}.")
        if i[0] > self.modal_data.max_model_order:
            raise ValueError(f"i[0]={i[0]} exceeds max_model_order={self.modal_data.max_model_order}.")
        if i[1] > self.num_solutions:
            raise ValueError(f"i[1]={i[1]} exceeds num_solutions={self.num_solutions}.")

    def _get_msh_values(self, i):
        """Return (mpc, mp, mpd, dmp, dmpd) for index *i*, or NaN tuple."""
        if not self.capabilities['msh']:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        mpc = self.MPC_matrix[i]
        mp = self.MP_matrix[i]
        mpd = self.MPD_matrix[i]
        mp_diffs_row = self.MP_diffs[i]
        nz = np.nonzero(mp_diffs_row)[0]
        dmp = np.min(mp_diffs_row[nz]) if len(nz) >= 1 else 0
        mpd_diffs_row = self.MPD_diffs[i]
        nz2 = np.nonzero(mpd_diffs_row)[0]
        dmpd = np.min(mpd_diffs_row[nz2]) if len(nz2) >= 1 else 0
        return mpc, mp, mpd, dmp, dmpd

    def get_modal_values(self, i):
        # needed for gui
        self._validate_index(i)

        n = self.order_dummy[i]
        f = self.masked_frequencies[i]
        d = self.modal_data.modal_damping[i]

        mpc, mp, mpd, dmp, dmpd = self._get_msh_values(i)
        stdf = self.modal_data.std_frequencies[i] if self.capabilities['std'] else np.nan
        stdd = self.modal_data.std_damping[i] if self.capabilities['std'] else np.nan
        mtn = np.nan  # not yet implemented
        MC = self.modal_data.modal_contributions[i] if self.capabilities['MC'] else np.nan
        ex_1, ex_2 = self.extra_func(self.modal_data, i, True) if self.extra_func is not None else (np.nan, np.nan)

        return n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1, ex_2

    def get_mode_shape(self, i):
        self._validate_index(i)
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

        out_dict = self._collect_state_dict()
        np.savez_compressed(fname, **out_dict)

    def _collect_state_dict(self):
        """Build and return the dictionary that represents the full saved state."""
        out_dict = {
            'self.state': self.state,
            'self.setup_name': self.setup_name,
            'self.start_time': self.start_time,
        }

        if self.state >= 1:
            self._collect_state_diff_matrices(out_dict)

        if self.state >= 2:
            self._collect_state_criteria(out_dict)

        if self.capabilities['auto']:
            self._collect_state_auto(out_dict)

        out_dict['self.select_modes'] = self.select_modes
        return out_dict

    def _collect_state_diff_matrices(self, out_dict):
        """Add soft-criteria difference matrices to *out_dict* (state >= 1)."""
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

    def _collect_state_criteria(self, out_dict):
        """Add stabilization criteria thresholds to *out_dict* (state >= 2)."""
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

    def _collect_state_auto(self, out_dict):
        """Add automatic clustering results to *out_dict* (auto mode)."""
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

    @classmethod
    def load_state(cls, fname, modal_data):
        logger.info('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        if 'self.state' not in in_dict:
            return None

        state = float(in_dict['self.state'])
        setup_name = str(in_dict['self.setup_name'].item())

        if setup_name != modal_data.setup_name:
            raise ValueError(
                f"Setup name mismatch: expected {setup_name!r}, "
                f"got {modal_data.setup_name!r}"
            )

        stabil_data = cls(modal_data)
        cls._parse_state_dict(stabil_data, in_dict, state)
        stabil_data.state = state
        return stabil_data

    @staticmethod
    def _parse_state_dict(stabil_data, in_dict, state):
        """Populate *stabil_data* from the loaded *in_dict*."""
        if state >= 1:
            StabilCalc._parse_state_diff_matrices(stabil_data, in_dict)

        if state >= 2:
            StabilCalc._parse_state_criteria(stabil_data, in_dict)

        if stabil_data.capabilities['auto']:
            StabilCalc._parse_state_auto(stabil_data, in_dict, state)

        select_modes = [tuple(a) for a in in_dict['self.select_modes']]
        frequencies = [stabil_data.masked_frequencies[idx[0], idx[1]]
                       for idx in select_modes]
        stabil_data.select_modes = [
            x for _, x in sorted(zip(frequencies, select_modes))]

    @staticmethod
    def _parse_state_diff_matrices(stabil_data, in_dict):
        """Restore soft-criteria matrices (state >= 1)."""
        if stabil_data.capabilities['ev']:
            stabil_data.lambda_diffs = np.ma.array(in_dict['self.lambda_diffs'])
        stabil_data.freq_diffs = np.ma.array(in_dict['self.freq_diffs'])
        stabil_data.damp_diffs = np.ma.array(in_dict['self.damp_diffs'])

        if stabil_data.capabilities['msh']:
            stabil_data.MAC_diffs = np.ma.array(in_dict['self.MAC_diffs'])
            stabil_data.MPD_diffs = np.ma.array(in_dict['self.MPD_diffs'])
            stabil_data.MP_diffs = np.ma.array(in_dict['self.MP_diffs'])
            stabil_data.MPC_matrix = np.ma.array(in_dict['self.MPC_matrix'])
            stabil_data.MP_matrix = np.ma.array(in_dict['self.MP_matrix'])
            stabil_data.MPD_matrix = np.ma.array(in_dict['self.MPD_matrix'])

    @staticmethod
    def _parse_state_criteria(stabil_data, in_dict):
        """Restore stabilization criteria thresholds (state >= 2)."""
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

    @staticmethod
    def _parse_state_auto(stabil_data, in_dict, state):
        """Restore automatic clustering data (auto mode)."""
        if state >= 3:
            stabil_data.num_iter = int(in_dict['self.num_iter'])
            stabil_data.threshold = float(in_dict['self.threshold'])
            stabil_data.clear_ctr = in_dict['self.clear_ctr']
        if state >= 4:
            stabil_data.use_stabil = bool(in_dict['self.use_stabil'])
            stabil_data.proximity_matrix_sq = in_dict['self.proximity_matrix_sq']
            stabil_data.cluster_assignments = in_dict['self.cluster_assignments']
        if state >= 5:
            stabil_data.select_clusters = list(in_dict['self.select_clusters'])
            stabil_data.nr_poles = list(in_dict['self.nr_poles'])
            stabil_data.selection_cut_off = float(in_dict['self.selection_cut_off'])


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

        if not self.capabilities['ev']:
            raise RuntimeError("This functionality requires eigenvalues to be available. "
                               "Ensure the modal analysis algorithm provides eigenvalues.")

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

        if num_iter is not None:
            if not isinstance(num_iter, int):
                raise TypeError(f"Expected int for 'num_iter', got {type(num_iter).__name__!r}.")
            if num_iter <= 0:
                raise ValueError(f"'num_iter' must be greater than 0, got {num_iter!r}.")
            self.num_iter = num_iter

        mask_pre = np.ma.array(self.get_stabilization_mask('mask_pre'))
        soft_criteria_matrices = self._build_soft_criteria_matrices(mask_pre)
        all_poles = self._whiten_poles(soft_criteria_matrices, mask_pre)
        mask_autoclear = self._run_kmeans_clearing(all_poles, mask_pre)
        self._compute_clearing_threshold(soft_criteria_matrices, mask_autoclear)

        self.masks['mask_autoclear'] = mask_autoclear
        self.update_stabilization_masks()
        self.state = 3

    def _build_soft_criteria_matrices(self, mask_pre):
        """Return list of min-reduced soft-criteria 2D matrices."""
        self.freq_diffs.mask = np.ma.nomask
        self.damp_diffs.mask = np.ma.nomask
        self.lambda_diffs.mask = np.ma.nomask
        if self.capabilities['msh']:
            self.MAC_diffs.mask = np.ma.nomask
            self.MP_diffs.mask = np.ma.nomask

        mask_pre_3d = self.freq_diffs == 0
        soft = []
        for matrix in [self.lambda_diffs, self.freq_diffs, self.damp_diffs]:
            matrix.mask = mask_pre_3d
            soft.append(matrix.min(axis=2))

        if self.capabilities['msh']:
            self.MAC_diffs.mask = mask_pre_3d
            soft.append(self.MAC_diffs.min(axis=2))
            self.MP_diffs.mask = mask_pre_3d
            soft.append(self.MP_diffs.min(axis=2))

        for matrix in soft:
            matrix.mask = mask_pre
        return soft

    def _whiten_poles(self, soft_criteria_matrices, mask_pre):
        """Stack, log-transform and whiten the soft-criteria vectors."""
        compressed = [m[2:, :].compressed() for m in soft_criteria_matrices]
        poles = np.vstack(compressed).T
        poles = np.log(poles)
        poles -= np.mean(poles, axis=0)
        poles /= np.std(poles, axis=0)
        return poles

    def _run_kmeans_clearing(self, all_poles, mask_pre):
        """Run 2-means and return the autoclear mask."""
        std_dev = np.std(all_poles, axis=0)
        ctr_init = np.array([-std_dev, std_dev])
        self.clear_ctr, idx = scipy.cluster.vq.kmeans2(
            all_poles, ctr_init, self.num_iter)
        logger.info(
            'Possibly physical poles 1st stage: {0} Spurious poles 1st stage: {1}'.format(
                collections.Counter(idx)[0], collections.Counter(idx)[1]))

        mask_pre.mask = np.ma.nomask
        new_idx = np.hstack(
            (np.ones(np.sum(np.logical_not(mask_pre[:2, :]))), idx))
        mask_autoclear = self.decompress_flat_mask(mask_pre, new_idx)
        return np.logical_or(mask_autoclear, mask_pre)

    def _compute_clearing_threshold(self, soft_criteria_matrices, mask_autoclear):
        """Set self.threshold as the 95th percentile of d_lambda + d_MAC."""
        soft_criteria_matrices[0].mask = np.ma.nomask
        soft_criteria_matrices[3].mask = np.ma.nomask
        distance_mat = soft_criteria_matrices[0] + soft_criteria_matrices[3]
        distance_mat.mask = mask_autoclear
        self.threshold = np.percentile(distance_mat.compressed(), q=95)

    def automatic_classification(self, threshold=None, use_stabil=False):
        if self.state < 3 and not use_stabil:
            self.automatic_clearing()
        logger.info('Classifying physical modes automatically...')

        if use_stabil:
            mask_autoclear = self.get_stabilization_mask('mask_stable')
        else:
            mask_autoclear = self.get_stabilization_mask('mask_autoclear')

        self.use_stabil = use_stabil

        if threshold is not None:
            if not isinstance(threshold, int):
                raise TypeError(f"Expected int for 'threshold', got {type(threshold).__name__!r}.")
            self.threshold = threshold

        if self.threshold is None:
            self.threshold = self._auto_threshold(mask_autoclear)

        lambda_compressed, mode_shapes_compressed = self._compress_modal_data(mask_autoclear)
        self._cluster_modes(lambda_compressed, mode_shapes_compressed, mask_autoclear)
        self._log_double_orders(mask_autoclear)
        self.order_dummy.mask = np.ma.nomask
        logger.info('Number of classified clusters: {}'.format(
            max(self.cluster_assignments)))
        self.state = 4

    def _auto_threshold(self, mask_autoclear):
        """Compute a 95th-percentile threshold from lambda+MAC distances."""
        self.freq_diffs.mask = np.ma.nomask
        mask_pre_3d = self.freq_diffs == 0
        self.lambda_diffs.mask = mask_pre_3d
        self.MAC_diffs.mask = mask_pre_3d
        distance_mat = (self.lambda_diffs.min(axis=2)
                        + self.MAC_diffs.min(axis=2))
        distance_mat.mask = mask_autoclear
        return np.percentile(distance_mat.compressed(), q=95)

    def _compress_modal_data(self, mask_autoclear):
        """Return compressed lambda and mode-shape arrays for unmasked poles."""
        length_mat = int(np.prod(mask_autoclear.shape) - np.sum(mask_autoclear))
        self.masked_lambda.mask = mask_autoclear
        lambda_compressed = self.masked_lambda.compressed()
        self.masked_lambda.mask = np.ma.nomask

        dim0, dim1 = mask_autoclear.shape
        mode_shapes_compressed = np.zeros(
            (self.modal_data.mode_shapes.shape[0], length_mat),
            dtype=np.complex128)
        n = 0
        for i in range(dim0):
            for j in range(dim1):
                if not mask_autoclear[i, j]:
                    mode_shapes_compressed[:, n] = self.modal_data.mode_shapes[:, j, i]
                    n += 1
        return lambda_compressed, mode_shapes_compressed

    def _cluster_modes(self, lambda_compressed, mode_shapes_compressed, mask_autoclear):
        """Build proximity matrix and perform hierarchical clustering."""
        l = len(lambda_compressed)
        div_lambda = np.maximum(
            np.repeat(np.expand_dims(np.abs(lambda_compressed), axis=1),
                      l, axis=1),
            np.repeat(np.expand_dims(np.abs(lambda_compressed), axis=0),
                      l, axis=0))
        lambda_prox = np.abs(
            lambda_compressed - lambda_compressed.reshape((l, 1))) / div_lambda
        mac_prox = 1 - calculateMAC(mode_shapes_compressed, mode_shapes_compressed)
        proximity_matrix = (self.weight_lambda * lambda_prox
                            + self.weight_MAC * mac_prox)
        proximity_matrix[proximity_matrix < np.finfo(proximity_matrix.dtype).eps] = 0

        self.proximity_matrix_sq = scipy.spatial.distance.squareform(
            proximity_matrix, checks=False)
        linkage_matrix = scipy.cluster.hierarchy.linkage(
            self.proximity_matrix_sq, method='average')
        self.cluster_assignments = scipy.cluster.hierarchy.fcluster(
            linkage_matrix, self.threshold, criterion='distance')

    def _log_double_orders(self, mask_autoclear):
        """Log any model orders appearing more than once per cluster."""
        for clusternr in range(1, max(self.cluster_assignments) + 1):
            flat_poles_ind = self.cluster_assignments != clusternr + 1
            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.order_dummy.mask = mask
            for order in range(self.modal_data.max_model_order):
                if np.sum(self.order_dummy == order) > 1:
                    logger.debug(f'Double Model Order: {self.order_dummy[order, :]}')

    def automatic_selection(self, number=0):

        if self.state < 4:
            self.automatic_classification()

        nr_poles, select_clusters = self._determine_cluster_selection(number)
        logger.info('Number of physical modes: {0}'.format(
            collections.Counter(select_clusters)[0]))
        self.select_clusters = select_clusters
        self.nr_poles = nr_poles

        self.selection_cut_off = np.inf
        for i, b in zip(self.nr_poles, self.select_clusters):
            if not b:
                self.selection_cut_off = min(i - 1, self.selection_cut_off)
        logger.info('Minimum number of elements in retained clusters: {}'.format(
            self.selection_cut_off))

        mask_autoclear = (
            self.get_stabilization_mask('mask_stable')
            if self.use_stabil
            else self.masks['mask_autoclear'])

        self.MAC_diffs.mask = self.MAC_diffs == 0
        MAC_diffs = self.MAC_diffs.min(axis=2)
        self.MAC_diffs.mask = np.ma.nomask

        if 'mask_autosel' not in self.masks:
            self.masks['mask_autosel'] = []

        for clusternr, inout in enumerate(select_clusters):
            if inout:
                continue
            self._select_cluster_representative(clusternr, mask_autoclear)

        for matrix in [self.masked_frequencies, self.masked_damping,
                       MAC_diffs, self.MPC_matrix, self.MPD_matrix]:
            matrix.mask = np.ma.nomask

        self.state = 5

    def _determine_cluster_selection(self, number):
        """Return (nr_poles, select_clusters) arrays for all clusters."""
        poles = [np.where(self.cluster_assignments == c)
                 for c in range(1, 1 + max(self.cluster_assignments))]
        nr_poles = np.array([len(a[0]) for a in poles], dtype=np.float64)
        max_nr = float(np.max(nr_poles))

        if number == 0:
            _, select_clusters = scipy.cluster.vq.kmeans2(
                nr_poles, np.array([max_nr, 1e-12]), self.num_iter)
        else:
            meta_list = sorted(enumerate(nr_poles), key=itemgetter(1), reverse=True)
            select_clusters = [1] * len(nr_poles)
            for i in range(number):
                select_clusters[meta_list[i][0]] = 0
        return nr_poles, select_clusters

    def _select_cluster_representative(self, clusternr, mask_autoclear):
        """Select the multivariate-median pole from cluster *clusternr*."""
        flat_poles_ind = self.cluster_assignments != clusternr + 1
        mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
        self.masks['mask_autosel'].append(np.ma.copy(mask))

        num_poles_left = int(np.prod(mask.shape) - np.sum(mask))
        while num_poles_left > 1:
            ind = []
            for matrix in [self.masked_frequencies, self.masked_damping]:
                matrix.mask = mask
                val = np.ma.median(matrix)
                min_ = np.min(matrix)
                max_ = np.max(matrix)
                if val - min_ <= max_ - val:
                    ind.append(np.where(matrix == max_))
                else:
                    ind.append(np.where(matrix == min_))
            for k in range(min(len(ind), num_poles_left - 1)):
                mask[ind[k]] = True
            num_poles_left = int(np.prod(mask.shape) - np.sum(mask))

        select_mode = np.where(np.logical_not(mask))
        self.add_mode((select_mode[0][0], select_mode[1][0]))
        if self.select_callback is not None:
            self.select_callback(self.select_modes[-1])

    def plot_clearing(self, save_path=None):

        mask_autoclear = self.masks['mask_autoclear']
        mask_pre = self.get_stabilization_mask('mask_pre')
        self.plot_mask(mask_autoclear, save_path)

        crits, labels = self._get_clearing_crits()
        self._plot_clearing_pairs(crits, labels, mask_autoclear, mask_pre, save_path)
        for crit in crits:
            crit.mask = np.ma.nomask

    def _get_clearing_crits(self):
        """Return (crits, labels) lists for clearing scatter plots."""
        def _min_nonzero(matrix):
            matrix.mask = matrix == 0
            result = matrix.min(axis=2)
            matrix.mask = np.ma.nomask
            return result

        freq_diffs = _min_nonzero(self.freq_diffs)
        MAC_diffs = _min_nonzero(self.MAC_diffs)
        damp_diffs = _min_nonzero(self.damp_diffs)
        crits = [freq_diffs, damp_diffs, MAC_diffs, self.MPC_matrix, self.MPD_matrix]
        labels = ['df', 'dd', 'MAC', 'MPC', 'MPD']
        return crits, labels

    def _plot_clearing_pairs(self, crits, labels, mask_autoclear, mask_pre, save_path):
        """Plot each unique pair of clearing criteria against each other."""
        new_crits = []
        for j, b in enumerate(crits):
            new_crits.append(b)
            for i, a in enumerate(new_crits):
                if a is b:
                    continue
                self._plot_one_clearing_pair(
                    a, b, labels[i], labels[j], mask_autoclear, mask_pre, save_path)

    def _plot_one_clearing_pair(self, a, b, labela, labelb, mask_autoclear, mask_pre, save_path):
        """Draw a single clearing scatter plot for criteria *a* vs *b*."""
        plot.figure(tight_layout=1)
        a.mask = mask_autoclear
        b.mask = mask_autoclear
        plot.plot(a.compressed(), b.compressed(), ls='', marker=',')
        plot.plot(np.mean(a), np.mean(b), ls='', marker='d', color='black')
        a.mask = mask_pre
        b.mask = mask_pre
        plot.plot(a.compressed(), b.compressed(), ls='', marker=',', color='grey')
        plot.plot(np.mean(a), np.mean(b), ls='', marker='d', color='grey')
        plot.xlabel(labela)
        plot.ylabel(labelb)
        plot.xlim((0, 1))
        plot.ylim((0, 1))
        if save_path is not None:
            plot.savefig(save_path + 'clear_{}_{}.pdf'.format(labela, labelb))
        else:
            plot.show()
            plot.pause(0.01)

    def plot_classification(self, save_path=None):
        rel_matrix = scipy.cluster.hierarchy.linkage(
            self.proximity_matrix_sq, method='average')
        lvs = scipy.cluster.hierarchy.leaves_list(rel_matrix)

        def _llf(_id):
            if len(lvs) > 500:
                if (np.where(_id == lvs)[0][0] % 100 == 0):
                    return str(np.where(_id == lvs)[0][0])
                else:
                    return str('')
            else:
                if (np.where(_id == lvs)[0][0] % 10 == 0):
                    return str(np.where(_id == lvs)[0][0])
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
        self._plot_cluster_sizes(save_path)
        self._plot_stabilization_clusters(save_path)

    def _plot_cluster_sizes(self, save_path):
        """Plot bar chart of cluster sizes split into accepted/rejected."""
        plot.figure(tight_layout=1)
        in_poles = sorted(
            self.nr_poles[self.nr_poles >= self.selection_cut_off], reverse=True)
        out_poles = sorted(
            self.nr_poles[(self.nr_poles < self.selection_cut_off) & (self.nr_poles > 0)],
            reverse=True)
        plot.bar(range(len(in_poles)), in_poles,
                 facecolor='red', edgecolor='none', align='center')
        plot.bar(range(len(in_poles), len(in_poles) + len(out_poles)), out_poles,
                 facecolor='blue', edgecolor='none', align='center')
        plot.xlim((0, len(self.nr_poles)))
        plot.tight_layout()
        if save_path is not None:
            plot.savefig(save_path + 'cluster_sizes.pdf')
        else:
            plot.show()
            plot.pause(0.001)

    def _plot_stabilization_clusters(self, save_path):
        """Plot stabilisation diagram with cluster frequency spans."""
        fig = plot.figure(tight_layout=1)
        ax1 = fig.add_subplot(211)

        mask_autoclear = self.masks['mask_autoclear']
        mask_pre = self.get_stabilization_mask('mask_pre')
        mask_pre_ = np.logical_not(
            np.logical_and(np.logical_not(mask_pre), mask_autoclear))

        self._scatter_masked(ax1, mask_pre_, 'grey', 'pole')
        self._scatter_masked(ax1, mask_autoclear, 'black', 'stable pole')
        self.order_dummy.mask = np.ma.nomask
        self.masked_frequencies.mask = np.ma.nomask

        ax1.autoscale_view(tight=True)
        ax1.set_ylabel('Model order [-]')
        ax1.set_title('Stabilization Diagram')
        ax1.set_ylim((0, 200))

        for mask in self.masks['mask_autosel']:
            self.masked_frequencies.mask = mask
            plot.axvspan(self.masked_frequencies.min(), self.masked_frequencies.max(),
                         facecolor='blue', alpha=.3, edgecolor='none')
        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        for mode in self.select_modes:
            f = self.modal_data.modal_frequencies[mode]
            n = self.order_dummy[mode]
            ax1.scatter(f, n, facecolors='none', marker='o', edgecolors='red', s=10)

        num_poles, fpoles = [], []
        for clusternr in range(1, 1 + max(self.cluster_assignments)):
            flat_poles_ind = self.cluster_assignments != clusternr
            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.masked_frequencies.mask = mask
            num_poles.append(np.prod(mask.shape) - np.sum(mask))
            fpoles.append(np.ma.mean(self.masked_frequencies))

        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.bar(fpoles, num_poles, width=0.01, align='center', edgecolor='none')
        ax2.axhline(self.selection_cut_off, c='r', ls='--', linewidth=2)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Nr. of elements')
        ax2.set_title('Clusters')
        plot.tight_layout()
        plot.xlim((0, self.prep_signals.sampling_rate / 2))
        if save_path is not None:
            plot.savefig(save_path + 'select_clusters.pdf')
        else:
            plot.show()
            plot.pause(0.001)

    def _scatter_masked(self, ax, mask, color, label):
        """Scatter plot of poles using *mask*, colored by *color*."""
        self.masked_frequencies.mask = mask
        self.order_dummy.mask = mask
        ax.scatter(
            self.masked_frequencies.compressed(),
            self.order_dummy.compressed(),
            marker='o', facecolors='none', edgecolors=color, s=10, label=label)

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

            n, f, stdf, d, stdd, mpc, mp, mpd, _dmp, _dmpd, _mtn, MC, _ex_1, _ex_2 = self.get_modal_values(
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


        return np.array(all_n), np.array(all_f), np.array(all_std_f), np.array(all_d), np.array(
            all_std_d), np.array(all_MPC), np.array(all_MP), np.array(all_MPD), np.array(all_MC), np.array(all_msh),


class StabilPlot(object):
    """Static matplotlib stabilisation diagram renderer.

    Draws poles from a :class:`StabilCalc` object onto a matplotlib figure,
    colour-coded by their stabilisation status (stable / partially stable /
    unstable).  Used as the backend for both the interactive
    :class:`~pyOMA.GUI.StabilGUI.StabilGUI` and the Jupyter
    :class:`~pyOMA.GUI.JupyterGUI.JupyterGUI`.

    Parameters
    ----------
    stabil_calc : StabilCalc
        Populated stabilisation-calculation object.
    fig : matplotlib.figure.Figure, optional
        External figure to draw into.  A new figure is created when ``None``.
    """

    def __init__(self, stabil_calc, fig=None):
        """
        Parameters
        ----------
        stabil_calc : StabilCalc
            Populated stabilisation-calculation object.
        fig : matplotlib.figure.Figure, optional
            External figure to draw into.  A new figure is created when
            ``None``.
        """
        super().__init__()

        if not isinstance(stabil_calc, StabilCalc):
            logger.warning(f'Argument stabil_calc is wrong object type {type(stabil_calc)}')
        self.stabil_calc = stabil_calc
        if fig is None:
            self.fig = Figure(facecolor='white')  # , dpi=100, figsize=(16, 12))
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

            # self.colors['plot_mpc'] = 'grey'
            # self.colors['plot_mpd']=  'grey'
            # self.colors['plot_dmac']=  'black'

            # self.labels['plot_mpc']=   'modal phase collinearity criterion'
            # self.labels['plot_mpd']=   'mean phase deviation criterion'
            # self.labels['plot_dmac']=  'unstable in mac'

            # self.markers['plot_mpc']=   TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            # self.markers['plot_mpd']=   TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            # self.markers['plot_dmac']=  TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)

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
                self.add_mode(mode, list_ind)

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

    def _update_criterion_plots(self, criteria):
        caps = self.stabil_calc.capabilities
        criterion_plot_map = [
            ('stdf_max', 'std', 'plot_stdf'),
            ('stdd_max', 'std', 'plot_stdd'),
            ('mtn_min', 'mtn', 'plot_mtn'),
            ('dmtn_min', 'mtn', 'plot_dmtn'),
            ('MC_min', 'MC', 'plot_MC'),
        ]
        for criterion, cap, plot_name in criterion_plot_map:
            if criterion in criteria and caps[cap]:
                self.plot_stabil(plot_name)

    def _update_auto_plots(self):
        if not self.stabil_calc.capabilities['auto']:
            return
        if self.stabil_calc.state >= 3 and not self.stabil_calc.use_stabil:
            self.plot_stabil('plot_autoclear')
        if self.stabil_calc.state >= 5:
            self.plot_stabil('plot_autosel')

    def update_stabilization(self, **criteria):
        self.stabil_calc.update_stabilization_masks(**criteria)
        self._update_criterion_plots(criteria)
        self.plot_stabil('plot_pre')
        self.plot_stabil('plot_stable')
        self._update_auto_plots()
        if self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdf')
        if self.cursor:
            cursor_name_mask = self.cursor.name_mask
            cursor_mask = self.stabil_calc.get_stabilization_mask(
                cursor_name_mask)
            self.cursor.set_mask(cursor_mask, cursor_name_mask)

    def plot_stabil_autosel(self, color, marker, zorder, size, label):
        name = 'plot_autosel'
        if self.stable_plot[name] is not None:
            for plot in self.stable_plot[name]:
                plot.remove()

        visibility = True
        masks = self.stabil_calc.masks['mask_autosel']

        colors = list(matplotlib.cm.gist_rainbow(
                np.linspace(
                    0, 1, len(masks))))  # @UndefinedVariable
        shuffle(colors)
        self.stable_plot[name] = []
        for color, mask in zip(colors, masks):
            self.stabil_calc.masked_frequencies.mask = mask
            self.stabil_calc.order_dummy.mask = mask
            self.stable_plot[name].append(self.ax.scatter(
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

        return

    def plot_stabil_stdf(self, name, color, zorder, label):
        if self.stable_plot[name] is not None:
            try:
                children = self.stable_plot[name].get_children()
                if children:
                    visibility = children[0].get_visible()
                    self.stable_plot[name].remove()
            except IndexError as e:
                logger.debug(f'Failed to remove stabil_stdf objects {e}')
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
            self.stable_plot[name] = self.ax.errorbar(self.stabil_calc.masked_frequencies.compressed(),
                self.stabil_calc.order_dummy.compressed(),
                xerr=scipy.stats.t.ppf(0.95, num_blocks) * std_error, zorder=zorder,
                fmt='none', ecolor=color, label=label, visible=visibility)
        return

    def plot_stabil(self, name):
        # print(name)
        color = self.colors[name]
        marker = self.markers[name]
        # print(marker, name)
        zorder = self.zorders[name]
        size = self.sizes[name]
        label = self.labels[name]

        if name == 'plot_autosel':
            self.plot_stabil_autosel(color, marker, zorder, size, label)

        elif name == 'plot_stdf':
            self.plot_stabil_stdf(name, color, zorder, label)

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
                sum_mc = np.sum(
                    self.stabil_calc.modal_data.modal_contributions[order,:])
                if np.iscomplex(sum_mc):
                    # abs used for complex modal contributions (pLSCF)
                    sum_mc = np.abs(sum_mc)
                MCs[order] = sum_mc
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

    def _handle_existing_psd_lines(self, b, NFFT):
        """Toggle visibility or remove existing PSD lines.

        Returns True if the caller should return immediately, None if no plot exists.
        """
        if not self.psd_plot:
            return None
        if not b or NFFT == self.stabil_calc.prep_signals.n_lines:
            for channel in self.psd_plot:
                for line in channel:
                    line._visible = b
            self.fig.canvas.draw_idle()
            return True
        for channel in self.psd_plot:
            for line in channel:
                line.remove()
        self.psd_plot = []
        return False

    def plot_sv_psd(self, b, NFFT=None):
        '''
         .. TODO::
             * add GUI for choosing PSD parameters
         '''
        if self._handle_existing_psd_lines(b, NFFT) is True:
            return
        if self.stabil_calc.prep_signals is None:
            raise RuntimeError('Measurement Data was not provided!')
        if not b:
            return
        if NFFT is None and self.stabil_calc.prep_signals.n_lines is None:
            NFFT = 2048
        sv_psd = self.stabil_calc.prep_signals.sv_psd(NFFT)
        freq_psd = self.stabil_calc.prep_signals.freqs
        sv_psd_db_scaled = 10 * np.log10(sv_psd)
        sv_psd_db_scaled -= np.min(sv_psd_db_scaled)
        sv_psd_db_scaled /= 2 * np.max(sv_psd_db_scaled)
        n_channels = sv_psd.shape[0]
        for channel in range(n_channels):
            self.psd_plot.append(self.ax.plot(
                freq_psd, sv_psd_db_scaled[channel, :], color='grey',
                alpha=(n_channels - channel) / n_channels,
                linestyle='solid', visible=b,
                zorder=-1, transform=self.ax.get_xaxis_transform()))
        self.fig.canvas.draw_idle()

    def update_xlim(self, xlim):
        self.ax.set_xlim(xlim)
        self.fig.canvas.draw_idle()

    def update_ylim(self, ylim):
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()


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
        for obj in plot_obj.get_children():
            if obj is None:
                continue
            obj.set_visible(b)
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
            # print('plot_pre not found')
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
                self.fig.canvas.print_figure(str(fname))
            except Exception:
                import traceback
                traceback.print_exc()

    def mode_selected(self, event):
        '''
        connect this function to the button press event of the canvas
        
        '''

        if event.name == "button_press_event" and event.inaxes == self.ax:

            # Check if in zooming or panning mode; credit: https://stackoverflow.com/questions/48446351/
            zooming_panning = False
            try:  # Qt Backend
                zooming_panning = (self.fig.canvas.cursor().shape() != 0)  # 0 is the arrow, which means we are not zooming or panning.
            except Exception: pass
            try:  # nbAgg Backend
                zooming_panning = str(self.fig.canvas.toolbar.cursor) != 'Cursors.POINTER'
            except Exception: pass
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

        if len(self.scatter_objs) <= list_ind:
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
        # if self.useblit:
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

    # show_current_info = pyqtSignal(tuple)
    # mode_selected = pyqtSignal(tuple)
    # mode_deselected = pyqtSignal(tuple)

    def __init__(
            self,
            ax,
            order_data,
            f_data,
            mask=None,
            useblit=True,
            datalist=None,
            **lineprops):
        if datalist is None:
            datalist = []

        Cursor.__init__(self, ax, useblit=useblit, **lineprops)
        # QObject.__init__(self)
        self.callbacks = {'show_current_info':lambda *args, **kwargs: None,
                          'mode_selected':lambda *args, **kwargs: None,
                          'mode_deselected':lambda *args, **kwargs: None, }
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

    def add_callback(self, name, func):
        if name not in self.callbacks:
            raise ValueError(f"Unknown callback {name!r}. Known: {list(self.callbacks)}.")
        self.callbacks[name] = func

    def set_mask(self, mask, name):
        self.mask = mask
        self.fig_resized()
        self.name_mask = name

    def fig_resized(self, event=None):
        # self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)

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
        # 1. Override event.data to force it to snap-to nearest data item
        # 2. On a mouse-click, select the data item and append it to a list of selected items
        # 3. The second mouse-click on a previously selected item, removes it from the list
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

        Cursor.onmove(self, event)
        # for scatter in self.scatter_objs: scatter.set_visible(False)

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
            int(a * 10 ** sig_fig) == int(b * 10 ** sig_fig)
            )


if __name__ == '__main__':
    pass
