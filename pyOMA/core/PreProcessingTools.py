# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Signal pre-processing: GeometryProcessor, PreProcessSignals, SignalPlot."""
import os
import csv
import datetime

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from .Helpers import nearly_equal, simplePbar, validate_array, ConfigFile

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class GeometryProcessor(object):
    """Stores structural geometry for mode-shape visualisation.

    Holds node coordinates, structural connectivity lines, and parent-child
    (skewed-sensor) relationships.  Passed to
    :class:`~pyOMA.core.PlotMSH.ModeShapePlot` after loading with
    :meth:`load_geometry`.

    Parameters
    ----------
    nodes : dict, optional
        Mapping ``{node_name: (x, y, z)}``.
    lines : list of (str, str), optional
        Connectivity list ``[(node_start, node_end), ...]``.
    parent_childs : list of tuple, optional
        Skewed-sensor parent-child relations, each entry
        ``(parent_node, x_amp, y_amp, z_amp, child_node, x_amp, y_amp, z_amp)``.

    Notes
    -----
    Conventions:

    * ``chan_dofs = [(chan, node, (x_amplif, y_amplif, z_amplif)), ...]``
    * Channels are numbered ``0 ... N-1`` (complete sequence).
    * Node names are strings; coordinates are ``(x, y, z)`` float tuples.
    * Lines are unordered pairs ``(node_start, node_end)``.
    * Parent-child entries are 8-tuples as described in *Parameters*.

    .. TODO::
         * change parent_child assignment to skewed coordinate
         * change parent_childs to az, elev
    """

    def __init__(self, nodes=None, lines=None, parent_childs=None):
        if nodes is None:
            nodes = {}
        if lines is None:
            lines = []
        if parent_childs is None:
            parent_childs = []
        super().__init__()
        self.nodes = {}
        if not isinstance(nodes, dict):
            raise TypeError(f"nodes must be dict, got {type(nodes).__name__!r}")
        self.add_nodes(nodes)

        self.lines = []
        if not isinstance(lines, (list, tuple, np.ndarray)):
            raise TypeError(f"lines must be list, tuple, or ndarray, got {type(lines).__name__!r}")
        self.add_lines(lines)

        self.parent_childs = []
        if not isinstance(parent_childs, (list, tuple, np.ndarray)):
            raise TypeError(f"parent_childs must be list, tuple, or ndarray, got {type(parent_childs).__name__!r}")
        self.add_parent_childs(parent_childs)

    @staticmethod
    def nodes_loader(filename):
        '''
        nodes file uses one header line
        tab-separated file
        node is treated as a string
        x,y,z are treated as floats (in scientific format)
        '''
        nodes = {}
        with open(filename, 'r') as f:
            f.__next__()
            for line1 in csv.reader(f, delimiter='\t', skipinitialspace=True):
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                node, x, y, z = [float(line[i]) if i >= 1 else line[i].strip(
                    ' ') for i in range(4)]  # cut trailing empty columns
                nodes[node] = [x, y, z]
        return nodes

    @staticmethod
    def lines_loader(filename):
        '''
        lines file uses one header line
        tab-separated file
        nodenames are treated as strings
        '''
        lines = []
        with open(filename, 'r') as f:
            f.__next__()
            for line1 in csv.reader(f, delimiter='\t', skipinitialspace=True):
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                node_start, node_end = \
                    [line[i] for i in range(2)]  # cut trailing empty columns
                lines.append((node_start, node_end))
        return lines

    @staticmethod
    def parent_childs_loader(filename):
        '''
        lines file uses one header line
        tab-separated file
        nodenames are treated as strings
        amplification factors are treated as floats
        '''
        parent_childs = []
        with open(filename, 'r') as f:
            f.__next__()
            reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
            for line1 in reader:
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl = [
                    float(line[i]) if i not in [0, 4] else line[i].strip(' ') for i in range(8)]
                parent_childs.append(
                    (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl))
        return parent_childs

    @classmethod
    def load_geometry(
            cls,
            nodes_file,
            lines_file=None,
            parent_childs_file=None):
        """Load geometry from tab-separated text files.

        Parameters
        ----------
        nodes_file : str
            Path to the nodes file (node name + x, y, z coordinates,
            tab-separated, one header line).
        lines_file : str, optional
            Path to the lines file (start_node, end_node pairs,
            tab-separated).
        parent_childs_file : str, optional
            Path to the parent-child file describing skewed-sensor
            relationships.

        Returns
        -------
        GeometryProcessor
            Populated geometry object ready to pass to
            :class:`~pyOMA.core.PlotMSH.ModeShapePlot`.
        """

        geometry_data = cls()

        nodes = geometry_data.nodes_loader(nodes_file)
        geometry_data.add_nodes(nodes)

        if lines_file is not None and os.path.exists(lines_file):
            lines = geometry_data.lines_loader(lines_file)
            geometry_data.add_lines(lines)

        if parent_childs_file is not None and os.path.exists(parent_childs_file):
            parent_childs = geometry_data.parent_childs_loader(
                parent_childs_file)
            geometry_data.add_parent_childs(parent_childs)

        return geometry_data

    def add_nodes(self, nodes):
        for item in nodes.items():
            try:
                self.add_node(*item)
            except BaseException:
                logger.warning(
                    'Something was wrong while adding node {}. Continuing!'.format(item))
                continue

    def add_node(self, node_name, coordinate_list):
        node_name = str(node_name)
        if node_name in self.nodes.keys():
            logger.warning('Node {} is already defined. Overwriting.'.format(node_name))

        if not isinstance(coordinate_list, (list, tuple)):
            raise RuntimeError(
                'Coordinates must be provided as (x,y,z) tuples/lists.')
        if len(coordinate_list) != 3:
            raise RuntimeError(
                'Coordinates must be provided as (x,y,z) tuples/lists.')

        try:
            node_name = str(node_name)
            coordinate_list = list(coordinate_list)
            for i in range(3):
                coordinate_list[i] = float(coordinate_list[i])
        except ValueError:
            raise RuntimeError(
                'Coordinate {} at position {} could not be converted to float.'.format(
                    coordinate_list[i], i))
        except BaseException:
            raise

        self.nodes[node_name] = tuple(coordinate_list)

    def _remove_lines_for_node(self, node_name):
        """Remove all lines connected to *node_name* from ``self.lines``."""
        while True:
            for j in range(len(self.lines)):
                if node_name in self.lines[j]:
                    del self.lines[j]
                    break
            else:
                break

    def _remove_parent_childs_for_node(self, node_name):
        """Remove all parent-child entries that reference *node_name*."""
        while True:
            for j, parent_child in enumerate(self.parent_childs):
                if node_name == parent_child[0] or node_name == parent_child[4]:
                    del self.parent_childs[j]
                    break
            else:
                break

    def take_node(self, node_name):
        if node_name not in self.nodes:
            logger.warning('Node not defined. Exiting')
            return

        self._remove_lines_for_node(node_name)
        self._remove_parent_childs_for_node(node_name)
        del self.nodes[node_name]

        logger.info('Node {} removed.'.format(node_name))

    def add_lines(self, lines):

        for line in lines:
            try:
                self.add_line(line)
            except BaseException:
                logger.warning(
                    'Something was wrong while adding line {}. Continuing!'.format(line))
                continue

    def add_line(self, line):
        if not isinstance(line, (list, tuple)):
            raise RuntimeError(
                'Line has to be provided in format (start_node, end_node).')
        if len(line) != 2:
            raise RuntimeError(
                'Line has to be provided in format (start_node, end_node).')

        line = [str(line[0]), str(line[1])]
        if line[0] not in self.nodes or line[1] not in self.nodes:
            logger.warning('One of the end-nodes of line {} not defined!'.format(line))
        else:
            for line_ in self.lines:
                if line_[0] == line[0] and line_[1] == line[1]:
                    logger.info('Line {} was defined, already.'.format(line))
            self.lines.append(line)

    def take_line(self, line=None, line_ind=None):
        if line is not None and line_ind is not None:
            raise ValueError("At most one of 'line' or 'line_ind' may be specified, not both.")

        if line is not None:
            for line_ind in range(len(self.lines)):
                line_ = self.lines[line_ind]
                if line[0] == line_[0] and line[1] == line_[1]:
                    break
            else:
                logger.warning('Line {} was not found.'.format(line))
                return
        del self.lines[line_ind]
        logger.info('Line {} at index {} removed.'.format(line, line_ind))

    def add_parent_childs(self, parent_childs):
        for ms in parent_childs:
            try:
                self.add_parent_child(ms)
            except BaseException:
                logger.warning(
                    'Something was wrong while adding parent-child-definition {}. Continuing!'.format(ms))
                continue

    def _validate_parent_child_format(self, ms):
        """Validate format of a parent-child definition tuple."""
        if not isinstance(ms, (list, tuple)):
            raise RuntimeError(
                'parent child definition has to be provided in format (start_node, end_node).')
        if len(ms) != 8:
            raise RuntimeError(
                'parent child definition has to be provided in format (parent_node, x_ampli, y_ampli, z_ampli, child_node, x_ampli, y_ampli, z_ampli).')
        return (
            str(ms[0]), float(ms[1]), float(ms[2]), float(ms[3]),
            str(ms[4]), float(ms[5]), float(ms[6]), float(ms[7]))

    def _check_parent_child_duplicate(self, ms):
        """Log a warning if *ms* is already present in ``self.parent_childs``."""
        for ms_ in self.parent_childs:
            if all(ms_[i] == ms[i] for i in range(8)):
                logger.info(
                    'parent child definition {} was defined, already.'.format(ms))
                break

    def add_parent_child(self, ms):
        ms = self._validate_parent_child_format(ms)
        if ms[0] not in self.nodes or ms[4] not in self.nodes:
            logger.warning(
                'One of the nodes of parent child definition {} not defined!'.format(ms))
        else:
            self._check_parent_child_duplicate(ms)
            self.parent_childs.append(ms)

    def take_parent_child(self, ms=None, ms_ind=None):
        if ms is not None and ms_ind is not None:
            raise ValueError("At most one of 'ms' or 'ms_ind' may be specified, not both.")

        if ms is not None:
            for ms_ind in range(len(self.parent_childs)):
                ms_ = self.parent_childs[ms_ind]
                b = False
                for i in range(8):
                    b = b and ms_[i] == ms[i]
                if b:
                    break
            else:
                logger.warning('parent child definition {} was not found.'.format(ms))
                return

        del self.parent_childs[ms_ind]
        logger.info('parent child definition {} at index {} removed.'.format(ms, ms_ind))

    def rescale_geometry(self, factor):
        pass


class PreProcessSignals(object):
    """Pre-processor for multi-channel ambient-vibration signals.

    Provides signal conditioning (filtering, decimation, offset removal,
    scaling), spectral estimation (Welch, Blackman-Tukey), and book-keeping of
    channel metadata (reference channels, measurement quantities, channel-DOF
    assignments).  All downstream pyOMA modules (system identification,
    stabilisation diagram, mode-shape visualisation) consume an instance of
    this class.

    .. TODO::
        * time-step integration of signals
        * Multi-block Blackman-Tukey PSD
    """

    def _validate_inputs(self, signals, sampling_rate, F):
        """Validate constructor arguments for signals, sampling_rate, and F."""
        if not isinstance(signals, np.ndarray):
            raise TypeError(f"signals must be a numpy ndarray, got {type(signals)}")
        if signals.shape[0] <= signals.shape[1]:
            raise ValueError(
                f"signals must have more rows (time steps) than columns (channels); "
                f"got shape {signals.shape}")
        if not isinstance(sampling_rate, (int, float)):
            raise TypeError(f"sampling_rate must be a number, got {type(sampling_rate)}")
        if F is not None and not isinstance(F, np.ndarray):
            raise TypeError(f"F must be a numpy ndarray, got {type(F)}")

    def _resolve_quantity_defaults(self, accel_channels, velo_channels, disp_channels):
        """Return (accel, velo, disp) channel lists with None replaced by defaults."""
        if disp_channels is None:
            disp_channels = []
        if velo_channels is None:
            velo_channels = []
        if accel_channels is None:
            accel_channels = [c for c in range(self.num_analised_channels)
                              if c not in disp_channels and c not in velo_channels]
        return accel_channels, velo_channels, disp_channels

    def _warn_undefined_quantities(self, accel_channels, velo_channels, disp_channels):
        """Warn if any channel is not assigned to exactly one quantity."""
        for chan in range(self.num_analised_channels):
            n_assigned = (
                (chan in accel_channels)
                + (chan in velo_channels)
                + (chan in disp_channels)
            )
            if n_assigned != 1:
                logger.warning(f'Quantity of channel {chan} is not defined.')

    def _setup_channel_mapping(self, accel_channels, velo_channels, disp_channels):
        """Initialise the channel-quantity mappings (accel/velo/disp)."""
        self._accel_channels = []
        self._velo_channels = []
        self._disp_channels = []

        accel_channels, velo_channels, disp_channels = self._resolve_quantity_defaults(
            accel_channels, velo_channels, disp_channels)

        self._warn_undefined_quantities(accel_channels, velo_channels, disp_channels)

        self.accel_channels = accel_channels
        self.velo_channels = velo_channels
        self.disp_channels = disp_channels

    def _setup_metadata(self, setup_name, channel_headers, start_time):
        """Validate and store setup_name, channel_headers, and start_time."""
        if setup_name is None:
            setup_name = ''
        if not isinstance(setup_name, str):
            raise TypeError(f"setup_name must be str, got {type(setup_name).__name__!r}")
        self.setup_name = setup_name

        if channel_headers is not None:
            if len(channel_headers) != self.num_analised_channels:
                raise ValueError(
                    f"channel_headers must have length {self.num_analised_channels} "
                    f"(num_analised_channels), got {len(channel_headers)}"
                )
        else:
            channel_headers = list(range(self.num_analised_channels))
        self.channel_headers = channel_headers

        if start_time is not None:
            if not isinstance(start_time, datetime.datetime):
                raise TypeError(
                    f"start_time must be datetime.datetime, got {type(start_time).__name__!r}")
        else:
            start_time = datetime.datetime.now()
        self.start_time = start_time

    def _init_spectral_state(self):
        """Initialise all spectral estimation result attributes to None."""
        self.corr_matrix_wl = None
        self.corr_matrices_wl = None
        self.var_corr_wl = None

        self.psd_matrix_wl = None
        self.psd_matrices_wl = None
        self.var_psd_wl = None

        self.n_lines_wl = None
        self.m_lags_wl = None
        self.n_segments_wl = None

        self.corr_matrix_bt = None
        self.corr_matrices_bt = None
        self.var_corr_bt = None

        self.psd_matrix_bt = None
        self.psd_matrices_bt = None
        self.var_psd_bt = None

        self.n_lines_bt = None
        self.m_lags_bt = None
        self.n_segments_bt = None

        # self.s_vals_cf = None
        self.s_vals_psd = None

    def __init__(self, signals, sampling_rate, ref_channels=None,
                 accel_channels=None, velo_channels=None, disp_channels=None,
                 **kwargs):
        """
        Parameters
        ----------
        signals : np.ndarray, shape (n_samples, n_channels)
            Raw measurement time series; must have more rows than columns.
        sampling_rate : float
            Sampling frequency in Hz.
        ref_channels : list of int, optional
            Column indices of reference (fixed) sensors.  Defaults to all
            channels.
        accel_channels : list of int, optional
            Column indices of acceleration channels.  Defaults to all channels
            not in *velo_channels* or *disp_channels*.
        velo_channels : list of int, optional
            Column indices of velocity channels.
        disp_channels : list of int, optional
            Column indices of displacement channels.

        Other Parameters
        ----------------
        setup_name : str, optional
            Label for this measurement setup.
        channel_headers : list of str, optional
            Human-readable channel names; defaults to ``[0, 1, 2, ...]``.
        start_time : datetime.datetime, optional
            Measurement start timestamp; defaults to ``datetime.datetime.now()``.
        F : np.ndarray, optional
            Optional forcing signal array (used only for FRF-based ERA).
        """
        start_time = kwargs.pop('start_time', None)
        F = kwargs.pop('F', None)
        setup_name = kwargs.pop('setup_name', None)
        channel_headers = kwargs.pop('channel_headers', None)

        super().__init__()

        self._validate_inputs(signals, sampling_rate, F)
        self.signals = np.copy(signals)
        self.signals_filtered = np.copy(signals)
        self.sampling_rate = sampling_rate
        self.F = F

        self._ref_channels = None
        if ref_channels is None:
            ref_channels = list(range(signals.shape[1]))
        self.ref_channels = ref_channels

        self._setup_channel_mapping(accel_channels, velo_channels, disp_channels)

        self._setup_metadata(setup_name, channel_headers, start_time)

        self.chan_dofs = []
        self.channel_factors = [1 for _ in range(self.num_analised_channels)]
        self.scaling_factors = None
        self._last_meth = None

        self._init_spectral_state()

    @classmethod
    def _load_chan_dofs_and_update_headers(cls, chan_dofs_file, headers):
        """Load chan_dofs from file if given and update headers in-place; return chan_dofs."""
        if chan_dofs_file is not None:
            chan_dofs = cls.load_chan_dofs(chan_dofs_file)
        else:
            chan_dofs = None
        if chan_dofs is not None:
            cls._apply_chan_dof_headers(chan_dofs, headers)
        return chan_dofs

    @classmethod
    def _apply_delete_if_needed(cls, signals, chan_dofs, delete_channels,
                                ref_channels, motion_channels, headers):
        """Apply delete_channels if any; return updated tuple."""
        accel_channels, velo_channels, disp_channels = motion_channels
        if delete_channels:
            return cls._apply_delete_channels(
                signals, chan_dofs, delete_channels,
                ref_channels, accel_channels, velo_channels, disp_channels,
                headers)
        return (signals, chan_dofs, ref_channels,
                accel_channels, velo_channels, disp_channels, headers)

    @classmethod
    def init_from_config(
            cls,
            conf_file,
            meas_file,
            chan_dofs_file=None,
            **kwargs):
        '''
        initializes the PreProcessor object with a configuration file

        to remove channels at loading time use 'usecols' keyword argument
        if delete_channels are specified, these will be checked against
        all other channel definitions, which will be adjusted accordingly
        '''
        cfg = ConfigFile(conf_file)
        name = cfg.str('Setup Name')
        sampling_rate = cfg.float('Sampling Rate [Hz]')
        ref_channels = cfg.int_list('Reference Channels')
        delete_channels = cfg.int_list('Delete Channels')
        accel_channels = cfg.int_list('Accel. Channels')
        velo_channels = cfg.int_list('Velo. Channels')
        disp_channels = cfg.int_list('Disp. Channels')

        loaded_signals = cls.load_measurement_file(meas_file, **kwargs)
        signals, headers, start_time = cls._resolve_signals_and_headers(
            loaded_signals, sampling_rate)

        chan_dofs = cls._load_chan_dofs_and_update_headers(chan_dofs_file, headers)

        (signals, chan_dofs, ref_channels,
         accel_channels, velo_channels,
         disp_channels, headers) = cls._apply_delete_if_needed(
            signals, chan_dofs, delete_channels,
            ref_channels, (accel_channels, velo_channels, disp_channels), headers)

        num_channels = signals.shape[1]
        if not accel_channels and not velo_channels and not disp_channels:
            accel_channels = [i for i in range(num_channels)]

        prep_signals = cls(signals, sampling_rate,
                           ref_channels,
                           accel_channels, velo_channels, disp_channels,
                           setup_name=name, channel_headers=headers,
                           start_time=start_time,
                           **kwargs)
        if chan_dofs:
            prep_signals.add_chan_dofs(chan_dofs)

        return prep_signals

    @staticmethod
    def _resolve_signals_and_headers(loaded_signals, sampling_rate):
        """Unpack loaded signals; return (signals, headers, start_time)."""
        if not isinstance(loaded_signals, np.ndarray):
            headers, _, start_time, sample_rate, signals = loaded_signals
        else:
            signals = loaded_signals
            start_time = datetime.datetime.now()
            sample_rate = sampling_rate
            headers = ['Channel_{}'.format(i) for i in range(signals.shape[1])]
        if not sample_rate == sampling_rate:
            logger.warning(
                'Sampling Rate from file: {} does not correspond with specified '
                'Sampling Rate from configuration {}'.format(sample_rate, sampling_rate))
        return signals, headers, start_time

    @staticmethod
    def _apply_chan_dof_headers(chan_dofs, headers):
        """Update *headers* in-place from channel names stored in *chan_dofs*."""
        for chan_dof in chan_dofs:
            if len(chan_dof) == 5:
                chan = chan_dof[0]
                chan_name = chan_dof[4]
                if len(chan_name) == 0:
                    continue
                elif headers[chan] == 'Channel_{}'.format(chan):
                    headers[chan] = chan_name
                elif headers[chan] != chan_name:
                    logger.info(
                        'Different headers for channel {} in signals file ({}) '
                        'and in channel-DOF-assignment ({}).'.format(
                            chan, headers[chan], chan_name))

    @staticmethod
    def _find_chan_dof_entry(chan_dofs, channel):
        """Find the chan_dof entry for *channel*; return (node, az, elev, cname) or None."""
        for chan_dof in chan_dofs:
            if chan_dof[0] == channel:
                node, az, elev = chan_dof[1:4]
                cname = chan_dof[4] if len(chan_dof) == 5 else ''
                return node, az, elev, cname
        return None

    @staticmethod
    def _apply_delete_channels(signals, chan_dofs, delete_channels,
                                ref_channels, accel_channels, velo_channels,
                                disp_channels, headers):
        """Remove *delete_channels* from all channel lists and the signal array."""
        num_all_channels = signals.shape[1]
        new_chan_dofs = []
        new_ref_channels = []
        new_accel_channels = []
        new_velo_channels = []
        new_disp_channels = []
        new_headers = []
        new_channel = 0
        for channel in range(num_all_channels):
            if channel in delete_channels:
                logger.info('Now removing Channel {} (no. {})!'.format(
                    headers[channel], channel))
                continue
            entry = PreProcessSignals._find_chan_dof_entry(chan_dofs, channel)
            if entry is None:
                logger.warning('Could not find channel in chan_dofs')
                continue
            node, az, elev, cname = entry
            new_chan_dofs.append([new_channel, node, az, elev, cname])
            if channel in ref_channels:
                new_ref_channels.append(new_channel)
            if channel in accel_channels:
                new_accel_channels.append(new_channel)
            if channel in velo_channels:
                new_velo_channels.append(new_channel)
            if channel in disp_channels:
                new_disp_channels.append(new_channel)
            new_headers.append(headers[channel])
            new_channel += 1

        signals = np.delete(signals, delete_channels, axis=1)
        return (signals, new_chan_dofs, new_ref_channels,
                new_accel_channels, new_velo_channels, new_disp_channels,
                new_headers)

    @staticmethod
    def load_chan_dofs(fname):
        '''
        chan_dofs[i] = (chan_num, node_name, az, elev, chan_name)
                    = (int,       str,       float,float, str)

        azimuth angle starting from x axis towards y axis
        elevation defined from x-y plane up
        x: 0.0, 0.0
        y: 90.0, 0.0
        z: 0.0, 90.0
        channels not  present in the file will be removed later
        nodes do not have to, but should exist, as this information is
        also used for merging multiple setups, which does not rely on
        any "real" geometry
        '''
        chan_dofs = []
        with open(fname, 'r') as f:
            f.__next__()
            for line1 in csv.reader(f, delimiter='\t', skipinitialspace=True):
                line = []
                for val in line1:
                    if not val:
                        continue
                    line += val.split()
                if not line:
                    continue
                if line[0].startswith('#'):
                    break
                entry = PreProcessSignals._parse_chan_dof_line(line)
                if entry is not None:
                    chan_dofs.append(entry)
        return chan_dofs

    @staticmethod
    def _parse_chan_dof_line(line):
        """Parse one tab-split line from a chan_dofs file into a 5-element list."""
        while len(line) <= 5:
            line.append('')
        chan_num, node, az, elev, chan_name = [line[i].strip(' ') for i in range(5)]
        chan_num, az, elev = int(float(chan_num)), float(az), float(elev)
        if node == 'None':
            node = None
        return [chan_num, node, az, elev, chan_name]

    @staticmethod
    def load_measurement_file(fname, **kwargs):
        '''
        A method for loading a signals file

        Parameters
        ----------
        fname : str
                The full path of the signals file

        Returns
        -------
        headers : list of str
                The names of all channels
        units : list of str
                The units of all channels
        start_time : datetime.datetime
                The starting time of the measured signal
        sample_rate : float
                The sample rate, at wich the signal was acquired
        signals : ndarray
                Array of shape (num_timesteps, num_channels) which contains
                the acquired signal
        '''

        raise NotImplementedError(
            'This method must be provided by the user for each specific analysis task and assigned to the class before instantiating the instance.')

    def add_chan_dofs(self, chan_dofs):
        '''
        chan_dofs = [ (chan_num, node_name, az, elev, chan_name) ,  ... ]
        This function is not checking if channels or nodes actually exist
        the former should be added
        the latter might only be possible, if the geometry object is known to the class
        
        '''
        for chan_dof in chan_dofs:
            chan_dof[0] = int(chan_dof[0])
            if chan_dof[1] is not None:
                chan_dof[1] = str(chan_dof[1])
            chan_dof[2] = float(chan_dof[2])
            chan_dof[3] = float(chan_dof[3])
            if len(chan_dof) == 4:
                chan_dof.append('')
            self.chan_dofs.append(chan_dof)
        # self.chan_dofs=chan_dofs

    def take_chan_dof(self, chan, node, dof):

        for j in range(len(self.chan_dofs)):
            if self.chan_dofs[j][0] == chan and \
               self.chan_dofs[j][1] == node and \
               nearly_equal(self.chan_dofs[j][2][0], dof[0], 3) and \
               nearly_equal(self.chan_dofs[j][2][1], dof[1], 3) and \
               nearly_equal(self.chan_dofs[j][2][2], dof[2], 3):
                del self.chan_dofs[j]
                break
        else:
            if self.chan_dofs:
                logger.warning('chandof not found')

    def save_state(self, fname):

        # print('fname = ', fname)
        logger.info('Saving results to  {}'.format(fname))

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {}

        out_dict['self.signals'] = self.signals
        out_dict['self.sampling_rate'] = self.sampling_rate
        out_dict['self.ref_channels'] = self._ref_channels
        out_dict['self.accel_channels'] = self._accel_channels
        out_dict['self.velo_channels'] = self._velo_channels
        out_dict['self.disp_channels'] = self._disp_channels

        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.channel_headers'] = self.channel_headers
        out_dict['self.start_time'] = self.start_time

        out_dict['self.chan_dofs'] = self.chan_dofs
        out_dict['self.scaling_factors'] = self.scaling_factors
        out_dict['self.channel_factors'] = self.channel_factors
        out_dict['self._last_meth'] = self._last_meth

        out_dict['self.corr_matrix_wl'] = self.corr_matrix_wl
        out_dict['self.corr_matrices_wl'] = self.corr_matrices_wl
        out_dict['self.psd_matrix_wl'] = self.psd_matrix_wl
        out_dict['self.psd_matrices_wl'] = self.psd_matrices_wl
        out_dict['self.var_corr_wl'] = self.var_corr_wl
        out_dict['self.var_psd_wl'] = self.var_psd_wl
        out_dict['self.n_lines_wl'] = self.n_lines_wl
        out_dict['self.m_lags_wl'] = self.m_lags_wl
        out_dict['self.n_segments_wl'] = self.n_segments_wl

        out_dict['self.corr_matrix_bt'] = self.corr_matrix_bt
        out_dict['self.corr_matrices_bt'] = self.corr_matrices_bt
        out_dict['self.psd_matrix_bt'] = self.psd_matrix_bt
        out_dict['self.n_lines_bt'] = self.n_lines_bt
        out_dict['self.m_lags_bt'] = self.m_lags_bt
        out_dict['self.n_segments_bt'] = self.n_segments_bt

        out_dict['self.var_corr_bt'] = self.var_corr_bt

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname):

        logger.info('Loading results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)

        signals = validate_array(in_dict['self.signals'])
        sampling_rate = validate_array(in_dict['self.sampling_rate'])
        _ref_channels = validate_array(in_dict['self.ref_channels'])
        _accel_channels = validate_array(in_dict['self.accel_channels'])
        _velo_channels = validate_array(in_dict['self.velo_channels'])
        _disp_channels = validate_array(in_dict['self.disp_channels'])
        channel_headers = validate_array(in_dict['self.channel_headers'])
        start_time = validate_array(in_dict['self.start_time'])
        setup_name = validate_array(in_dict['self.setup_name'])

        preprocessor = cls(signals, sampling_rate,
                           _ref_channels,
                           _accel_channels, _velo_channels, _disp_channels,
                           setup_name=setup_name, channel_headers=channel_headers,
                           start_time=start_time,
                           )

        chan_dofs = [[int(float(chan_dof[0])), str(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3]), str(
            chan_dof[4] if 5 == len(chan_dof) else '')] for chan_dof in in_dict['self.chan_dofs']]
        preprocessor.add_chan_dofs(chan_dofs)

        try:
            preprocessor.scaling_factors = validate_array(in_dict['self.scaling_factors'])
            preprocessor.channel_factors = validate_array(in_dict['self.channel_factors'])
            preprocessor._last_meth = validate_array(in_dict['self._last_meth'])

            preprocessor.corr_matrix_wl = validate_array(in_dict['self.corr_matrix_wl'])
            preprocessor.corr_matrices_wl = validate_array(in_dict.get('self.corr_matrices_wl'))
            preprocessor.psd_matrix_wl = validate_array(in_dict['self.psd_matrix_wl'])
            preprocessor.psd_matrices_wl = validate_array(in_dict.get('self.psd_matrices_wl'))
            preprocessor.var_corr_wl = validate_array(in_dict.get('self.var_corr_wl'))
            preprocessor.var_psd_wl = validate_array(in_dict['self.var_psd_wl'])
            preprocessor.n_lines_wl = validate_array(in_dict['self.n_lines_wl'])
            preprocessor.m_lags_wl = validate_array(in_dict.get('self.m_lags_wl'))
            preprocessor.n_segments_wl = validate_array(in_dict['self.n_segments_wl'])

            preprocessor.corr_matrix_bt = validate_array(in_dict['self.corr_matrix_bt'])
            preprocessor.corr_matrices_bt = validate_array(in_dict.get('self.corr_matrices_bt'))
            preprocessor.psd_matrix_bt = validate_array(in_dict['self.psd_matrix_bt'])
            preprocessor.n_lines_bt = validate_array(in_dict['self.n_lines_bt'])
            preprocessor.m_lags_bt = validate_array(in_dict.get('self.m_lags_bt'))
            preprocessor.n_segments_bt = validate_array(in_dict['self.n_segments_bt'])

            preprocessor.var_corr_bt = validate_array(in_dict['self.var_corr_bt'])

        except KeyError as e:
            # loading data saved with old version, spectral values must be recomputed
            logger.warning(f'Failed to load part of the saved file at Key {e}')

        return preprocessor

    def _remove_channel_from_quantity(self, channel, quant_list, quant_name):
        """If *channel* is in *quant_list*, warn and remove it."""
        if channel in quant_list:
            logger.warning(
                f'Channel {self.channel_headers[channel]} is already defined'
                f' as a {quant_name} channel. Removing')
            quant_list.remove(channel)

    def validate_channels(self, channels, quant_check=False):
        if quant_check:
            accel_channels = self.accel_channels
            velo_channels = self.velo_channels
            disp_channels = self.disp_channels

        for channel in channels:
            if channel < 0:
                raise ValueError('A channel number cannot be negative!')
            if channel > self.num_analised_channels - 1:
                raise ValueError('A channel number cannot be greater'
                                 ' than the number of all channels!')
            if quant_check:
                self._remove_channel_from_quantity(channel, accel_channels, 'acceleration')
                self._remove_channel_from_quantity(channel, velo_channels, 'velocity')
                self._remove_channel_from_quantity(channel, disp_channels, 'displacement')

    def _channel_numbers(self, channels=None, refs=None):
        """
        Method to return channel numbers
        
        Interpretation of the argument values:
        * None: a list of all channel indices
        * list-of-int: a validated list of given channel indices
        * list-of-str: a list of channel indices for each channel name in the given order
        * int: a single-item list of the given channel index
        * str: a single-item list of the channel index for the given name
        * 'auto' (only refs): a single item list corresponding to the respective channel
        
        Parameters
        ----------
            channels: None, list-of-int, list-of-str, int, str
                The selected channels.
            refs: 'auto', list-of-indices, optional
                The reference channel indices to be contained in the reference channel list
                
        Returns
        -------
            channel_numbers: list
                The generated channel indices
            ref_numbers: list-of-lists
                The corresponding reference channels for each channel in
                channel_numbers, such that it can be looped over in an inner loop.
        """
        channel_numbers = self._resolve_channel_list(channels)
        ref_numbers = self._resolve_ref_list(refs, channel_numbers)
        return channel_numbers, ref_numbers

    def _resolve_channel_list(self, channels):
        """Convert *channels* argument to a list of integer channel indices."""
        if channels is None:
            return list(range(self.num_analised_channels))
        if isinstance(channels, int):
            return [channels]
        if isinstance(channels, str):
            return [self._str_channel_to_index(channels)]
        # list / tuple / ndarray
        return self._channel_list_to_indices(channels)

    def _str_channel_to_index(self, channel):
        """Convert a string channel specifier to an integer index."""
        try:
            return int(channel)
        except ValueError:
            return self.channel_headers.index(channel)

    def _channel_list_to_indices(self, channels):
        """Convert a sequence of channel specifiers to a list of integer indices."""
        result = []
        for channel in channels:
            if isinstance(channel, (int, np.int32, np.int64)):
                result.append(int(channel))
            elif isinstance(channel, str):
                result.append(self._str_channel_to_index(channel))
            else:
                raise ValueError(
                    f'Channel {channel} in channels is an invalid channel definition.')
        return result

    def _resolve_ref_list(self, refs, channel_numbers):
        """Convert *refs* argument to a list-of-lists of reference channel indices."""
        n = len(channel_numbers)
        if refs is None:
            return [self.ref_channels] * n
        if refs == 'auto':
            return [[ind] for ind in channel_numbers]
        if isinstance(refs, int):
            return [[refs]] * n
        if isinstance(refs, str):
            return [[self.channel_headers.index(refs)]] * n
        if isinstance(refs, (list, tuple, np.ndarray)):
            return [self._refs_list_to_indices(refs)] * n
        raise ValueError(f'{refs} not a valid reference channel specification.')

    def _refs_list_to_indices(self, refs):
        """Convert a list of reference channel specifiers to integer indices."""
        result = []
        for channel in refs:
            if isinstance(channel, int):
                result.append(channel)
            elif isinstance(channel, str):
                result.append(self.channel_headers.index(channel))
            else:
                raise ValueError(
                    f'Channel {channel} in refs is an invalid channel definition.')
        return result

    @property
    def ref_channels(self):
        return self._ref_channels

    @ref_channels.setter
    def ref_channels(self, ref_channels):
        ref_channels, _ = self._channel_numbers(ref_channels)
        self.validate_channels(ref_channels)
        self._clear_spectral_values()
        self._ref_channels = ref_channels

    @property
    def accel_channels(self):
        return self._accel_channels

    @accel_channels.setter
    def accel_channels(self, accel_channels):
        accel_channels, _ = self._channel_numbers(accel_channels)
        self.validate_channels(accel_channels, True)
        self._accel_channels = accel_channels

    @property
    def velo_channels(self):
        return self._velo_channels

    @velo_channels.setter
    def velo_channels(self, velo_channels):
        velo_channels, _ = self._channel_numbers(velo_channels)
        self.validate_channels(velo_channels, True)
        self._velo_channels = velo_channels

    @property
    def disp_channels(self):
        return self._disp_channels

    @disp_channels.setter
    def disp_channels(self, disp_channels):
        disp_channels, _ = self._channel_numbers(disp_channels)
        self.validate_channels(disp_channels, True)
        self._disp_channels = disp_channels

    @property
    def num_ref_channels(self):
        return len(self.ref_channels)

    @property
    def num_analised_channels(self):
        return self.signals.shape[1]

    @property
    def total_time_steps(self):
        return self.signals.shape[0]

    @property
    def duration(self):
        return self.total_time_steps / self.sampling_rate

    @property
    def dt(self):
        return 1 / self.sampling_rate

    @property
    def t(self):
        N = self.total_time_steps
        fs = self.sampling_rate
        # t[-1] != self.duration to ensure sample_spacing == self.dt
        return np.linspace(0, N / fs, N, False)

    @property
    def n_lines(self):
        if self._last_meth == 'welch':
            return self.n_lines_wl
        elif self._last_meth == 'blackman-tukey':
            return self.n_lines_bt
        else:
            return None

    @property
    def freqs(self):
        '''
        Returns
        ----------
            freqs: np.ndarray (n_lines, )
                Array with the frequency lines corresponding to the spectral values
        '''
        if self.n_lines:
            n_lines = self.n_lines
            fs = self.sampling_rate
            return np.fft.rfftfreq(n_lines, 1 / fs)

    @property
    def freqs_wl(self):
        '''
        Returns
        ----------
            freqs: np.ndarray (n_lines, )
                Array with the frequency lines corresponding to the spectral values
        '''
        if self.m_lags_wl:
            n_lines = self.n_lines_wl
            fs = self.sampling_rate
            return np.fft.rfftfreq(n_lines, 1 / fs)

    @property
    def freqs_bt(self):
        '''
        Returns
        ----------
            freqs: np.ndarray (n_lines, )
                Array with the frequency lines corresponding to the spectral values
        '''
        if self.n_lines_bt:
            n_lines = self.n_lines_bt
            fs = self.sampling_rate
            return np.fft.rfftfreq(n_lines, 1 / fs)

    @property
    def lags(self):
        if self.m_lags:
            m_lags = self.m_lags
            fs = self.sampling_rate
            return np.linspace(0, m_lags / fs, m_lags, False)

    @property
    def lags_wl(self):
        if self.m_lags_wl:
            m_lags = self.m_lags_wl
            fs = self.sampling_rate
            return np.linspace(0, m_lags / fs, m_lags, False)

    @property
    def lags_bt(self):
        if self.m_lags_bt:
            m_lags = self.m_lags_bt
            fs = self.sampling_rate
            return np.linspace(0, m_lags / fs, m_lags, False)

    @property
    def m_lags(self):
        if self._last_meth == 'welch':
            return self.m_lags_wl
        elif self._last_meth == 'blackman-tukey':
            return self.m_lags_bt
        else:
            return None

    @property
    def n_segments(self):
        if self._last_meth == 'welch':
            return self.n_segments_wl
        elif self._last_meth == 'blackman-tukey':
            return self.n_segments_bt
        else:
            return None

    # @property
    # def m_lags_wl(self):
    #     if self.n_lines_wl:
    #         return self.n_lines_wl // 2 + 1
    #
    # @property
    # def m_lags_bt(self):
    #     if self.n_lines_bt:
    #         return self.n_lines_bt // 2 + 1

    @property
    def corr_matrix(self):
        if self._last_meth == 'welch':
            return self.corr_matrix_wl
        elif self._last_meth == 'blackman-tukey':
            return self.corr_matrix_bt
        else:
            return None

    @property
    def corr_matrices(self):
        if self._last_meth == 'welch':
            return self.corr_matrices_wl
        elif self._last_meth == 'blackman-tukey':
            return self.corr_matrices_bt
        else:
            return None

    @property
    def psd_matrix(self):
        if self._last_meth == 'welch':
            return self.psd_matrix_wl
        elif self._last_meth == 'blackman-tukey':
            return self.psd_matrix_bt
        else:
            return None

    @property
    def signal_power(self):
        if not np.all(np.isclose(np.mean(self.signals, axis=0), 0)):
            logger.warning("Signal has constant offsets. Power values may be errorneous")
        return np.mean(np.square(self.signals), axis=0)

    @property
    def signal_rms(self):
        return np.sqrt(self.signal_power)

    def add_noise(self, amplitude=0, snr=0):
        """Add Gaussian white noise to the signals (useful for simulation studies).

        Parameters
        ----------
        amplitude : float, optional
            Absolute noise amplitude (standard deviation).  Ignored when
            *snr* is non-zero.
        snr : float, optional
            Noise amplitude as a fraction of the per-channel RMS.
            At least one of *amplitude* or *snr* must be non-zero.
        """
        logger.info(
            'Adding Noise with Amplitude {} and {} percent RMS'.format(
                amplitude,
                snr *
                100))
        if amplitude == 0 and snr == 0:
            raise ValueError("At least one of 'amplitude' or 'snr' must be non-zero.")

        if snr != 0 and amplitude == 0:
            rms = self.signal_rms
            amplitude = rms * snr
        else:
            amplitude = [
                amplitude for channel in range(
                    self.num_analised_channels)]

        for channel in range(self.num_analised_channels):
            self.signals[:, channel] += np.random.normal(0, amplitude[channel], self.total_time_steps)
        self._clear_spectral_values()

    def correct_offset(self):
        '''
        corrects a constant offset from measured signals
        
        ..TODO::
            * remove linear, ... ofsets as well
        '''
        logger.info('Correcting offset of measured signals')
        self.signals -= self.signals.mean(axis=0)
        self._clear_spectral_values()

        return

    def precondition_signals(self, method='iqr'):
        """Remove the DC offset and scale each channel by its spread.

        First calls :meth:`correct_offset`, then divides each channel by either
        the inter-quartile range (IQR, 5th-95th percentile) or the full signal
        range.  Scaling factors are stored in ``self.channel_factors``.

        Parameters
        ----------
        method : {'iqr', 'range'}, optional
            Spreading measure used for normalisation.  Default is ``'iqr'``.
        """

        if method not in ['iqr', 'range']:
            raise ValueError(f"method must be one of 'iqr', 'range', got {method!r}")

        self.correct_offset()

        for i in range(self.signals.shape[1]):
            tmp = self.signals[:, i]
            if method == 'iqr':
                factor = np.subtract(*np.percentile(tmp, [95, 5]))
            elif method == 'range':
                factor = np.max(tmp) - np.min(tmp)
            self.signals[:, i] /= factor
            self.channel_factors[i] = factor

        self._clear_spectral_values()

    @staticmethod
    def _default_filter_order(ftype, ftype_list):
        """Return default filter order (4 for IIR, 21 for FIR)."""
        return 4 if ftype_list.index(ftype) < 5 else 21

    @staticmethod
    def _resolve_btype_freqs(lowpass, highpass):
        """Build frequency list and filter type string from lowpass/highpass."""
        freqs = []
        if lowpass is not None:
            freqs.append(float(lowpass))
        if highpass is not None:
            freqs.append(float(highpass))
        if len(freqs) == 2:
            freqs.sort()
            return freqs, 'bandpass'
        if highpass is not None:
            return freqs, 'highpass'
        return freqs, 'lowpass'

    def _setup_filter_params(self, lowpass, highpass, order, ftype, RpRs):
        """Validate filter parameters and derive (ftype_list, order, nyq, freqs, btype, RpRs)."""
        if RpRs is None:
            RpRs = [3, 3]
        if highpass is None and lowpass is None:
            raise ValueError('Neither a lowpass or a highpass corner frequency was provided.')
        ftype_list = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'moving_average', 'brickwall']
        if ftype not in ftype_list:
            raise ValueError(f'Filter type {ftype} is not any of the available types: {ftype_list}')
        if order is None:
            order = self._default_filter_order(ftype, ftype_list)
        if order < 1:
            raise ValueError('Order must be greater equal 1')
        nyq = self.sampling_rate / 2
        freqs, btype = self._resolve_btype_freqs(lowpass, highpass)
        freqs = list(np.array(freqs) / nyq)
        return ftype_list, order, nyq, freqs, btype, RpRs

    def _run_filter(self, ftype, ftype_list, freqs, btype, order, RpRs):
        """Apply the chosen filter to self.signals; return (signals_filtered, sos, fir_irf)."""
        measurement = self.signals
        if ftype in ftype_list[0:5]:  # IIR filter
            signals_filtered, sos = self._apply_iir_filter(
                measurement, freqs, btype, ftype, order, RpRs)
            fir_irf = None
        else:  # FIR filter
            signals_filtered, fir_irf = self._apply_fir_filter(
                measurement, freqs, btype, ftype, order)
            sos = None
        return signals_filtered, sos, fir_irf

    def filter_signals(self, lowpass=None, highpass=None,
                       overwrite=True,
                       order=None, ftype='butter', RpRs=None,
                       plot_ax=None):
        """Apply a zero-phase IIR or FIR filter to the measurement signals.

        Parameters
        ----------
        lowpass : float, optional
            Lowpass corner frequency in Hz.
        highpass : float, optional
            Highpass corner frequency in Hz.  At least one of *lowpass* or
            *highpass* must be given; providing both creates a bandpass filter.
        overwrite : bool, optional
            If ``True`` (default), store the filtered signals in
            ``self.signals``.  If ``False``, return the filtered array without
            modifying ``self``.
        order : int, optional
            Filter order.  Defaults to 4 for IIR types and 21 for FIR types.
        ftype : str, optional
            Filter type: ``'butter'``, ``'cheby1'``, ``'cheby2'``,
            ``'ellip'``, ``'bessel'``, ``'moving_average'``, or
            ``'brickwall'``.
        RpRs : list of float, optional
            ``[rp, rs]`` — maximum passband ripple and minimum stopband
            attenuation (dB) for Chebyshev/elliptic filters.
        plot_ax : matplotlib.axes.Axes or list of Axes, optional
            When provided, plot the filter frequency response (and optionally
            impulse response) into the given axes.

        Returns
        -------
        np.ndarray
            Filtered signal array (returned regardless of *overwrite*).
        """
        logger.info('Filtering signals in the band: {} .. {} with a {} order {} filter.'.format(highpass, lowpass, order, ftype))

        ftype_list, order, nyq, freqs, btype, RpRs = self._setup_filter_params(
            lowpass, highpass, order, ftype, RpRs)

        signals_filtered, sos, fir_irf = self._run_filter(ftype, ftype_list, freqs, btype, order, RpRs)

        if np.isnan(signals_filtered).any():
            logger.warning('Your filtered signals contain NaNs. Check your filter settings! Continuing...')

        if plot_ax is not None:
            self._plot_filter_response(plot_ax, ftype, ftype_list, sos, fir_irf, order, nyq)

        if overwrite:
            self.signals = signals_filtered
            if self.F is not None:
                self.F = self.F_filt
        self.signals_filtered = signals_filtered
        self._clear_spectral_values()

        return signals_filtered

    def _apply_iir_filter(self, measurement, freqs, btype, ftype, order, RpRs):
        """Design and apply a zero-phase IIR filter; return (filtered_signals, sos)."""
        order = int(order)
        Wn = freqs[0] if len(freqs) == 1 else freqs
        sos = scipy.signal.iirfilter(
            order, Wn, rp=RpRs[0], rs=RpRs[1],
            btype=btype, ftype=ftype, output='sos')
        signals_filtered = scipy.signal.sosfiltfilt(sos, measurement, axis=0)
        if self.F is not None:
            self.F_filt = scipy.signal.sosfiltfilt(sos, self.F, axis=0)
        return signals_filtered, sos

    def _apply_fir_filter(self, measurement, freqs, btype, ftype, order):
        """Design and apply a causal FIR filter; return (filtered_signals, fir_irf)."""
        if ftype == 'brickwall':
            fir_irf = scipy.signal.firwin(numtaps=order, cutoff=freqs, pass_zero=btype, fs=np.pi)
        else:  # moving_average
            if freqs:
                logger.warning('For the moving average filter, no cutoff frequencies can be defined.')
            fir_irf = np.ones((order)) / order
        signals_filtered = scipy.signal.lfilter(fir_irf, [1.0], measurement, axis=0)
        if self.F is not None:
            self.F_filt = scipy.signal.lfilter(fir_irf, [1.0], self.F, axis=0)
        return signals_filtered, fir_irf

    def _plot_filter_response(self, plot_ax, ftype, ftype_list, sos, fir_irf, order, nyq):
        """Plot the filter frequency (and optionally impulse) response."""
        N = 2048
        dt = 1 / self.sampling_rate

        if isinstance(plot_ax, (list, np.ndarray)):
            freq_ax = plot_ax[1]
            tim_ax = plot_ax[0]
        else:
            freq_ax = plot_ax
            tim_ax = None

        if ftype in ftype_list[0:5]:  # IIR Filter
            self._plot_iir_response(freq_ax, tim_ax, sos, nyq, N, dt)
        else:  # FIR Filter
            self._plot_fir_response(freq_ax, tim_ax, fir_irf, order, N, dt)

    def _plot_iir_response(self, freq_ax, tim_ax, sos, nyq, N, dt):
        """Plot IIR filter frequency and optional impulse response."""
        w, h = scipy.signal.sosfreqz(sos, worN=np.fft.rfftfreq(N) * 2 * np.pi)
        # convert to decibels (square: double filtering; factor 20: RMS quantity)
        frf = 20 * np.log10(abs(h) ** 2)
        freq_ax.plot((nyq / np.pi) * w, frf, color='lightgrey', ls='dashed')
        if tim_ax is not None:
            irf = np.fft.irfft(h, n=10 * N)
            logger.debug(f'IRF Integral {np.sum(irf)*dt}')
            dur = N * dt
            t = np.linspace(0, dur - dt, 10 * N)
            tim_ax.plot(t, irf, color='lightgrey')

    def _plot_fir_response(self, freq_ax, tim_ax, fir_irf, order, N, dt):
        """Plot FIR filter frequency and optional impulse response."""
        dur = order * dt
        # zero-pad the FRF to achieve spectral-interpolated IRF
        frf = np.fft.fft(fir_irf)
        if order % 2:
            # odd numtaps: maximum frequency present as conjugate
            neg = frf[order // 2 + 1:order]
            pos = frf[:order // 2 + 1]
        else:
            # even numtaps: only minimum frequency present
            pos = frf[:order // 2]
            neg = frf[order // 2:order]
            pos = np.hstack([pos, np.conj(neg[0:1])])
        frf_pad = np.hstack([pos, np.zeros((N - order // 2 * 2 - 1,), dtype=complex), neg])
        irf_fine = np.fft.ifft(frf_pad)
        if np.max(irf_fine.imag) > np.finfo(np.float64).eps:
            raise RuntimeError(
                "Interpolated IRF has a non-negligible imaginary part "
                f"(max={np.max(irf_fine.imag)!r}); the IFFT result is not real-valued as expected."
            )
        irf_fine = irf_fine.real
        dt_new = dur / N
        irf_fine /= dt_new / dt
        logger.debug(f'IRF Integral {np.sum(fir_irf) * dt}, {np.sum(irf_fine) * dt_new}')
        # zero-pad the IRF to achieve high-resolution FRF
        irf_pad = np.zeros((N,))
        irf_pad[:order] = fir_irf
        frf_fine = 20 * np.log10(abs(np.fft.fft(irf_pad)))
        freq_ax.plot(np.fft.fftshift(np.fft.fftfreq(N, dt)),
                     np.fft.fftshift(frf_fine), color='lightgrey', ls='dashed')
        if tim_ax is not None:
            t = np.linspace(-dur / 2, dur / 2 - dt_new, N)
            tim_ax.plot(t, irf_fine, color='lightgrey')

    @staticmethod
    def _resolve_decimate_filter_params(order, filter_type, decimate_factor):
        """Derive the anti-aliasing filter order and RpRs for decimation."""
        if order is None:
            if filter_type in ['brickwall', 'moving_average']:
                order = 21 * decimate_factor - 1  # odd to avoid errors when highpass filtering
            else:
                order = 8
        else:
            order = abs(order)
        if not isinstance(order, int):
            raise TypeError(f"order must be int, got {type(order).__name__!r}")
        if not (order > 1):
            raise ValueError(f"order must be > 1, got {order}")
        if filter_type in ('cheby1', 'cheby2', 'ellip'):
            RpRs = [0.05, 0.05]  # standard for signal.decimate
        else:
            RpRs = [None, None]
        return order, RpRs

    def _apply_downsampling(self, sig_filtered, decimate_factor):
        """Downsample *sig_filtered* by *decimate_factor* and update self.signals/sampling_rate."""
        self.sampling_rate /= decimate_factor
        N_dec = int(np.floor(self.total_time_steps / decimate_factor))
        # ceil would also work, but breaks indexing for aliasing noise estimation
        # with floor though, care must be taken to shorten the time domain signal to N_dec full blocks before slicing
        sig_decimated = np.copy(sig_filtered[0:N_dec * decimate_factor:decimate_factor, :])
        # correct for power loss due to decimation
        # https://en.wikipedia.org/wiki/Downsampling_(signal_processing)#Anti-aliasing_filter
        sig_decimated *= decimate_factor
        if self.F is not None:
            self.F = self.F_filt[slice(None, None, decimate_factor)]
        self.signals = sig_decimated
        self._clear_spectral_values()

    def decimate_signals(self, decimate_factor, nyq_rat=2.5,
                         highpass=None, order=None, filter_type='cheby1'):
        """Decimate the signals by an integer factor.

        An anti-aliasing lowpass filter is applied before downsampling.  To
        achieve large total reduction factors, call this method multiple times
        with moderate per-step factors (e.g. two passes of x3 instead of one
        pass of x9).

        Parameters
        ----------
        decimate_factor : int
            Integer downsampling factor (must be >= 1).
        nyq_rat : float, optional
            The lowpass corner frequency is set to
            ``sampling_rate / (decimate_factor * nyq_rat)``.
            Must be >= 2.  Default is 2.5.
        highpass : float or None, optional
            Additional highpass corner frequency in Hz applied simultaneously.
        order : int, optional
            Anti-aliasing filter order.  Defaults to 8 (IIR) or
            ``21 * decimate_factor - 1`` (FIR).
        filter_type : str, optional
            Filter type passed to :meth:`filter_signals`.  Default is
            ``'cheby1'``.
        """

        if highpass:
            logger.info(f'Decimating signals by factor {decimate_factor}'
                        f' and additional highpass filtering at {highpass}'
                        f' to a sampling rate of {self.sampling_rate/decimate_factor} Hz')
        else:
            logger.info(f'Decimating signals by factor {decimate_factor}'
                        f' to a sampling rate of {self.sampling_rate/decimate_factor} Hz')

        # input validation
        decimate_factor = abs(decimate_factor)

        if not isinstance(decimate_factor, int):
            raise TypeError(f"decimate_factor must be int, got {type(decimate_factor).__name__!r}")
        if not (decimate_factor >= 1):
            raise ValueError(f"decimate_factor must be >= 1, got {decimate_factor}")
        if not (nyq_rat >= 2.0):
            raise ValueError(f"nyq_rat must be >= 2.0, got {nyq_rat}")

        order, RpRs = self._resolve_decimate_filter_params(order, filter_type, decimate_factor)

        nyq = self.sampling_rate / decimate_factor

        sig_filtered = self.filter_signals(
            lowpass=nyq / nyq_rat,
            highpass=highpass,
            overwrite=False,
            order=order,
            ftype=filter_type,
            RpRs=RpRs,)

        self._apply_downsampling(sig_filtered, decimate_factor)

    def _clear_spectral_values(self):
        """
        Convenience method to clear all previously computed spectral values.
        To be called when any modifications, such as filtering, decimation,
        etc. are applied to the signals.
        """

        self.scaling_factors = None
        self._last_meth = None

        self.psd_matrix_bt = None
        self.psd_matrix_wl = None

        self.n_lines_wl = None
        self.n_lines_bt = None

        self.n_segments_bt = None
        self.n_segments_wl = None

        self.corr_matrix_bt = None
        self.corr_matrix_wl = None

        self.var_corr_bt = None
        self.var_psd_wl = None

    def psd_welch(self, n_lines=None, n_segments=None, refs_only=True, window='hamming', **kwargs):
        '''
        Estimate the (cross- and auto-) power spectral densities (PSD),
        according to Welch's method. No overlapping is allowed (deliberately
        to ensure statistical independence of blocks for variance estimation).
        Segments are n_lines // 2 long and zero padded to n_lines to allow
        estimation of the full correlation sequence, which is twice as long
        as the input signal. Normalization is applied w.r.t. conservation of
        energy, i.e. magnitudes will change with n_lines but power stays
        constant.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            n_segments: integer, optional
                Number of segments to perform averaging over
                resulting segment length must be smaller or equal n_lines
            refs_only: bool, optional
                Compute cross-PDSs only with reference channels
            window: str or tuple or array_like, optional
                Desired window to use. See scipy.signal.get_window() for more information
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to scipy.signals.csd

        Returns
        -------
            psd_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lines // 2 + 1)
                containing the power density values of the respective
                channels and frequencies
                
        '''

        N = self.total_time_steps
        n_lines, n_segments, N_segment, _n_segments = \
            self._resolve_psd_welch_params(n_lines, n_segments, N)

        self._last_meth = 'welch'

        cached = self._check_psd_welch_cache(n_lines, n_segments, refs_only, kwargs, _n_segments, window)
        if cached is not None:
            return cached

        logger.info(f"Estimating PSD by Welch's method with {n_lines}"
                    f' frequency lines, {_n_segments} non-overlapping'
                    f' segments and a {window} window...')

        psd_matrix = self._compute_psd_welch(
            n_lines, n_segments, N_segment, _n_segments, refs_only, window, **kwargs)

        if self.scaling_factors is None:
            self.scaling_factors = psd_matrix.max(axis=2)

        self.psd_matrix_wl = psd_matrix
        self.n_lines_wl = n_lines
        self.n_segments_wl = n_segments

        self.m_lags_wl = None
        self.corr_matrix_wl = None
        self.corr_matrices_wl = None
        self.var_corr_wl = None
        self.s_vals_psd = None

        return psd_matrix

    @staticmethod
    def _validate_n_lines(n_lines, N):
        """Validate n_lines and ensure it is even; return corrected n_lines."""
        if not isinstance(n_lines, int):
            raise ValueError(
                f"{n_lines} is not a valid number of n_lines for a spectral densities")
        if n_lines % 2:
            n_lines += 1
            logger.warning(
                f"Only even number of frequency lines are supported setting n_lines={n_lines}")
        if n_lines > 2 * N:
            logger.warning(
                f'Number of frequency lines {n_lines} should not'
                f'be larger than twice the number of timesteps {N}')
        return n_lines

    @staticmethod
    def _validate_n_segments(n_segments):
        """Validate that n_segments is an integer."""
        if not isinstance(n_segments, int):
            raise ValueError(f"{n_segments} is not a valid number of segments")

    def _welch_load_cached_params(self, n_lines, n_segments):
        """Load n_lines/n_segments from Welch cache when both are None."""
        if n_lines is None and n_segments is None:
            n_lines = self.n_lines_wl
            n_segments = self.n_segments_wl
            if n_lines is None and n_segments is None:
                raise RuntimeError('Either n_lines or n_segments must be provided on first run.')
        return n_lines, n_segments

    @staticmethod
    def _welch_resolve_cases(n_lines, n_segments, N):
        """Resolve (n_lines, _n_segments, N_segment) for Welch cases 2–4."""
        if n_segments is None:
            N_segment = n_lines
            return n_lines, N // N_segment, N_segment
        if n_lines is None:
            N_segment = N // n_segments
            return N_segment, n_segments, N_segment
        N_segment = min(N // n_segments, n_lines)
        return n_lines, n_segments, N_segment

    def _resolve_psd_welch_params(self, n_lines, n_segments, N):
        """Validate and resolve n_lines/n_segments for psd_welch.

        Returns (n_lines, n_segments, N_segment, _n_segments).
        """
        if n_lines is not None:
            n_lines = self._validate_n_lines(n_lines, N)
        if n_segments is not None:
            self._validate_n_segments(n_segments)
        n_lines, n_segments = self._welch_load_cached_params(n_lines, n_segments)
        n_lines, _n_segments, N_segment = self._welch_resolve_cases(n_lines, n_segments, N)

        if n_lines % 2:
            n_lines += 1
        if N_segment > n_lines:
            raise ValueError(
                f"The segment length {N_segment} must not be larger than "
                f"the number of frequency lines {n_lines}")
        if N_segment < n_lines / 2:
            logger.warning(
                f"The segment length {N_segment} is much smaller than "
                f"the number of frequency lines {n_lines} (zero-padded)")
        return n_lines, n_segments, N_segment, _n_segments

    def _check_psd_welch_cache(self, n_lines, n_segments, refs_only, kwargs, _n_segments, window):
        """Return cached psd_matrix_wl if still valid, else return None."""
        if kwargs:
            logger.debug("Not returning because: kwargs provided")
            return None
        if self.psd_matrix_wl is None:
            logger.debug("Not returning because: self.psd_matrix_wl not available")
            return None
        if self.n_lines_wl != n_lines:
            logger.debug("Not returning because: n_lines differs from previous")
            return None
        if n_segments is not None and self.psd_matrices_wl.shape[0] != n_segments:
            logger.debug("Not returning because: n_segments differs from previous")
            return None
        if (self.psd_matrix_wl.shape[1] == self.num_ref_channels) != refs_only:
            logger.debug("Not returning because: non-/reference-based not matching previous")
            return None
        logger.debug(f"Returning PSD by Welch's method with {n_lines}"
                     f' frequency lines, {_n_segments} non-overlapping'
                     f' segments and a {window} window...')
        return self.psd_matrix_wl

    @staticmethod
    def _compute_channel_pair_psd(sig_block, channel_1, ref_channel, fs, win,
                                   seg_params, **kwargs):
        """Compute normalised cross-PSD for one channel pair in one segment block."""
        N_segment, n_lines, _n_segments = seg_params
        _, Pxy_den = scipy.signal.csd(
            sig_block[:, channel_1],
            sig_block[:, ref_channel],
            fs,
            window=win,
            nperseg=N_segment,
            nfft=n_lines,
            noverlap=0,
            return_onesided=True,
            scaling='density',
            **kwargs)
        if channel_1 == ref_channel:
            if not np.isclose(Pxy_den.imag, 0).all():
                raise RuntimeError(
                    "Auto-PSD (channel_1 == ref_channel) has a non-negligible "
                    "imaginary part; expected a real-valued result from the Welch "
                    "cross-spectral density computation."
                )
            Pxy_den.imag = 0
        # compensate averaging over segments
        Pxy_den *= _n_segments
        Pxy_den *= fs       # reverse 1/Hz of scaling="density"
        Pxy_den /= 2        # compensate onesided
        Pxy_den /= 2        # compensate zero-padding
        Pxy_den *= n_lines  # compensate energy loss through short segments
        return Pxy_den

    def _compute_psd_welch(self, n_lines, n_segments, N_segment, _n_segments,
                            refs_only, window, **kwargs):
        """Compute PSD matrices for all segments and return the mean."""
        fs = self.sampling_rate
        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
            ref_channels = self.ref_channels
        else:
            num_ref_channels = num_analised_channels
            ref_channels = list(range(num_ref_channels))

        signals = self.signals
        psd_matrix_shape = (num_analised_channels, num_ref_channels, n_lines // 2 + 1)
        psd_matrices = []
        win = scipy.signal.get_window(window, N_segment, fftbins=True)
        pbar = simplePbar(_n_segments * num_analised_channels * num_ref_channels)

        for i_seg in range(_n_segments):
            this_psd_matrix = np.empty(psd_matrix_shape, dtype=complex)
            this_signals_block = signals[i_seg * N_segment:(i_seg + 1) * N_segment, :]
            for channel_1 in range(num_analised_channels):
                for channel_2, ref_channel in enumerate(ref_channels):
                    next(pbar)
                    Pxy_den = self._compute_channel_pair_psd(
                        this_signals_block, channel_1, ref_channel, fs, win,
                        (N_segment, n_lines, _n_segments), **kwargs)
                    this_psd_matrix[channel_1, channel_2, :] = Pxy_den
            psd_matrices.append(this_psd_matrix)

        psd_matrix = np.mean(psd_matrices, axis=0)
        self.psd_matrices_wl = np.stack(psd_matrices, axis=0)
        self.var_psd_wl = np.var(psd_matrices, axis=0)
        return psd_matrix

    def corr_welch(self, m_lags=None, n_segments=None, refs_only=True, **kwargs):
        '''
        Estimate the (cross- and auto-) correlation functions (C/ACF),
        by the inverse Fourier Transform of Power Spectral Densities,
        estimated according to Welch's method. Bias due to windowing of
        the underlying PSD persists.  Normalization is done according to
        the unbiased estimator, i.e. 0-lag correlation value must be
        multiplied by n_lines to get the signals cross-power.
        
        Note that:
            m_lags \= n_lines // 2 + 1
        
            n_lines \= (m_lags - 1) * 2
            
            N_segment = N // n_segments

        Parameters
        ----------
            m_lags: integer, optional
                Total number of lags (positive). Note: this includes the
                0-lag, therefore excludes the m_lags-lag.
            n_segments: integer, optional
                Number of segments to perform averaging over
                resulting segment length must be smaller or equal n_lines
            refs_only: bool, optional
                Compute cross-ACFss only with reference channels
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to self.psd_welch and further

        Returns
        -------
            corr_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, m_lags)
                containing the correlation values of the respective
                channels and lags
        
        See also
        --------
            psd_welch:
                PSD estimation algorithm used by this method.
                
                        
        .. TODO ::
            * deconvolve window (if possible)
        '''
        self._last_meth = 'welch'

        if m_lags is not None:
            if not isinstance(m_lags, int):
                raise ValueError(
                    f"{m_lags} is not a valid number of lags for a correlation sequence")
        if n_segments is not None:
            if not isinstance(n_segments, int):
                raise ValueError(f"{n_segments} is not a valid number of segments")

        N = self.total_time_steps
        m_lags, n_segments, _n_segments, _n_lines = \
            self._resolve_corr_welch_params(m_lags, n_segments, N)

        cached = self._check_corr_welch_cache(m_lags, n_segments, refs_only, kwargs)
        if cached is not None:
            return cached

        # onesided RFFT suffices for real inputs; correlation is real so IRFFT suffices
        self.psd_welch(n_lines=_n_lines, n_segments=_n_segments, refs_only=refs_only, **kwargs)

        logger.info("Estimating Correlation Function by Welch's method with"
            f" {m_lags} time lags and {_n_segments} non-overlapping"
            f" segments.")

        corr_matrix = self._compute_corr_welch(m_lags, n_segments, refs_only)

        self.corr_matrix_wl = corr_matrix
        self.m_lags_wl = m_lags
        return corr_matrix

    def _corr_welch_from_cache(self, m_lags, n_segments):
        """Resolve m_lags/n_segments from cached values; raise if not available."""
        if self.m_lags_wl is not None:
            m_lags = self.m_lags_wl
        elif self.n_lines_wl is not None:
            m_lags = self.n_lines_wl // 2 + 1
        n_segments = self.n_segments_wl
        if m_lags is None and n_segments is None:
            raise RuntimeError('Either m_lags or n_segments must be provided on first run.')
        return m_lags, n_segments

    @staticmethod
    def _corr_welch_both_given(m_lags, n_segments, N):
        """Resolve _n_segments/_n_lines when both m_lags and n_segments are given."""
        _n_segments = n_segments
        _n_lines = (m_lags - 1) * 2
        N_segment = min(N // n_segments, _n_lines)
        if N_segment > (m_lags - 1) * 2:
            raise ValueError(
                f"The segment length {N_segment} must not be larger than "
                f"the number of frequency lines {(m_lags - 1) * 2}")
        return _n_segments, _n_lines

    def _resolve_corr_welch_params(self, m_lags, n_segments, N):
        """Resolve corr_welch parameters; return (m_lags, n_segments, _n_segments, _n_lines)."""
        # case 1: no arguments — use cached
        if m_lags is None and n_segments is None:
            m_lags, n_segments = self._corr_welch_from_cache(m_lags, n_segments)
        # case 2: no variance requested
        if n_segments is None and m_lags is not None:
            _n_segments = N // ((m_lags - 1) * 2)
            _n_lines = None
        # case 3: variance requested, lags not specified
        elif n_segments is not None and m_lags is None:
            _n_segments = n_segments
            m_lags = N // n_segments // 2 + 1
            _n_lines = None
        # case 4: both specified
        else:
            _n_segments, _n_lines = self._corr_welch_both_given(m_lags, n_segments, N)
        return m_lags, n_segments, _n_segments, _n_lines

    def _check_corr_welch_cache(self, m_lags, n_segments, refs_only, kwargs):
        """Return cached corr_matrix_wl slice if still valid, else return None."""
        if kwargs:
            logger.debug("Not returning because: kwargs provided")
            return None
        if self.corr_matrix_wl is None:
            logger.debug("Not returning because: self.corr_matrix_wl not available")
            return None
        if self.m_lags_wl < m_lags:
            logger.debug("Not returning because: m_lags differs from previous")
            return None
        if n_segments is not None and self.n_segments_wl != n_segments:
            logger.debug("Not returning because: n_segments differs from previous")
            return None
        if (self.corr_matrix_wl.shape[1] == self.num_ref_channels) != refs_only:
            logger.debug("Not returning because: non-/reference-based not matching previous")
            return None
        logger.debug("Returning Correlation Function by Welch's method with"
            f" {m_lags} time lags and {self.n_segments_wl} non-overlapping segments.")
        return self.corr_matrix_wl[..., :m_lags]

    def _compute_corr_welch(self, m_lags, n_segments, refs_only):
        """Compute correlation matrices from precomputed psd_matrices_wl; store and return mean."""
        if n_segments is None or n_segments == 1:
            psd_matrices = self.psd_matrix_wl[np.newaxis, ...]
            n_segments = 1
        else:
            psd_matrices = self.psd_matrices_wl

        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels

        corr_matrix_shape = (num_analised_channels, num_ref_channels, m_lags)
        corr_matrices = []
        pbar = simplePbar(n_segments * num_analised_channels * num_ref_channels)

        for i_segment in range(n_segments):
            this_corr_matrix = np.empty(corr_matrix_shape)
            this_psd_matrix = psd_matrices[i_segment, ...]
            for channel_1 in range(num_analised_channels):
                for channel_2 in range(num_ref_channels):
                    next(pbar)
                    this_psd = this_psd_matrix[channel_1, channel_2, :]
                    this_corr = np.fft.irfft(this_psd)
                    if not np.all(np.isclose(this_corr.imag, 0)):
                        raise RuntimeError(
                            "Correlation function computed via IFFT has a non-negligible "
                            "imaginary part; expected a real-valued result from the inverse "
                            "FFT of the PSD matrix."
                        )
                    this_corr = this_corr[:m_lags].real
                    this_corr /= (m_lags - 1) * 2
                    this_corr_matrix[channel_1, channel_2, :] = this_corr
            corr_matrices.append(this_corr_matrix)

        corr_matrix = np.mean(corr_matrices, axis=0)
        self.corr_matrices_wl = np.stack(corr_matrices, axis=0)
        self.var_corr_wl = np.var(corr_matrices, axis=0)
        return corr_matrix

    def corr_blackman_tukey(self, m_lags, n_segments=None, refs_only=True, **kwargs):
        '''
        Estimate the (cross- and auto-) correlation functions (C/ACF),
        by direct computation of the standard un-biased estimator:
        
        .. math::
        
           \\hat{R}_{fg}[m] = \\frac{1}{N - m}\\sum_{n=0}^{N - m - 1} f[n] g[n + m]
        
        Computes correlation functions of all channels with selected reference
        channels up to, but excluding, a time lag of m_lags. Normalization
        is done according to the unbiased estimator, i.e. 0-lag correlation
        value must be multiplied by n_lines to get the signals cross-power.
        
        Variance estimation for each time lag is performed by dividing
        the signals into n_segments non-overlapping blocks for individual
        estimation of correlation functions. With increasing numbers of
        non-overlapping blocks the confidence intervals of the correlation
        functions increase ("worsen"), especially at higher lags and
        short block lengths, due to a larger number of time steps being
        discarded.
        
        Note that:
            m_lags \= n_lines // 2 + 1
        
            n_lines \= (m_lags - 1) * 2

        Parameters
        ----------
            m_lags: integer, optional
                Number of lags (positive). Note: this includes the
                0-lag, therefore excludes the "m_lags"-lag.
            n_segments: integer, optional
                Number of blocks to perform averaging over. If blocks
                are shorter than m_lags it raises a ValueError.
            refs_only: bool, optional
                Compute cross-ACFss only with reference channels
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are currently not used
        
        Returns
        -------
            corr_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, m_lags)
                containing the correlation values of the respective
                channels and lags
        
        See also
        --------
            corr_welch:
                Correlation function estimation by Welch's method, possibly
                faster, but distorted for short segments and biased through
                windowing.
        '''

        self._last_meth = 'blackman-tukey'

        if m_lags is not None:
            if not isinstance(m_lags, int):
                raise ValueError(f"{m_lags} is not a valid number of lags for a correlation sequence")
        if n_segments is not None:
            if not isinstance(n_segments, int):
                raise ValueError(f"{n_segments} is not a valid number of blocks")

        N = self.total_time_steps
        m_lags, n_segments, N_block = self._corr_bt_resolve_params(m_lags, n_segments, N)

        cached = self._check_corr_bt_cache(m_lags, n_segments, refs_only, kwargs)
        if cached is not None:
            return cached

        return self._corr_blackman_tukey_core(m_lags, n_segments, N_block, refs_only)

    def _corr_bt_load_cached_params(self, m_lags, n_segments):
        """Load m_lags/n_segments from BT cache when both are None."""
        if m_lags is None and n_segments is None:
            m_lags = self.m_lags_bt
            n_segments = self.n_segments_bt
            if m_lags is None and n_segments is None:
                raise RuntimeError('Either m_lags or n_segments must be provided on first run.')
        return m_lags, n_segments

    def _corr_bt_resolve_params(self, m_lags, n_segments, N):
        """Resolve m_lags, n_segments, N_block for corr_blackman_tukey."""
        m_lags, n_segments = self._corr_bt_load_cached_params(m_lags, n_segments)
        if n_segments is None:
            m_lags, n_segments, N_block = self._corr_bt_case2(m_lags, N)
        elif m_lags is None:
            m_lags = N // n_segments
            N_block = m_lags
        else:
            N_block = N // n_segments
            if N_block < m_lags:
                raise ValueError(
                    f"The segment length {N_block} must not be shorther than the number of lags {m_lags}")
        return m_lags, n_segments, N_block

    def _corr_bt_case2(self, m_lags, N):
        """Resolve case 2: m_lags given, n_segments not given."""
        if self.n_segments_bt is None:
            N_block = N
            n_segments = 1
        else:
            n_segments = self.n_segments_bt
            N_block = N // n_segments
            if N_block < m_lags:
                n_segments = 1
                N_block = N
        return m_lags, n_segments, N_block

    def _check_corr_bt_cache(self, m_lags, n_segments, refs_only, kwargs):
        """Return cached corr_matrix_bt slice if still valid, else None."""
        if kwargs:
            logger.debug("Not returning because: kwargs provided")
            return None
        if self.corr_matrix_bt is None:
            logger.debug("Not returning because: self.corr_matrix_bt not available")
            return None
        if self.m_lags_bt < m_lags:
            logger.debug("Not returning because: m_lags differs from previous")
            return None
        if n_segments is not None and self.n_segments_bt != n_segments:
            logger.debug("Not returning because: n_segments differs from previous")
            return None
        if (self.corr_matrix_bt.shape[1] == self.num_ref_channels) != refs_only:
            logger.debug("Not returning because: non-/reference-based not matching previous")
            return None
        logger.debug("Using previously computed Correlation Functions (BT)...")
        return self.corr_matrix_bt[..., :m_lags]

    def _corr_blackman_tukey_core(self, m_lags, n_segments, N_block, refs_only):
        """Compute Blackman-Tukey correlation functions; store and return result."""
        logger.info(f'Estimating Correlation Functions (BT) with m_lags='
                    f'{m_lags} and n_segments={n_segments}...')

        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
            ref_channels = self.ref_channels
        else:
            num_ref_channels = num_analised_channels
            ref_channels = list(range(num_ref_channels))

        signals = self.signals
        corr_matrix_shape = (num_analised_channels, num_ref_channels, m_lags)
        corr_matrices = []

        pbar = simplePbar(m_lags * n_segments)
        for block in range(n_segments):
            this_corr_matrix = np.empty(corr_matrix_shape)
            this_signals_block = signals[block * N_block:(block + 1) * N_block, :]
            for lag in range(m_lags):
                next(pbar)
                y_r = this_signals_block[:N_block - lag, ref_channels]
                y_a = this_signals_block[lag:, :]
                # standard un-biased estimator (revert rectangular window)
                this_corr_matrix[:, :, lag] = (y_a.T @ y_r) / (N_block - lag)
            corr_matrices.append(this_corr_matrix)

        corr_matrix = np.mean(corr_matrices, axis=0)

        if not np.all(corr_matrix.shape == corr_matrix_shape):
            raise RuntimeError(
                f"Computed correlation matrix shape {corr_matrix.shape} does not match "
                f"expected shape {corr_matrix_shape}; internal block-Toeplitz construction error."
            )

        self.corr_matrix_bt = corr_matrix
        self.corr_matrices_bt = np.stack(corr_matrices, axis=0)
        self.var_corr_bt = np.var(corr_matrices, axis=0)
        self.m_lags_bt = m_lags
        self.n_segments_bt = n_segments

        self.psd_matrix_bt = None
        self.s_vals_psd = None

        return corr_matrix

    def psd_blackman_tukey(self, n_lines=None, refs_only=True, window='hamming', **kwargs):
        '''
        Estimate the (cross- and auto-) power spectral densities (PSD),
        by Fourier Transform of correlation functions estimated
        according to Blackman-Tukey's method. Non-negativeness of the
        PSD is ensured by using a lag window, i.e. convolving the temporal
        window with itself. Normalization is applied w.r.t. conservation of
        energy, i.e. magnitudes will change with n_lines but power stays
        constant.

        Note that:
            m_lags = n_lines // 2 + 1
            n_lines = (m_lags - 1) * 2
            
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            refs_only: bool, optional
                Compute cross-PDSs only with reference channels
            window: str or tuple or array_like, optional
                Desired temporal window to be applied to the correlation
                sequence after conversion to a lag window by "self-convolution"
                See scipy.signal.get_window() for more information
            
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to self.corr_blackman_tukey

        Returns
        -------
            psd_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lines // 2 + 1)
                containing the power density values of the respective
                channels and frequencies
                
        '''
        logger.debug(f'Arguments psd_blackman_tukey: n_lines={n_lines}, refs_only={refs_only}, window={window}, {kwargs}')

        self._last_meth = 'blackman-tukey'

        N = self.total_time_steps
        n_lines = self._psd_bt_validate_n_lines(n_lines, N)

        # .. TODO:: implement multi-block psd
        n_segments = None
        n_lines, _ = self._psd_bt_resolve_params(n_lines, n_segments, N)

        cached = self._check_psd_bt_cache(n_lines, refs_only, kwargs)
        if cached is not None:
            return cached

        return self._psd_bt_compute(n_lines, refs_only, window, **kwargs)

    def _psd_bt_validate_n_lines(self, n_lines, N):
        """Validate n_lines for psd_blackman_tukey; return corrected n_lines or None."""
        if n_lines is not None:
            if not isinstance(n_lines, int):
                raise ValueError(
                    f"{n_lines} is not a valid number of n_lines for a spectral densities")
            if n_lines % 2:
                n_lines += 1
                logger.warning(
                    f"Only even number of frequency lines are supported setting n_lines={n_lines}")
            if n_lines > 2 * N:
                logger.warning(
                    f'Number of frequency lines {n_lines} should not'
                    f'be larger than twice the number of timesteps {self.total_time_steps}')
        return n_lines

    def _bt_load_cached_params(self, n_lines, n_segments):
        """Load n_lines/n_segments from BT cache when both are None."""
        if n_lines is None and n_segments is None:
            if self.n_lines_bt is None and self.m_lags_bt is not None:
                n_lines = (self.m_lags_bt - 1) * 2
            else:
                n_lines = self.n_lines_bt
            n_segments = self.n_segments_bt
            if n_lines is None and n_segments is None:
                raise RuntimeError('Either n_lines or n_segments must be provided on first run.')
        return n_lines, n_segments

    @staticmethod
    def _bt_resolve_cases(n_lines, n_segments, N):
        """Resolve (n_lines, N_segment) for BT cases 2–4."""
        if n_segments is None:
            return n_lines, n_lines
        if n_lines is None:
            N_segment = N // n_segments
            return N_segment, N_segment
        return n_lines, min(N // n_segments, n_lines)

    def _psd_bt_resolve_params(self, n_lines, n_segments, N):
        """Resolve n_lines/n_segments for psd_blackman_tukey; return (n_lines, N_segment)."""
        n_lines, n_segments = self._bt_load_cached_params(n_lines, n_segments)
        n_lines, N_segment = self._bt_resolve_cases(n_lines, n_segments, N)

        if n_lines % 2:
            n_lines += 1
        if N_segment > n_lines:
            raise ValueError(
                f"The segment length {N_segment} must not be larger than "
                f"the number of frequency lines {n_lines}")
        if N_segment < n_lines / 2:
            logger.warning(
                f"The segment length {N_segment} is much smaller than "
                f"the number of frequency lines {n_lines} (zero-padded)")
        return n_lines, N_segment

    def _check_psd_bt_cache(self, n_lines, refs_only, kwargs):
        """Return cached psd_matrix_bt if still valid, else None."""
        if kwargs:
            logger.debug("Not returning because: kwargs provided")
            return None
        if self.psd_matrix_bt is None:
            logger.debug("Not returning because: self.psd_matrix_bt not available")
            return None
        if self.psd_matrix_bt.shape[2] != n_lines // 2 + 1:
            logger.debug("Not returning because: n_lines differs from previous")
            return None
        if (self.psd_matrix_bt.shape[1] == self.num_ref_channels) != refs_only:
            logger.debug("Not returning because: non-/reference-based not matching previous")
            return None
        logger.debug("Using previously computed Power Spectral Density (BT)...")
        return self.psd_matrix_bt

    def _psd_bt_compute(self, n_lines, refs_only, window, **kwargs):
        """Compute and store the Blackman-Tukey PSD matrix; return it."""
        logger.info("Estimating Power Spectral Density by Blackman-Tukey's method...")

        corr_matrix = self.corr_blackman_tukey(n_lines // 2 + 1, refs_only=refs_only, **kwargs)

        num_analised_channels = self.num_analised_channels
        if refs_only:
            num_ref_channels = self.num_ref_channels
        else:
            num_ref_channels = num_analised_channels

        psd_matrix_shape = (num_analised_channels, num_ref_channels, n_lines // 2 + 1)
        psd_matrix = np.empty(psd_matrix_shape, dtype=complex)

        # create a symmetrical window, i.e. lacking the last 0 (for an even number of lines)
        win = scipy.signal.get_window(window, n_lines // 2, fftbins=True)
        # Zero-Pad both sides (= zero pad once and circular convolution)
        # to allow the window to "slide along" the correct number of lags in np.convolve = 3 * n_lines//2 - 1
        # here first (!) zero pad is n_lines//2-1 because it is convolve
        win_pad = np.concatenate((np.zeros(n_lines // 2 - 1), win, np.zeros(n_lines // 2)))
        # Convolve zero-padded and unpadded window
        # resulting shape: M - N + 1 = (3 * n_lines//2 - 1) - (n_lines//2) + 1 = 2 * n_lines//2 = n_lines
        corr_win = np.convolve(win_pad, win, 'valid')
        corr_win /= n_lines // 2  # unbiased not needed here, because it is "windowed"

        # normalization factor for power equivalence
        norm_fact = self.total_time_steps
        # equivalent noise bandwidth of the window for density scaling
        eq_noise_bw = np.sum(win ** 2) / np.sum(win) ** 2 * (n_lines // 2)

        pbar = simplePbar(num_analised_channels * num_ref_channels)
        for channel_1 in range(num_analised_channels):
            for channel_2 in range(num_ref_channels):
                next(pbar)
                corr_seq = corr_matrix[channel_1, channel_2, :]
                # https://en.wikipedia.org/wiki/Cross-correlation#Properties
                corr_sequence = np.concatenate(
                    (np.flip(corr_seq)[:n_lines // 2], corr_seq[:n_lines // 2]))
                corr_sequence *= norm_fact
                spec_btr = np.fft.fft(corr_sequence * corr_win)
                spec_btr = spec_btr[:n_lines // 2 + 1]
                spec_btr *= 2          # compensate one-sided
                spec_btr *= eq_noise_bw  # compensate window
                psd_matrix[channel_1, channel_2, :] = spec_btr

        logger.debug(f'PSD Auto-/Cross-Powers: {np.mean(np.abs(psd_matrix), axis=2)}')

        if self.scaling_factors is None:
            self.scaling_factors = psd_matrix.max(axis=2)

        self.n_lines_bt = n_lines
        self.psd_matrix_bt = psd_matrix
        self._last_meth = 'blackman-tukey'

        return psd_matrix

    def welch(self, n_lines, **kwargs):
        logger.warning("DeprecationWarning: method welch() will soon be dropped. Use psd_welch and/or corr_welch instead")
        psd_matrix = self.psd_welch(n_lines, **kwargs)
        corr_matrix = self.corr_welch()

        return corr_matrix, psd_matrix

    def correlation(self, m_lags=None, method=None, **kwargs):
        '''
        A convenience method for obtaining the correlation sequence by
        the default or any specified estimation method.
        
        Parameters
        ----------
            m_lags: integer, optional
                Number of lags (positive). Note: this includes the
                0-lag, therefore excludes the m_lags-lag.
            method: str, optional
                The method to use for spectral estimation
        
        Other Parameters
        -----------------
            kwargs:
                Additional parameters are passed to the spectral estimation method
        
        Returns
        -------
            corr_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, m_lags)
                containing the correlation values of the respective
                channels and lags
        
        '''
        logger.debug(f'Arguments correlation: m_lags={m_lags}, method={method}, {kwargs}')

        if method is None:
            if self._last_meth is None:
                method = 'blackman-tukey'
            else:
                method = self._last_meth
        if method == 'welch':
            return self.corr_welch(m_lags, **kwargs)
        elif method == 'blackman-tukey':
            return self.corr_blackman_tukey(m_lags, **kwargs)
        else:
            raise ValueError(f'Unknown method {method}')

    def psd(self, n_lines=None, method=None, **kwargs):
        '''
        A convenience method for obtaining the PSD by the default or any
        specified estimation method.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            method:
                The method to use for spectral estimation

        Other Parameters
        ----------------
            **kwargs:
                Additional parameters are passed to the spectral estimation method
                
        Returns
        -------
            psd_matrix: np.ndarray
                Array of shape (num_channels, num_ref_channels, n_lines // 2 + 1)
                containing the power density values of the respective
                channels and frequencies
        
        '''

        logger.debug(f'Arguments psd: n_lines={n_lines}, method={method}, {kwargs}')

        if method is None:
            if self._last_meth is None:
                method = 'welch'
            else:
                method = self._last_meth
        if method == 'welch':
            # if n_lines is None:
            #     n_lines = self.n_lines_wl
            # if not isinstance(n_lines, int):
            #     raise ValueError(f"{n_lines} is not a valid number of frequency lines for a psd sequence")
            return self.psd_welch(n_lines, **kwargs)
        elif method == 'blackman-tukey':
            # if n_lines is None:
            #     if self.n_lines_bt is not None:
            #         n_lines = self.n_lines_bt
            #     elif self.m_lags_bt is not None:
            #         n_lines = (self.m_lags_bt - 1) * 2
            # if not isinstance(n_lines, int):
            #     raise ValueError(f"{n_lines} is not a valid number of frequency lines for a psd sequence")
            return self.psd_blackman_tukey(n_lines, **kwargs)
        else:
            raise ValueError(f'Unknown method {method}')

    def sv_psd(self, n_lines=None, **kwargs):
        '''
        Compute the singular values of the power spectral density matrices,
        for which the complete (all cross spectral densities) matrices are used.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
        
        Other Parameters
        ----------------
            kwargs:
                Additional parameters are passed to the spectral estimation method
        '''

        if self.s_vals_psd is not None and (n_lines is None or self.s_vals_psd.shape[1] == n_lines // 2 + 1):
            return self.s_vals_psd

        psd_matrix = self.psd(n_lines,
                              # refs_only=False,
                              **kwargs)
        n_sigma = np.min(psd_matrix.shape[:2])
        # n_sigma = self.num_analised_channels

        n_lines = self.n_lines
        s_vals_psd = np.empty((n_sigma, n_lines // 2 + 1))
        for k in range(n_lines // 2 + 1):
            # might use only real part to account for slightly asynchronous data
            # see [Au (2017): OMA, Chapter 7.5]
            s_vals_psd[:, k] = np.linalg.svd(psd_matrix[:,:, k], True, False)

        self.s_vals_psd = s_vals_psd

        return s_vals_psd


class SignalPlot(object):
    """Plotting helper for :class:`PreProcessSignals`.

    Provides :meth:`plot_signals`, :meth:`plot_timeseries`,
    :meth:`plot_correlation`, :meth:`plot_psd`, and
    :meth:`plot_svd_spectrum` as convenience wrappers around the spectral
    estimation methods of :class:`PreProcessSignals`.

    Parameters
    ----------
    prep_signals : PreProcessSignals
        The pre-processed signal object to visualise.
    """

    def __init__(self, prep_signals):
        """
        Parameters
        ----------
        prep_signals : PreProcessSignals
            The pre-processed signal object to visualise.
        """
        if not isinstance(prep_signals, PreProcessSignals):
            logger.warning(f'Argument prep_signals ist not of type PreProcessSignals but {type(prep_signals)}')
        self.prep_signals = prep_signals

    def _plot_signals_setup_axes(self, per_channel_axes, psd_scale, num_channels,
                                  axest, axesf):
        """Create or validate axes for plot_signals; return (axest, axesf)."""
        if axest is None or axesf is None:
            axest, axesf = self._plot_signals_create_axes(
                per_channel_axes, psd_scale, num_channels, axest, axesf)
        # validate sizes
        if per_channel_axes:
            if len(axest) < num_channels:
                raise ValueError(
                    f'The number of provided axes objects '
                    f'(time domain) = {len(axest)} does not match the '
                    f'number of channels={num_channels}')
            if psd_scale != 'svd' and len(axesf) < num_channels:
                raise ValueError(
                    f'The number of provided axes objects '
                    f'(frequency domain) = {len(axesf)} does not match the '
                    f'number of channels={num_channels}')
        else:
            axest = self._broadcast_axes(axest, num_channels, 'time domain')
            axesf = self._broadcast_axes(axesf, num_channels, 'frequency domain')
        return axest, axesf

    @staticmethod
    def _create_per_channel_axes(psd_scale, num_channels, axest, axesf):
        """Create figure axes when per_channel_axes=True."""
        if psd_scale != 'svd':
            _, axes = plt.subplots(nrows=num_channels, ncols=2,
                                   sharey='col', sharex='col')
            if axest is None:
                axest = axes[:, 0]
            if axesf is None:
                axesf = axes[:, 1]
        else:
            if axest is None:
                nxn = int(np.ceil(np.sqrt(num_channels)))
                _, axest = plt.subplots(nrows=int(np.ceil(num_channels / nxn)),
                                        ncols=nxn, sharey=True, sharex=True)
                axest = axest.flatten()
            if axesf is None:
                _, axesf = plt.subplots(nrows=1, ncols=1)
                axesf = np.repeat(axesf, num_channels)
        return axest, axesf

    @staticmethod
    def _create_shared_axes(num_channels, axest, axesf):
        """Create figure axes when per_channel_axes=False."""
        if axest is None:
            _, axest = plt.subplots(nrows=1, ncols=1)
            axest = np.repeat(axest, num_channels)
        if axesf is None:
            _, axesf = plt.subplots(nrows=1, ncols=1)
            axesf = np.repeat(axesf, num_channels)
        return axest, axesf

    def _plot_signals_create_axes(self, per_channel_axes, psd_scale, num_channels,
                                   axest, axesf):
        """Create axes figures for plot_signals; return (axest, axesf)."""
        if per_channel_axes:
            return self._create_per_channel_axes(psd_scale, num_channels, axest, axesf)
        return self._create_shared_axes(num_channels, axest, axesf)

    @staticmethod
    def _broadcast_axes(axes, num_channels, label):
        """Expand a single-element axes to num_channels; raise if too short."""
        if not isinstance(axes, (tuple, list, np.ndarray)):
            return np.repeat(axes, num_channels)
        if len(axes) == 1:
            return np.repeat(axes, num_channels)
        if len(axes) < num_channels:
            raise ValueError(
                f'The number of provided axes objects ({label}) = {len(axes)} '
                f'does not match the number of channels={num_channels}')
        return axes

    def _plot_one_channel(self, axt, axf, channel, prep_signals, plot_ctx):
        """Plot time-domain and frequency-domain data for one channel.

        Parameters
        ----------
        plot_ctx : dict with keys timescale, psd_scale, refs, plot_kwarg_dict, refs_only, method
        """
        timescale = plot_ctx['timescale']
        psd_scale = plot_ctx['psd_scale']
        refs = plot_ctx['refs']
        plot_kwarg_dict = plot_ctx['plot_kwarg_dict']
        refs_only = plot_ctx['refs_only']
        method = plot_ctx['method']
        if timescale == 'lags':
            self.plot_correlation(prep_signals.m_lags, [channel], axt, timescale, refs,
                                  plot_kwarg_dict.copy(),
                                  refs_only=refs_only, method=method)
        else:
            self.plot_timeseries(channels=[channel], ax=axt,
                                 scale='timescale', **plot_kwarg_dict.copy())
        axt.grid(True, axis='y', ls='dotted')
        self.plot_psd(prep_signals.n_lines, [channel], axf, psd_scale, refs,
                      plot_kwarg_dict.copy(),
                      refs_only=refs_only, method=method)

    def plot_signals(self, channels=None, axest=None, axesf=None,
                     plot_kwarg_dict=None, **kwargs):
        '''
        Plot time domain and/or frequency domain signals in various configurations:
         1. time history and spectrum of a single channel in two axes -> set channels = [channel] goto 2
         2. time history of multiple channels (all channels or specified)
            * if axes arguments are not None:must be (tuples, lists, ndarrays) of size = (num_channels,) regardless of the actual figure layout
            * else: generate axes for each channel and arrange them in lists

            a. time domain overlay in a single axes -> single axes is repeated in the axes list
                i. spectrum overlay in a single axes -> single axes is repeated in the axes list
                ii. svd spectrum in a single axes -> needs an additional argument
            b. in multiple axes' in a grid figure -> axes are generated as subplots
                i. spectrum in multiple axes' -> axes are generated as subplots
                ii. svd spectrum in a single axes -> needs an additional argument

        Parameters
        ----------
            channels : None, list-of-int, list-of-str, int, str
                The selected channels (see self._channel_numbers for explanation)
            axest: ndarray of size num_channels of matplotlib.axes.Axes objects
                User provided axes objects, into which to plot time domain signals
            axesf: ndarray of size num_channels of matplotlib.axes.Axes objects
                User provided axes objects, into which to plot spectra

        Other Parameters
        ----------------
            per_channel_axes: bool
                Whether to plot all channels into a single or multiple axes
            timescale: str ['time', 'samples', 'lags']
                Whether to display time, sample or lag values on the horizontal axis
                'lags' implies plotting (auto)-correlations instead of raw time histories
            psd_scale: str, ['db', 'power', 'rms', 'svd', 'phase']
                Scaling/Output quantity of the ordinate (value axis)
            plot_kwarg_dict:
                A dictionary to pass arguments to matplotlib.plot
            kwargs:
                Additional kwargs are passed to the spectral estimation method

        .. TO DO::
            * share y-axis scaling on axes' only between channels of the same
              measurement quantity (acceleration, velocity, displacement/strains)
        '''
        per_channel_axes = kwargs.pop('per_channel_axes', False)
        timescale = kwargs.pop('timescale', 'time')
        psd_scale = kwargs.pop('psd_scale', 'db')
        if plot_kwarg_dict is None:
            plot_kwarg_dict = {}
        prep_signals = self.prep_signals
        refs = kwargs.pop('refs', None)
        channel_numbers, ref_numbers = prep_signals._channel_numbers(channels, refs)
        all_ref_numbers = set(sum(ref_numbers, []))
        # if all requested reference channels are in prep_signals.ref_channels,
        # a reduced correlation function may be computed
        refs_only = all_ref_numbers.issubset(prep_signals.ref_channels)
        # if not all are needed, but the user requested so, compute full correlation matrix
        if (refs_only and not kwargs.pop('refs_only', True)) or psd_scale == 'svd':
            refs_only = False
        num_channels = len(channel_numbers)

        axest, axesf = self._plot_signals_setup_axes(
            per_channel_axes, psd_scale, num_channels, axest, axesf)

        # precompute relevant spectral matrices
        n_lines = kwargs.pop('n_lines', None)
        method = kwargs.pop('method', None)
        prep_signals.psd(n_lines, method, refs_only=refs_only, **kwargs.copy())
        if timescale == 'lags':
            prep_signals.correlation(prep_signals.m_lags, method, refs_only=refs_only, **kwargs.copy())

        plot_ctx = {
            'timescale': timescale,
            'psd_scale': psd_scale,
            'refs': refs,
            'plot_kwarg_dict': plot_kwarg_dict,
            'refs_only': refs_only,
            'method': method,
        }
        for axt, axf, channel in zip(axest, axesf, channel_numbers):
            self._plot_one_channel(axt, axf, channel, prep_signals, plot_ctx)

        if not per_channel_axes:
            axest[-1].legend()
            axesf[-1].legend()
        else:
            figt = axest[0].get_figure()
            figt.legend()
            figf = axesf[0].get_figure()
            figf.legend()

        return axest, axesf

    @staticmethod
    def _channel_quantity_label(prep_signals, channel):
        """Return the single-letter quantity label ('a', 'v', 'd', or 'f') for a channel."""
        if channel in prep_signals.accel_channels:
            return 'a'
        if channel in prep_signals.velo_channels:
            return 'v'
        if channel in prep_signals.disp_channels:
            return 'd'
        return 'f'

    def plot_timeseries(self, channels=None, ax=None, scale='time', **kwargs):
        '''
        Plots the time histories of the signals
        
        Parameters
        ----------
            channels : int, list, tuple, np.ndarray
                The channels to plot, may be names, indices, etc.
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object to plot into
            scale: str, ['lags','samples']
                Whether to display time or sample values on the horizontal axis
        
        Other Parameters
        ----------------
            kwargs :
                Additional kwargs are passed to matplotlib.plot
        Returns
        -------
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object containing the graphs
                
        .. TODO::
             * correct labeling of channels and axis (using accel\_, velo\_, and disp\_channels)
        '''

        prep_signals = self.prep_signals
        signals = prep_signals.signals

        t = prep_signals.t
        if scale == 'samples':
            t *= prep_signals.sampling_rate
            xlabel = r'$n\,[-]$'
            ylabel = r'$f[n]$'
        else:
            xlabel = r'$t\,[\mathrm{s}]$'
            ylabel = r'$f(t)$'

        channel_numbers, _ = prep_signals._channel_numbers(channels)

        if ax is None:
            ax = plt.subplot(111)

        for channel in channel_numbers:
            f = self._channel_quantity_label(prep_signals, channel)
            channel_name = prep_signals.channel_headers[channel]
            ax.plot(t, signals[:, channel], label=rf'${f}_\mathrm{{{channel_name}}}$', **kwargs)

        ax.set_xlim((0, prep_signals.duration))
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(xlabel)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel)

        return ax

    def plot_correlation(self, m_lags=None, channels=None, ax=None,
                         scale='lags', refs=None, plot_kwarg_dict=None, **kwargs):
        '''
        Plots the Cross- and Auto-Correlation sequences of the signals.
        If correlations have not been estimated yet and no method
        parameter is supplied, Blackman-Tukeys's method is used, else the
        most recently used estimation method is employed.
        
        Parameters
        ----------
            m_lags: integer, optional
                Number of lags (positive). Note: this includes the
                0-lag, therefore excludes the m_lags-lag.
            channels : int, list, tuple, np.ndarray
                The channels to plot, may be names, numbers/indices, etc.
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object to plot into
            scale: str, ['lags','samples']
                Whether to display lag or sample values on the horizontal axis
            refs: 'auto', list-of-indices, optional
                Reference channels to consider for cross-correlations
            
        Other Parameters
        ----------------
            method:
                The method to use for spectral estimation
            plot_kwarg_dict:
                A dictionary to pass arguments to matplotlib.plot
            **kwargs :
                Additional kwargs are passed to the spectral estimation
                method or contain figure/axes formatting options

        Returns
        -------
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object containing the graphs
                
        .. TODO::
            * correct labeling of channels and axis (using accel\_, velo\_, and disp\_channels)

        '''
        if plot_kwarg_dict is None:
            plot_kwarg_dict = {}

        prep_signals = self.prep_signals
        method = kwargs.pop('method', prep_signals._last_meth)
        channel_numbers, ref_numbers = prep_signals._channel_numbers(channels, refs)
        all_ref_numbers = set(sum(ref_numbers, []))
        refs_only = all_ref_numbers.issubset(prep_signals.ref_channels)
        refs_only = self._resolve_corr_refs_only(refs_only, method, prep_signals, kwargs)

        corr_matrix = prep_signals.correlation(m_lags, refs_only=refs_only, method=method, **kwargs)

        if refs_only is not (prep_signals.num_ref_channels == corr_matrix.shape[1]):
            raise ValueError(
                f"refs_only={refs_only!r} is inconsistent with the returned correlation matrix: "
                f"num_ref_channels={prep_signals.num_ref_channels}, corr_matrix.shape[1]={corr_matrix.shape[1]}"
            )

        lags = prep_signals.lags
        if scale == 'samples':
            lags *= prep_signals.sampling_rate
            xlabel = r'$m\,[-]$'
            ylabel = r'$\hat{R}_{i,j}[m]$'
        else:
            xlabel = r'$\tau\,[\mathrm{s}]$'
            ylabel = r'$\hat{R}_{i,j}(\tau)$'

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        norm_fact = self._corr_norm_factor(prep_signals)

        for channel_number, current_ref_numbers in zip(channel_numbers, ref_numbers):
            channel_name = prep_signals.channel_headers[channel_number]
            for ref_index, ref_number in enumerate(current_ref_numbers):
                corr = self._extract_corr(corr_matrix, channel_number, ref_index, ref_number, refs_only)
                label = self._corr_label(channel_name, ref_number, channel_number,
                                         prep_signals.channel_headers)
                ax.plot(lags, corr * norm_fact, label=label, **plot_kwarg_dict)

        ax.set_xlim((0, lags.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    @staticmethod
    def _extract_corr(corr_matrix, channel_number, ref_index, ref_number, refs_only):
        """Extract one correlation trace from the correlation matrix."""
        if refs_only:
            return corr_matrix[channel_number, ref_index, :]
        return corr_matrix[channel_number, ref_number, :]

    @staticmethod
    def _corr_label(channel_name, ref_number, channel_number, channel_headers):
        """Build a correlation plot label for one channel pair."""
        if ref_number == channel_number:
            return rf'$\hat{{R}}_\mathrm{{{channel_name}}}$'
        return rf'$\hat{{R}}_\mathrm{{{channel_headers[ref_number]},{channel_name}}}$'

    @staticmethod
    def _resolve_corr_refs_only(refs_only, method, prep_signals, kwargs):
        """Resolve refs_only for plot_correlation based on precomputed matrices and user input."""
        if refs_only:
            if method == 'welch' and prep_signals.corr_matrix_wl is not None:
                refs_only = prep_signals.num_ref_channels == prep_signals.corr_matrix_wl.shape[1]
                logger.debug('reverting refs_only: False -> Welch precomputed')
            elif method == 'blackman-tukey' and prep_signals.corr_matrix_bt is not None:
                refs_only = prep_signals.num_ref_channels == prep_signals.corr_matrix_bt.shape[1]
                logger.debug('reverting refs_only: False -> Blackman-Tukey precomputed')
            if not kwargs.pop('refs_only', True):
                refs_only = False
                logger.debug('reverting refs_only: False -> User input')
        return refs_only

    @staticmethod
    def _corr_norm_factor(prep_signals):
        """Return the normalisation factor for correlation plotting."""
        if prep_signals._last_meth == 'welch':
            return prep_signals.n_lines_wl
        if prep_signals._last_meth == 'blackman-tukey':
            return prep_signals.total_time_steps
        raise RuntimeError('Last used method was not stored in prep_signals object.')

    def plot_psd(self, n_lines=None, channels=None, ax=None,
                 scale='db', refs=None, plot_kwarg_dict=None, **kwargs):
        '''
        Plots the Cross- and Auto-Power-Spectral Density of the signals.
        PSD estimation is performed by default using Welch's method.
        
        Parameters
        ----------
            n_lines: integer, optional
                Number of frequency lines (positive + negative)
            channels : int, list, tuple, np.ndarray
                The channels to plot, may be names, indices, etc.
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object to plot into
            scale: str, ['db', 'power', 'rms', 'svd', 'phase']
               Scaling/Output quantity of the ordinate (value axis)
            refs: 'auto', list-of-indices, optional
                Reference channels to consider for cross-correlations
        
        Other Parameters
        ----------------
            method:
                The method to use for spectral estimation
            plot_kwarg_dict:
                A dictionary to pass arguments to matplotlib.plot
            **kwargs :
                Additional kwargs are passed to the spectral estimation method

        Returns
        -------
            ax: matplotlib.axes.Axes, optional
                Matplotlib Axes object containing the graphs
                
        .. TODO::
            * correct labeling of channels and axis (using accel\_, velo\_, and disp\_channels)
            * do we need a svd in non-db scale?
            * do we need sample scaling on the abscissa
        '''
        if plot_kwarg_dict is None:
            plot_kwarg_dict = {}

        prep_signals = self.prep_signals
        if scale not in ['db', 'power', 'rms', 'svd', 'phase']:
            raise ValueError(
                f"scale must be one of 'db', 'power', 'rms', 'svd', 'phase', got {scale!r}")

        method = kwargs.pop('method', None)
        channel_numbers, ref_numbers, refs_only, psd_matrix = self._psd_resolve_matrix(
            prep_signals, scale, channels, refs, n_lines, method, kwargs)

        freqs = prep_signals.freqs

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        for channel_number, current_ref_numbers in zip(channel_numbers, ref_numbers):
            channel_name = prep_signals.channel_headers[channel_number]
            for ref_index, ref_number in enumerate(current_ref_numbers):
                psd, label = self._psd_channel_data(
                    psd_matrix, channel_number, ref_index, ref_number,
                    channel_name, refs_only, scale, prep_signals)
                ax.plot(freqs, psd, label=label, **plot_kwarg_dict)

        ax.set_xlim((0, freqs.max()))
        ax.set_xlabel(r'$f\,[\mathrm{Hz}]$')
        self._psd_set_ylabel(ax, scale)

        return ax

    @staticmethod
    def _refine_refs_only(refs_only, method, prep_signals, kwargs):
        """Narrow refs_only based on cached matrix dimensions and kwarg override."""
        if refs_only:
            if method == 'welch' and prep_signals.psd_matrix_wl is not None:
                refs_only = prep_signals.num_ref_channels == prep_signals.psd_matrix_wl.shape[1]
            elif method == 'blackman-tukey' and prep_signals.psd_matrix_bt is not None:
                refs_only = prep_signals.num_ref_channels == prep_signals.psd_matrix_bt.shape[1]
            if not kwargs.pop('refs_only', True):
                refs_only = False
        return refs_only

    @staticmethod
    def _psd_resolve_matrix(prep_signals, scale, channels, refs, n_lines, method, kwargs):
        """Resolve psd_matrix and channel/ref lists for plot_psd."""
        if scale == 'svd':
            if refs is not None or kwargs.pop('refs_only', False):
                logger.warning("Reference channels are not used in SVD PSD.")
            channel_numbers, ref_numbers = prep_signals._channel_numbers(channels, [0])
            psd_matrix = prep_signals.sv_psd(n_lines, method=method, refs_only=False, **kwargs)
            return channel_numbers, ref_numbers, False, psd_matrix
        channel_numbers, ref_numbers = prep_signals._channel_numbers(channels, refs)
        all_ref_numbers = set(sum(ref_numbers, []))
        refs_only = all_ref_numbers.issubset(prep_signals.ref_channels)
        refs_only = SignalPlot._refine_refs_only(refs_only, method, prep_signals, kwargs)
        psd_matrix = prep_signals.psd(n_lines, refs_only=refs_only, method=method, **kwargs)
        if refs_only is not (prep_signals.num_ref_channels == psd_matrix.shape[1]):
            raise ValueError(
                f"refs_only={refs_only!r} is inconsistent with the returned PSD matrix: "
                f"num_ref_channels={prep_signals.num_ref_channels}, "
                f"psd_matrix.shape[1]={psd_matrix.shape[1]}"
            )
        return channel_numbers, ref_numbers, refs_only, psd_matrix

    @staticmethod
    def _psd_channel_data(psd_matrix, channel_number, ref_index, ref_number,
                          channel_name, refs_only, scale, prep_signals):
        """Extract and scale psd values and construct label for one channel pair."""
        if scale == 'svd':
            psd = 10 * np.log10(np.abs(psd_matrix[channel_number, :]))
            label = rf'$\hat{{\sigma}}_\mathrm{{{channel_number}}}$'
            return psd, label
        if refs_only:
            psd = psd_matrix[channel_number, ref_index, :]
        else:
            psd = psd_matrix[channel_number, ref_number, :]
        if scale == 'db':
            psd = 10 * np.log10(np.abs(psd))
        elif scale == 'power':
            psd = np.abs(psd)
        elif scale == 'rms':
            psd = np.sqrt(np.abs(psd))
        elif scale == 'phase':
            psd = np.angle(psd) / np.pi * 180
        if ref_number == channel_number:
            label = rf'$\hat{{S}}_\mathrm{{{channel_name}}}$'
        else:
            ref_name = prep_signals.channel_headers[ref_number]
            label = rf'$\hat{{S}}_\mathrm{{{ref_name},{channel_name}}}$'
        return psd, label

    @staticmethod
    def _psd_set_ylabel(ax, scale):
        """Set the y-axis label on *ax* according to *scale*."""
        labels = {
            'svd': 'Singular Value Magnitude [dB]',
            'db': 'PSD [dB]',
            'power': 'Power Spectral Density [...]',
            'rms': 'Magnitude Spectral Density [...]',
            'phase': 'Cross Spectrum Phase[°]',
        }
        ax.set_ylabel(labels.get(scale, ''))

    def plot_svd_spectrum(self, NFFT=512, log_scale=True, ax=None):
        prep_signals = self.prep_signals
        logger.warning("DeprecationWarning: method plot_svd_spectrum() will soon be dropped. Use plot_psd(scale='svd')")
        if not log_scale:
            raise NotImplementedError("Log scale for SVD plots cannot be deactivated")
        return prep_signals.plot_psd(n_lines=NFFT, ax=ax, scale='svd')


def load_measurement_file(fname, **kwargs):
    '''
    assign this function to the class before instantiating the object
    PreProcessSignals.load_measurement_file = load_measurement_file
    '''

    # define a function to return the following variables
    headers = ['channel_name', 'channel_name']
    units = ['unit', 'unit', ]
    start_time = datetime.datetime()
    sample_rate = float()
    measurement = np.array([])

    # channels im columns
    if not (measurement.shape[0] > measurement.shape[1]):
        raise ValueError(
            f"measurement must have more rows (time steps) than columns (channels), "
            f"got shape {measurement.shape}"
        )

    return headers, units, start_time, sample_rate, measurement


def main():
    pass


def spectral_estimation():
    # signal parameters
    N = 2 ** 15
    fs = 128
    _dt = 1 / fs

    t, y, omegas, psd, corr = SDOF_ambient(N, fs)
    # spectral estimation parameters
    nperseg_fac = 1
    _window = np.hamming
    n_lines = N // nperseg_fac

    _tau = np.linspace(0, n_lines / fs, n_lines, False)
    _omegasr = np.fft.rfftfreq(n_lines, 1 / fs) * 2 * np.pi

    do_plot = True

    if do_plot:
        _fig1, axes = plt.subplots(2, 2, sharex='row', sharey='row')
        ax1, ax2, ax3, ax4 = axes.flat
        for ax in axes.flat:
            ax.axhline(0, color='gray', linewidth=0.5)
        handles = []

    if do_plot:
        ax1.plot(np.fft.fftshift(omegas) / 2 / np.pi, np.fft.fftshift(psd), label='analytic', color='black', lw=0.5)
        ax3.plot(t, corr, label='analytic', color='black', lw=0.5)
        ax2.plot(np.fft.fftshift(omegas) / 2 / np.pi, np.fft.fftshift(psd), label='analytic', color='black', lw=0.5)
        handles.append(ax4.plot(t, corr, label='analytic', color='black', lw=0.5)[0])

    print(f'Theoretic powers')
    print(f'PSD: {np.mean(psd)}')
    # print(f'0-lag correlation: {correlation[0]}')

    prep_signals = PreProcessSignals(y[:, np.newaxis], fs)
    prep_signals.plot_signals(timescale='lags', axest=[ax3], axesf=[ax1], dbscale=False)
    plt.show()


def SDOF_ambient(N=2 ** 15, fs=128):
    _dt = 1 / fs

    omegas = np.fft.fftfreq(N, 1 / fs) * 2 * np.pi
    t = np.linspace(0, N / fs, N, False)

    # generate sdof system
    zeta = 0.05
    omega = 5 * 2 * np.pi * np.sqrt(1 - zeta ** 2)  # damped f = 5 Hz
    m = 1
    k = omega ** 2 * m
    # c = zeta*2*sqrt(m*k)
    H = -omegas ** 2 / (k * (1 + 2j * zeta * omegas / omega - (omegas / omega) ** 2))

    # generate ambient input forces
    f_scale = 10
    phase = np.random.uniform(-np.pi, np.pi, (N // 2 + 1,))
    ampli = np.exp(1j * np.concatenate((phase[:N // 2 + N % 2], -1 * np.flip(phase[1:]))))
    Pomega = f_scale * np.ones(N, dtype=complex) * ampli

    # make the ifft real-valued
    Pomega.imag[0] = 0
    Pomega[N // 2 + N % 2] = np.abs(Pomega[N // 2 + N % 2])
    H.imag[0] = 0
    H[N // 2 + N % 2] = np.abs(H[N // 2 + N % 2])

    # generate the  ambient response signal
    y = np.fft.ifft(H * Pomega)
    # add noise
    noise = np.random.normal(0, 0.125, N)  # noise adds zero energy due to zero mean?

    # discard machine-precision zero imaginary parts
    if np.all(np.isclose(y.imag, 0)): y = y.real
    else: raise RuntimeError()

    power = np.sum(y ** 2)
    power_noise = np.sum(noise ** 2)
    print(f'Power time-domain: {power}')
    print(f'SNR={10*np.log10(power/power_noise)} dB')

    # compute analytical spectrum and correlation functions with correct scaling
    # psd = np.abs(H)**2*f_scale**2
    psd = omegas ** 4 / (k ** 2 * (1 + (4 * zeta ** 2 - 2) * (omegas / omega) ** 2 + (omegas / omega) ** 4)) * f_scale ** 2
    psd /= np.mean(psd)
    # analytical solution for convolution difficult, use numerical inverse of analytical PSD
    corr = np.fft.ifft(psd)
    # discard machine-precision zero imaginary parts
    if np.all(np.isclose(corr.imag, 0)): corr = corr.real
    else: raise RuntimeError()

    return t, (y + noise) / np.sqrt(power), omegas, psd, corr


def system_frf(N=2 ** 16, fmax=130, L=200, E=2.1e11, rho=7850, A=0.0343, zeta=0.01):

    df = fmax / (N // 2 + 1)
    dt = 1 / df / N
    _fs = 1 / dt

    omegas = np.linspace(0, fmax, N // 2 + 1, False) * 2 * np.pi
    expected_domega = (omegas[-1] - omegas[0]) / (N // 2 + 1 - 1)
    if df * 2 * np.pi != expected_domega:
        raise RuntimeError(
            f"Frequency resolution mismatch: df*2*pi={df * 2 * np.pi!r} "
            f"!= (omegas[-1]-omegas[0])/(N//2)={expected_domega!r}; internal frequency grid inconsistency."
        )

    num_modes = int(np.floor((fmax * 2 * np.pi * np.sqrt(rho / E) * L / np.pi * 2 + 1) / 2))
    omegans = (2 * np.arange(1, num_modes + 1) - 1) / 2 * np.pi / L * np.sqrt(E / rho)

    frf = np.zeros((N // 2 + 1,), dtype=complex)
    zetas = np.zeros_like(omegans)
    zetas[:] = zeta
    kappas = omegans ** 2
#     kappas[:]=E*A/L
    for j, (omegan, zeta) in enumerate(zip(omegans, zetas)):
        frf += -omegan ** 2 / (kappas[j] * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan) ** 2))  # Accelerance

    return omegas, frf


if __name__ == '__main__':
    main()
