# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Backend-agnostic mode-shape computation base class.

:class:`ModeShapeBase` collects everything about a mode-shape visualisation
that does not depend on a particular rendering backend: constructor-argument
validation, merging-mode detection (single-setup, PoGER/PreGER and PoSER),
mode lookup, the channel-DOF -> node displacement math and the
amplitude/part/visibility state machine.

A concrete backend (e.g. :class:`~pyOMA.core.PlotMSH.ModeShapePlot`)
subclasses this and supplies the rendering by overriding the ``_render_*``
template-method hooks, the :attr:`widget` property and ``draw_msh``.

This module intentionally imports no plotting library of any kind.
"""

import dataclasses
import warnings
import os
from pathlib import Path

import numpy as np

from .PostProcessingTools import MergePoSER
from .VarSSIRef import VarSSIRef
from .VarPLSCF import VarPLSCF
from .SSICovRef import PogerSSICovRef
from .MultiSetupSSI import PreGERSSI, VarPreGERSSI
from .ModalBase import ModalBase
from .PreProcessingTools import PreProcessSignals
from .StabilDiagram import StabilCalc
from .Helpers import calc_xyz

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def require_picking_backend(mode_shape_plot, editor_name):
    '''Raise :class:`TypeError` unless *mode_shape_plot* supports picking.

    The geometry and channel-DOF editors select artists directly off a
    matplotlib ``Axes3D`` and mutate its artist containers.  A backend
    without that (the pyvista ones render through VTK and have no
    ``subplot``) must be refused up front, with an explanation, rather
    than failing with an :class:`AttributeError` somewhere inside a
    redraw.

    Parameters
    ----------
    mode_shape_plot : ModeShapeBase
        Plot object the editor was asked to work with.
    editor_name : str
        Editor name, used in the error message.

    Raises
    ------
    TypeError
        If the backend does not support picking.
    '''
    if not getattr(mode_shape_plot, 'supports_picking', False):
        raise TypeError(
            f'{editor_name} needs a mode-shape backend that supports '
            f'matplotlib picking, but got '
            f'{type(mode_shape_plot).__name__!r}. Geometry editing is '
            f'matplotlib-only; use ModeShapePlot for editing and a pyvista '
            f'backend for display.')


@dataclasses.dataclass
class ModeShapePlotConfig:
    """Visual style configuration for :class:`ModeShapePlot`.

    Group visual-style keyword arguments so that :class:`ModeShapePlot`
    can be constructed with a single *config* object instead of many
    individual keyword parameters.

    Parameters
    ----------
    beamcolor : color or sequence, optional
        Color used to draw beam/line elements.  Matplotlib-specific.
    beamstyle : str or sequence, optional
        Linestyle used to draw beam/line elements.  Matplotlib-specific.
    nodecolor : color, optional
        Color used to draw nodes.  Matplotlib-specific.
    nodemarker : marker, optional
        Marker symbol used to draw nodes.  Matplotlib-specific.
    nodesize : float, optional
        Marker size for nodes.  Matplotlib-specific.
    dpi : int, optional
        Figure resolution in dots per inch.
    amplitude : float, optional
        Scaling factor for modal displacement amplitudes.
    linewidth : float or sequence, optional
        Line width for beam/line elements.  Matplotlib-specific.
    callback_fun : callable or None, optional
        Called after each mode change; signature ``f(plot, mode_index)``.
    real : bool, optional
        When *True*, plot the real part of complex mode shapes.
    scale : float, optional
        Fractional scale for axis arrows and channel-DOF arrows.
    save_ani_path : pathlib.Path or None, optional
        Directory in which animation frames are saved.

    Notes
    -----
    The fields ``beamcolor``, ``beamstyle``, ``nodecolor``, ``nodemarker``,
    ``nodesize`` and ``linewidth`` are specific to the plotting backend
    and are ignored by backend-agnostic code.  They are retained on
    this dataclass for backwards compatibility.
    """

    beamcolor: object = 'dimgrey'
    beamstyle: object = '-'
    nodecolor: object = 'dimgrey'
    nodemarker: object = 'o'
    nodesize: float = 20
    dpi: int = 100
    amplitude: float = 1
    linewidth: object = 1
    callback_fun: object = None
    real: bool = False
    scale: float = 0.2
    save_ani_path: object = None


class ModeShapeBase(object):
    """Backend-agnostic computation core for mode-shape visualisation.

    Subclasses provide rendering by overriding :meth:`_render_mode`,
    :meth:`_render_amplitude`, :meth:`_render_part`, :meth:`draw_msh`,
    :meth:`compute_mode_displacements`'s rendering counterparts and the
    :attr:`widget` property.  See :class:`~pyOMA.core.PlotMSH.ModeShapePlot`
    for the reference rendering implementation.
    """

    # ── Configuration & validation ────────────────────────────────────────────

    def _resolve_config(self, config, fig, kwargs):
        '''Handle config vs. legacy individual style parameters.

        Pops legacy style keys from *kwargs*, builds a
        :class:`ModeShapePlotConfig` when needed, and returns
        ``(config, fig)``.

        Parameters
        ----------
        config : ModeShapePlotConfig or None
        fig : figure object or None
        kwargs : dict
            Remaining keyword arguments (modified in-place).

        Returns
        -------
        config : ModeShapePlotConfig
        fig : figure object or None
        '''
        _legacy_keys = [
            'amplitude', 'real', 'scale', 'dpi', 'nodecolor', 'nodemarker',
            'nodesize', 'beamcolor', 'beamstyle', 'linewidth', 'callback_fun',
            'save_ani_path',
        ]
        _legacy_params = {k: kwargs.pop(k, None) for k in _legacy_keys}
        _any_legacy = any(v is not None for v in _legacy_params.values())
        if _any_legacy and config is not None:
            raise ValueError(
                "Pass either 'config' or individual style parameters, not both.")
        if _any_legacy:
            warnings.warn(
                "Passing individual style parameters to ModeShapePlot is deprecated. "
                "Use ModeShapePlotConfig and pass it as config=ModeShapePlotConfig(...).",
                DeprecationWarning,
                stacklevel=3,
            )
            config = self._build_legacy_config(_legacy_params, _legacy_keys)
        if config is None:
            config = ModeShapePlotConfig()
        return config, fig

    @staticmethod
    def _build_legacy_config(legacy_params, keys):
        """Build a ModeShapePlotConfig from legacy keyword arguments."""
        defaults = ModeShapePlotConfig()
        kwargs = {}
        for k in keys:
            kwargs[k] = legacy_params[k] if legacy_params[k] is not None else getattr(defaults, k)
        return ModeShapePlotConfig(**kwargs)

    @staticmethod
    def _check_type(value, name, cls):
        """Raise TypeError if *value* is not None and not an instance of *cls*."""
        if value is not None and not isinstance(value, cls):
            raise TypeError(
                f"Expected {cls.__name__} for {name!r}, got {type(value).__name__!r}."
            )

    def _validate_data_types(self, stabil_calc, modal_data, prep_signals, merged_data):
        '''Type-check optional constructor arguments; raise TypeError on mismatch.

        Parameters
        ----------
        stabil_calc : StabilCalc or None
        modal_data : ModalBase or None
        prep_signals : PreProcessSignals or None
        merged_data : MergePoSER or None
        '''
        self._check_type(stabil_calc, 'stabil_calc', StabilCalc)
        self._check_type(modal_data, 'modal_data', ModalBase)
        self._check_type(prep_signals, 'prep_signals', PreProcessSignals)
        self._check_type(merged_data, 'merged_data', MergePoSER)

    @staticmethod
    def _check_bool(name, val):
        '''Validate a bool field; raise TypeError if not bool.'''
        if not isinstance(val, bool):
            raise TypeError(
                f"Expected bool for {name!r}, got {type(val).__name__!r}.")
        return val

    @staticmethod
    def _check_numeric(name, val):
        '''Validate an int or float field; raise TypeError otherwise.'''
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"Expected int or float for {name!r}, got {type(val).__name__!r}.")
        return val

    @staticmethod
    def _check_int(name, val):
        '''Validate an int field; raise TypeError otherwise.'''
        if not isinstance(val, int):
            raise TypeError(
                f"Expected int for {name!r}, got {type(val).__name__!r}.")
        return val

    @staticmethod
    def _check_numeric_or_seq(name, val):
        '''Validate an int/float or list/tuple/ndarray.'''
        if (not isinstance(val, (int, float))
                and not isinstance(val, (list, tuple, np.ndarray))):
            raise TypeError(
                f"Expected int, float, list, tuple, or ndarray for {name!r}, "
                f"got {type(val).__name__!r}.")
        return val

    @staticmethod
    def _check_callable_or_none(name, val):
        '''Validate callable or None; raise TypeError otherwise.'''
        if val is not None and not callable(val):
            raise TypeError(
                f"{name!r} must be callable, got {type(val).__name__!r}.")
        return val

    @staticmethod
    def _check_path_or_none(name, val):
        '''Validate Path or None; raise TypeError otherwise.'''
        if val is not None and not isinstance(val, Path):
            raise TypeError(
                f"Expected Path for {name!r}, got {type(val).__name__!r}.")
        return val

    # ── Merging-mode detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_merging_mode(merged_data, modal_data):
        '''Return the merging mode string based on which objects were supplied.

        Parameters
        ----------
        merged_data : MergePoSER or None
        modal_data : ModalBase or None

        Returns
        -------
        str or None
            ``'PoSER'``, ``'PoGER'``, ``'single'``, or *None*.
        '''
        if merged_data is not None:
            return 'PoSER'
        if isinstance(modal_data, (PogerSSICovRef, PreGERSSI)):
            # PreGER produces the same merged-shape / merged_chan_dofs structure
            # as PoGER, so it re-uses the PoGER plotting path.
            return 'PoGER'
        if modal_data is not None:
            return 'single'
        return None

    def _check_merging_requirements(self, merging, merged_data, modal_data,
                                    prep_signals, stabil_calc):
        '''Validate required/unnecessary arguments for the detected merging mode.

        Parameters
        ----------
        merging : str or None
        merged_data, modal_data, prep_signals, stabil_calc :
            Constructor arguments.
        '''
        if merging == 'PoSER':
            req = {}
            nreq = {'modal_data': modal_data, 'prep_signals': prep_signals,
                    'stabil_calc': stabil_calc}
        elif merging == 'PoGER':
            req = {'modal_data': modal_data, 'stabil_calc': stabil_calc}
            nreq = {'prep_signals': prep_signals, 'merged_data': merged_data}
        elif merging == 'single':
            req = {'modal_data': modal_data, 'stabil_calc': stabil_calc}
            nreq = {'merged_data': merged_data}
        else:
            req = {}
            nreq = {'prep_signals': prep_signals, 'stabil_calc': stabil_calc}
        for name, obj in req.items():
            if obj is None:
                raise TypeError(
                    f'Identified merging routine: {merging} requires argument '
                    f'{name}, which has not been provided.')
        for name, obj in nreq.items():
            if obj is not None:
                logger.info(
                    f'Identified merging routine: {merging} will not use '
                    f'argument {name}.')

    def _detect_and_apply_merging(self, merged_data, modal_data, prep_signals, stabil_calc):
        '''Detect merging mode and populate modal-data attributes on *self*.

        Parameters
        ----------
        merged_data, modal_data, prep_signals, stabil_calc :
            Constructor arguments.
        '''
        merging = self._detect_merging_mode(merged_data, modal_data)
        self._check_merging_requirements(merging, merged_data, modal_data,
                                         prep_signals, stabil_calc)
        if merging == 'PoSER':
            self._apply_poser_attrs(merged_data)
        elif merging == 'PoGER':
            self._apply_poger_attrs(modal_data, stabil_calc)
        elif merging == 'single':
            self._apply_single_attrs(modal_data, stabil_calc)
        else:
            self._apply_empty_attrs(prep_signals)

    def _apply_poser_attrs(self, merged_data):
        '''Populate instance attributes for the PoSER merging case.'''
        self.chan_dofs = merged_data.merged_chan_dofs
        self.num_channels = merged_data.merged_num_channels
        self.modal_frequencies = merged_data.mean_frequencies
        self.modal_damping = merged_data.mean_damping
        self.mode_shapes = merged_data.merged_mode_shapes
        self.std_frequencies = merged_data.std_frequencies
        self.std_damping = merged_data.std_damping
        self.select_modes = list(zip(
            range(len(self.modal_frequencies)),
            [0] * len(self.modal_frequencies)))
        self.setup_name = merged_data.setup_name
        self.start_time = merged_data.start_time

    def _apply_poger_attrs(self, modal_data, stabil_calc):
        '''Populate instance attributes for the PoGER merging case.'''
        self.chan_dofs = modal_data.merged_chan_dofs
        self.num_channels = modal_data.merged_num_channels
        self.modal_frequencies = modal_data.modal_frequencies
        self.modal_damping = modal_data.modal_damping
        self.mode_shapes = modal_data.mode_shapes
        if isinstance(modal_data, VarPreGERSSI):
            self.std_frequencies = modal_data.std_frequencies
            self.std_damping = modal_data.std_damping
        else:
            self.std_frequencies = None
            self.std_damping = None
        self.select_modes = stabil_calc.select_modes
        self.setup_name = modal_data.setup_name
        self.start_time = modal_data.start_time

    def _apply_single_attrs(self, modal_data, stabil_calc):
        '''Populate instance attributes for the single-setup case.'''
        prep_signals = modal_data.prep_signals
        self.chan_dofs = prep_signals.chan_dofs
        self.num_channels = prep_signals.num_analised_channels
        self.modal_frequencies = modal_data.modal_frequencies
        self.modal_damping = modal_data.modal_damping
        self.mode_shapes = modal_data.mode_shapes
        if isinstance(modal_data, (VarSSIRef, VarPLSCF)):
            self.std_frequencies = modal_data.std_frequencies
            self.std_damping = modal_data.std_damping
        else:
            self.std_frequencies = None
            self.std_damping = None
        self.select_modes = stabil_calc.select_modes
        self.setup_name = modal_data.setup_name
        self.start_time = modal_data.start_time

    def _apply_empty_attrs(self, prep_signals):
        '''Populate instance attributes when no modal data is available.'''
        if prep_signals is not None:
            self.chan_dofs = prep_signals.chan_dofs
            self.num_channels = prep_signals.num_analised_channels
        else:
            self.chan_dofs = []
            self.num_channels = 0
        self.modal_frequencies = np.array([[]])
        self.modal_damping = np.array([[]])
        self.mode_shapes = np.array([[[]]])
        self.select_modes = []
        self.setup_name = ''
        self.start_time = None

    # ── State initialisation ──────────────────────────────────────────────────

    def _init_state(self):
        '''Initialise the backend-agnostic visibility/animation flags.

        Subclasses extend this to allocate their rendering-object containers.
        '''
        # bool objects
        self.show_nodes = True
        self.show_lines = True
        self.show_nd_lines = True
        self.show_cn_lines = True
        self.show_traces = True
        self.show_parent_childs = True
        self.show_chan_dofs = True
        self.show_axis = True
        self.animated = False
        self.data_animated = False
        self.seq_num = 0

    # ── Geometry-derived state ────────────────────────────────────────────────

    def _compute_node_bounds(self):
        '''Compute axis-aligned bounding box of all nodes in geometry_data.

        Returns
        -------
        xmin, xmax, ymin, ymax, zmin, zmax : float
            Equal-side bounding-cube limits centred on the node cloud.
        '''
        nodes = list(self.geometry_data.nodes.values())
        if not nodes:
            return -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

        coords = np.array(nodes, dtype=float)
        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)

        xrang = xmax - xmin
        xmed = xmax - xrang / 2
        yrang = ymax - ymin
        ymed = ymax - yrang / 2
        zrang = zmax - zmin
        zmed = zmax - zrang / 2

        rang = max(xrang, yrang, zrang)

        xmin, xmax = xmed - rang / 2, xmed + rang / 2
        ymin, ymax = ymed - rang / 2, ymed + rang / 2
        zmin, zmax = zmed - rang / 2, zmed + rang / 2
        return xmin, xmax, ymin, ymax, zmin, zmax

    # ── Mode selection ────────────────────────────────────────────────────────

    def change_mode(self, frequency=None, index=None, mode_index=None,):
        '''
        If the user selects a new mode: plots the mode shape
        and returns modal values e.g. to a GUI caller.

        Parameters
        ----------
            frequency: float,optional
                A search for the closest frequency in the list of already
                selected indices (self.selected_indices) is performed
            index: integer, optional
                Alternatively, the index of the wanted mode can be directly given
            mode_index: integer, optional
                The number of the mode in the list of currently selected modes

        Returns
        -------
            order_index: integer
                Model order of the selected mode
            mode_index: integer
                Index of the selected mode at model order
            frequency: float
                natural frequency of the selected mode
            damping: float
                damping ratio of the selected mode
            MPC: float, optional
                Modal phase colinearity of the selected mode,
                if available from an instance of StabilDiagram.StabilCalc1
            MP: float, optional
                Mean phase of the selected mode,
                if available from an instance of StabilDiagram.StabilCalc1
            MPD: float, optional
                Mean phase deviation of the selected mode,
                if available from an instance of StabilDiagram.StabilCalc1

        '''
        # mode numbering starts at 1 python lists start at 0
        mode_index = self._lookup_mode_index(
            frequency=frequency, index=index, mode_index=mode_index)

        frequency = self.modal_frequencies[mode_index[0], mode_index[1]]
        damping = self.modal_damping[mode_index[0], mode_index[1]]
        MPC, MP, MPD = self._get_stabil_params(mode_index)
        self.mode_index = mode_index

        if self.save_ani_path:
            cwd = self.save_ani_path / f'{self.select_modes.index(self.mode_index)}/'
            if not os.path.exists(cwd):
                os.makedirs(cwd)

        self._render_mode()

        if self.callback_fun is not None:
            try:
                self.callback_fun(self, mode_index)
            except Exception as e:
                logger.warning(repr(e))

        # order, mode_num,....
        return mode_index[1], mode_index[0], frequency, damping, MPC, MP, MPD

    def _lookup_mode_index(self, frequency=None, index=None, mode_index=None):
        '''Resolve *frequency*, *index*, or *mode_index* to a concrete mode index.

        Parameters
        ----------
        frequency : float or None
            If given, the closest frequency in the selected modes is found.
        index : int or None
            Position in ``self.select_modes``.
        mode_index : tuple or None
            Direct ``(order, mode)`` index.

        Returns
        -------
        mode_index : tuple
            Resolved ``(order, mode)`` index.
        '''
        selected_indices = self.select_modes
        if frequency is not None:
            freqs = np.array([self.modal_frequencies[idx[0], idx[1]]
                              for idx in selected_indices])
            index = int(np.argmin(abs(freqs - frequency)))
        if index is not None:
            mode_index = selected_indices[index]
        if mode_index is None:
            raise RuntimeError('No arguments provided!')
        return mode_index

    def _get_stabil_params(self, mode_index):
        '''Return MPC, MP, MPD for *mode_index* from stabil_calc (or Nones).

        Parameters
        ----------
        mode_index : tuple
            ``(order, mode)`` index.

        Returns
        -------
        MPC, MP, MPD : float or None
        '''
        if self.stabil_calc:
            MPC = self.stabil_calc.MPC_matrix[mode_index[0], mode_index[1]]
            MP = self.stabil_calc.MP_matrix[mode_index[0], mode_index[1]]
            MPD = self.stabil_calc.MPD_matrix[mode_index[0], mode_index[1]]
        else:
            MPC, MP, MPD = None, None, None
        return MPC, MP, MPD

    def get_frequencies(self):
        '''
        Returns
        -------
            frequencies: list
                Identified frequencies of all currently selected modes.
        '''
        selected_indices = self.select_modes

        frequencies = sorted([self.modal_frequencies[index[0], index[1]]
                              for index in selected_indices])
        return frequencies

    def change_amplitude(self, amplitude=None):
        '''
        Changes the amplitude of the mode shape, and redraws the
        modeshapes based on this amplitude.

        Parameters
        ----------
            amplitude: float, optional
        '''
        if amplitude is None:
            return
        amplitude = float(amplitude)
        if amplitude == self.amplitude:
            return

        self.amplitude = amplitude

        if self.mode_shapes.shape[2]:
            self._render_amplitude()

    def change_part(self, b):
        '''
        Change, which part of the complex number modeshapes should be
        drawn and redraw the modeshapes

        Parameters
        ----------
            b: bool
                If b, draws the magnitude of the modal coordinated, else
                phase information is considered. Default: b = False

        '''
        if b == self.real:
            return

        self.real = b
        self._render_part()

    # ── Displacement computation ──────────────────────────────────────────────

    def compute_mode_displacements(self):
        '''Populate ``disp_nodes`` / ``phi_nodes`` for the current mode.

        Backend-agnostic core of ``draw_msh``: rescales the currently selected
        mode shape, resets the per-node displacement/phase tables and fills
        them from the channel-DOF assignments and the parent-child definitions.
        No rendering is performed.
        '''
        mode_shape = self.mode_shapes[:, self.mode_index[1], self.mode_index[0]]
        mode_shape = ModalBase.rescale_mode_shape(mode_shape)
        ampli = self.amplitude

        self.disp_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}
        self.phi_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}

        self._compute_chan_dof_displacements(mode_shape, ampli)
        self._compute_parent_child_displacements()

    def _disp_phase_mag(self, disp):
        '''Convert complex displacement *disp* to ``(phase, magnitude)``.

        The conversion respects the ``self.real`` flag: when *True* only the
        real part is used and phase is forced to zero.

        Parameters
        ----------
        disp : complex
            Complex modal displacement.

        Returns
        -------
        phase, mag : float
        '''
        if self.real:
            phase = np.angle(disp, True)
            mag = np.abs(disp)
            if phase < 0:
                phase += 180
                mag = -mag
            if 90 < phase < 270:
                mag = -mag
            phase = 0
        else:
            phase = np.angle(disp)
            mag = np.abs(disp)
        return phase, mag

    def _compute_chan_dof_displacements(self, mode_shape, ampli):
        '''Populate ``self.disp_nodes`` and ``self.phi_nodes`` from channel-DOF assignments.

        Handles three cases:
        * no sensor at a node  → skipped
        * exactly one sensor   → direction vector used directly
        * two or more sensors  → axis-aligned or least-squares transformation

        Parameters
        ----------
        mode_shape : ndarray, shape (n_channels,)
            Scaled modal displacement vector.
        ampli : float
            Amplitude scaling factor.
        '''
        chan_found = [False] * len(mode_shape)

        for node in self.geometry_data.nodes.keys():
            this_chan_dofs = []
            for chan_dof in self.chan_dofs:
                chan, node_, az, elev, _chan_name = chan_dof[0:4] + chan_dof[-1:]
                if node_ == node:
                    disp = mode_shape[chan]
                    x, y, z = calc_xyz(az * np.pi / 180, elev * np.pi / 180, r=1)
                    this_chan_dofs.append([chan, x, y, z, disp])
                    chan_found[chan] = True

            if not this_chan_dofs:
                continue

            if len(this_chan_dofs) == 1:
                self._assign_single_sensor_disp(node, this_chan_dofs[0], ampli)
            else:
                self._assign_multi_sensor_disp(node, this_chan_dofs, ampli)

        for chan, found in enumerate(chan_found):
            if not found:
                logging.warning(
                    f'Could not find channel - DOF assignment for channel {chan}!')

    def _assign_single_sensor_disp(self, node, chan_dof_entry, ampli):
        '''Assign displacement/phase for a node with a single sensor.

        Parameters
        ----------
        node : int
            Node key in ``self.geometry_data.nodes``.
        chan_dof_entry : list
            ``[chan, x, y, z, disp]``
        ampli : float
            Amplitude scaling factor.
        '''
        _chan, x, y, z, disp = chan_dof_entry
        phase, mag = self._disp_phase_mag(disp)
        for axis_idx, direction in enumerate([x, y, z]):
            self.phi_nodes[node][axis_idx] = phase
            self.disp_nodes[node][axis_idx] = direction * mag * ampli

    def _assign_axis_aligned_disp(self, node, this_chan_dofs, ampli):
        '''Assign displacement for axis-aligned sensors at *node*.

        Parameters
        ----------
        node : int
            Node key in ``self.geometry_data.nodes``.
        this_chan_dofs : list of [chan, x, y, z, disp]
            Sensor entries for this node (each sensor on one axis only).
        ampli : float
            Amplitude scaling factor.
        '''
        for _chan, x, y, z, disp in this_chan_dofs:
            phase, mag = self._disp_phase_mag(disp)
            if not np.isclose(x, 0):
                self.phi_nodes[node][0] = phase
                self.disp_nodes[node][0] = x * mag * ampli
            elif not np.isclose(y, 0):
                self.phi_nodes[node][1] = phase
                self.disp_nodes[node][1] = y * mag * ampli
            elif not np.isclose(z, 0):
                self.phi_nodes[node][2] = phase
                self.disp_nodes[node][2] = z * mag * ampli

    def _assign_multi_sensor_disp(self, node, this_chan_dofs, ampli):
        '''Assign displacement/phase for a node with two or more sensors.

        Uses axis-aligned decomposition when all sensors lie along coordinate
        axes, and least-squares otherwise.

        Parameters
        ----------
        node : int
            Node key in ``self.geometry_data.nodes``.
        this_chan_dofs : list of [chan, x, y, z, disp]
            All sensor entries for this node.
        ampli : float
            Amplitude scaling factor.
        '''
        dirs = np.array([[x, y, z] for _, x, y, z, _ in this_chan_dofs])
        active_per_axis = (~np.isclose(dirs, 0)).sum(axis=0)

        if active_per_axis[0] <= 1 and active_per_axis[1] <= 1 and active_per_axis[2] <= 1:
            self._assign_axis_aligned_disp(node, this_chan_dofs, ampli)
        else:
            self._assign_lstsq_sensor_disp(node, this_chan_dofs, ampli)

    def _assign_lstsq_sensor_disp(self, node, this_chan_dofs, ampli):
        '''Assign displacement via least-squares coordinate transformation.

        Used when sensors at *node* are not purely axis-aligned.

        Parameters
        ----------
        node : int
            Node key.
        this_chan_dofs : list of [chan, x, y, z, disp]
            Sensor entries for this node.
        ampli : float
            Amplitude scaling factor.
        '''
        num_sensors = max(len(this_chan_dofs), 3)
        normal_matrix = np.zeros((num_sensors, 3))
        disp_vec = np.zeros(num_sensors, dtype=complex)
        last_i = 0
        for i, (_chan, x, y, z, disp) in enumerate(this_chan_dofs):
            normal_matrix[i, :] = [x, y, z]
            disp_vec[i] = disp
            last_i = i

        if last_i == 1:
            logging.info(
                f'Not enough sensors for a full 3D transformation at node {node}, '
                'will complement vectors with a zero displacement assumption '
                'in orthogonal direction.')
            c = np.cross(normal_matrix[0, :], normal_matrix[1, :])
            c /= np.linalg.norm(c)
            normal_matrix[2, :] = c

        q_res = np.linalg.lstsq(normal_matrix, disp_vec, rcond=None)[0]
        for axis_idx in range(3):
            phase, mag = self._disp_phase_mag(q_res[axis_idx])
            self.phi_nodes[node][axis_idx] = phase
            self.disp_nodes[node][axis_idx] = mag * ampli

    def _compute_parent_child_displacements(self):
        '''Apply parent-child DOF propagation to ``self.disp_nodes`` and ``self.phi_nodes``.

        For each parent-child pair stored in ``self.geometry_data.parent_childs``,
        the parent node's displacement is projected onto the child DOF directions.
        '''
        for i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl in self.geometry_data.parent_childs:
            if (x_m > 0) + (y_m > 0) + (z_m > 0) > 1:
                logging.warning(
                    'parent DOF includes more than one cartesian direction. '
                    'Phase angles will be distorted.')

            parent_disp = (self.disp_nodes[i_m][0] * x_m +
                           self.disp_nodes[i_m][1] * y_m +
                           self.disp_nodes[i_m][2] * z_m)
            parent_phase = (self.phi_nodes[i_m][0] * x_m +
                            self.phi_nodes[i_m][1] * y_m +
                            self.phi_nodes[i_m][2] * z_m)

            self._propagate_child_dof(i_sl, 0, x_sl, parent_disp, parent_phase, 'x')
            self._propagate_child_dof(i_sl, 1, y_sl, parent_disp, parent_phase, 'y')
            self._propagate_child_dof(i_sl, 2, z_sl, parent_disp, parent_phase, 'z')

    def _propagate_child_dof(self, i_sl, axis_idx, scale, parent_disp, parent_phase, axis_name):
        '''Propagate a single parent displacement component to a child DOF axis.

        Parameters
        ----------
        i_sl : int
            Child node index.
        axis_idx : int
            0, 1, or 2 for X, Y, Z.
        scale : float
            Child-DOF scale factor in this axis direction.
        parent_disp : float
            Projected parent displacement magnitude.
        parent_phase : float
            Projected parent phase.
        axis_name : str
            Axis label for warning messages.
        '''
        if np.allclose(scale, 0):
            return
        if self.disp_nodes[i_sl][axis_idx] > 0:
            logging.warning(
                f'A modal coordinate of {self.disp_nodes[i_sl][axis_idx]} has already '
                f'been assigned to this DOF {axis_name} of node {i_sl}. Overwriting!')
        self.phi_nodes[i_sl][axis_idx] = parent_phase
        self.disp_nodes[i_sl][axis_idx] += parent_disp * scale

    def _compute_data_disp_nodes(self, num):
        """Accumulate sensor displacements into a per-node dict for frame *num*."""
        disp_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}
        for chan_dof in self.chan_dofs:
            chan_, node, az, elev = chan_dof[0:4]
            if node is None or node not in self.geometry_data.nodes:
                continue
            x, y, z = calc_xyz(az * np.pi / 180, elev * np.pi / 180)
            sig = self.prep_signals.signals_filtered[num, chan_] * self.amplitude
            disp_nodes[node][0] += sig * x
            disp_nodes[node][1] += sig * y
            disp_nodes[node][2] += sig * z
        return disp_nodes

    # ── Rendering template-method hooks ───────────────────────────────────────

    def _render_mode(self):
        '''Backend hook: render the currently selected mode.  No-op on the base.'''

    def _render_amplitude(self):
        '''Backend hook: re-render after an amplitude change.  No-op on the base.'''

    def _render_part(self):
        '''Backend hook: re-render after a real/complex-part change.  No-op on the base.'''

    # ── Backend/GUI contract ──────────────────────────────────────────────────
    #
    # These let a GUI drive any backend without reaching into matplotlib
    # internals such as ``fig``, ``subplot`` or ``mpl_connect``.

    #: Whether this backend exposes per-axis 3-D limits that a GUI may read
    #: and set.  Matplotlib's Axes3D does; a VTK camera does not, so those
    #: backends fall back to their own camera reset.
    supports_axis_limits = False

    #: Whether this backend can animate the recorded time histories via
    #: ``filter_and_animate_data``.  Only the matplotlib backend does.
    supports_data_animation = False

    #: Whether the geometry and channel-DOF editors can pick artists off
    #: this backend.  They reach into an ``Axes3D`` directly, so only the
    #: matplotlib backend qualifies; see :func:`require_picking_backend`.
    supports_picking = False

    @property
    def widget(self):
        '''Backend widget hosting the rendering.  Overridden by concrete backends.'''
        raise NotImplementedError(
            'ModeShapeBase provides no widget; use a concrete rendering backend.')

    def attach_qt_canvas(self, placeholder=None):
        '''Return the widget a Qt GUI should place in its layout.

        Backends that need to take over an existing canvas (matplotlib
        draws into a ``FigureCanvas`` supplied by the GUI) override this
        and return *placeholder*; backends that bring their own widget
        return that instead, and the caller swaps it in.

        Parameters
        ----------
        placeholder : QWidget, optional
            The widget currently occupying the layout slot.

        Returns
        -------
        QWidget
            The widget to display.
        '''
        return self.widget

    def connect_view_change(self, callback):
        '''Register *callback* to run after the user moves the camera.

        Returns
        -------
        list
            Connection handles, empty when the backend has nothing to
            connect (VTK interactors handle the camera themselves).
        '''
        return []

    def get_view_angles(self):
        '''Return ``(elev, azim, roll)`` in degrees, or *None* if unsupported.'''
        return None

    def get_view_limits(self):
        '''Return ``(xmin, xmax, ymin, ymax, zmin, zmax)``, or *None*.'''
        return None

    def set_view_limits(self, xmin, xmax, ymin, ymax, zmin, zmax):
        '''Set the per-axis 3-D limits.  No-op when unsupported.'''
