# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""3-D mode-shape visualisation (ModeShapePlot) for pyOMA results."""

# system i/o
import dataclasses
import warnings
import collections
import matplotlib.animation
import matplotlib.patches
import mpl_toolkits.mplot3d.axes3d
from .PostProcessingTools import MergePoSER
from .VarSSIRef import VarSSIRef
from .SSICovRef import PogerSSICovRef
from .ModalBase import ModalBase
from .PreProcessingTools import PreProcessSignals, GeometryProcessor
from .StabilDiagram import StabilCalc
from .Helpers import calc_xyz, nearly_equal
import itertools
from pathlib import Path

import numpy as np
import matplotlib.markers
import matplotlib.colors
import matplotlib.figure
import matplotlib.backend_bases
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Matplotlib
# check if python is running in headless mode i.e. as a server script
# if 'DISPLAY' in os.environ:
#     matplotlib.use("Qt5Agg", force=True)

# Numpy

# project

NoneType = type(None)


@dataclasses.dataclass
class ModeShapePlotConfig:
    """Visual style configuration for :class:`ModeShapePlot`.

    Group visual-style keyword arguments so that :class:`ModeShapePlot`
    can be constructed with a single *config* object instead of many
    individual keyword parameters.

    Parameters
    ----------
    beamcolor : matplotlib color or sequence, optional
        Color used to draw beam/line elements.
    beamstyle : str or sequence, optional
        Linestyle used to draw beam/line elements.
    nodecolor : matplotlib color, optional
        Color used to draw nodes.
    nodemarker : matplotlib marker, optional
        Marker symbol used to draw nodes.
    nodesize : float, optional
        Marker size for nodes.
    dpi : int, optional
        Figure resolution in dots per inch.
    amplitude : float, optional
        Scaling factor for modal displacement amplitudes.
    linewidth : float or sequence, optional
        Line width for beam/line elements.
    callback_fun : callable or None, optional
        Called after each mode change; signature ``f(plot, mode_index)``.
    real : bool, optional
        When *True*, plot the real part of complex mode shapes.
    scale : float, optional
        Fractional scale for axis arrows and channel-DOF arrows.
    save_ani_path : pathlib.Path or None, optional
        Directory in which animation frames are saved.
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


#: Named tuple grouping a node index together with its three DOF scale factors.
NodeCoords = collections.namedtuple('NodeCoords', ['node_index', 'x', 'y', 'z'])

#: Named tuple for specifying arrow start/end/length in parent-child matching.
_ArrowSpec = collections.namedtuple('_ArrowSpec', ['x_s', 'y_s', 'z_s', 'x_e', 'y_e', 'z_e', 'length'])


class ModeShapePlot(object):
    """3-D mode-shape visualisation for pyOMA modal analysis results.

    Renders a structural geometry (nodes, beams, parent-child relations) and
    superimposes animated or static mode-shape deformations.  Supports single-
    setup analyses as well as PoGER and PoSER multi-setup results.

    Attributes
    ----------
    amplitude : float
        Scaling factor applied to modal displacement amplitudes when drawing
        deformed shapes.  Increase to make the deformation more visible.

    Notes
    -----
    The object accepts different combinations of inputs depending on the
    merging strategy (single-setup, PoGER/PreGER, PoSER):

    ==================  ==============  ============  =============
    Variable            single-setup    PoGER/PreGER  PoSER merging
    ==================  ==============  ============  =============
    modal_freq./damp.   modal_data      modal_data    merged_data
    mode shapes         modal_data      modal_data    merged_data
    num_channels        prep_signals    modal_data    merged_data
    chan_dofs           prep_signals    modal_data    merged_data
    select_modes        stabil_calc     stabil_calc   merged_data
    ==================  ==============  ============  =============

    .. TODO::
         * clean up animation methods
         * remove "real modeshape" functionality as it might mislead inexperienced users
         * Fix parent-childs assignment: allow multiple channel averaging into
           a single child displacement, then transform to polar coordinates
         * Implement the plotting in pyvista for better 3D graphics
    """

    def __init__(self, geometry_data, stabil_calc=None, modal_data=None,
                 prep_signals=None, merged_data=None, config=None, **kwargs):
        '''
        Initializes the class object and automatically checks which of
        the merging use cases applies.

        See class docstring for the merging-routine table.

        Parameters
        ----------
            geometry_data : PreProcessingTools.GeometryProcessor, required
                    Object containing all the necessary geometry information.

            stabil_calc : StabilDiagram.StabilCalc, optional
                    Object containing the information, which modes were
                    selected from modal_data.

            modal_data : ModalBase.ModalBase, optional
                    Object of one the classes derived from ModalBase.ModalBase,
                    containing the estimated modal parameters at multiple
                    model orders.

            prep_signals : PreProcessingTools.PreProcessSignals, optional
                    Object containing the signals data and information
                    about it.

            merged_data : PostProcessingTools.MergePoSER, optional
                    Object containing the merged data.

            config : ModeShapePlotConfig, optional
                    Visual-style configuration object.  Preferred over passing
                    individual style keyword arguments.

            **kwargs :
                    Accepts ``selected_mode``, ``fig``, and deprecated
                    individual style params (amplitude, real, scale, dpi,
                    nodecolor, nodemarker, nodesize, beamcolor, beamstyle,
                    linewidth, callback_fun, save_ani_path).
        '''
        kwargs.pop('selected_mode', None)  # accepted but not used
        fig = kwargs.pop('fig', None)
        config, fig = self._resolve_config(config, fig, kwargs)

        if not isinstance(geometry_data, GeometryProcessor):
            raise TypeError(
                f"Expected GeometryProcessor for 'geometry_data', "
                f"got {type(geometry_data).__name__!r}.")
        self.geometry_data = geometry_data

        self._validate_data_types(stabil_calc, modal_data, prep_signals, merged_data)
        self.stabil_calc = stabil_calc
        self.modal_data = modal_data
        self.prep_signals = prep_signals
        self.merged_data = merged_data

        self._detect_and_apply_merging(merged_data, modal_data, prep_signals, stabil_calc)

        self.disp_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}
        self.phi_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}

        self._apply_config(config)
        self._init_state()
        self._setup_figure(fig)

        if not self.select_modes:
            self.mode_index = None
        else:
            self.mode_index = self.select_modes[0]

    def _resolve_config(self, config, fig, kwargs):
        '''Handle config vs. legacy individual style parameters.

        Pops legacy style keys from *kwargs*, builds a
        :class:`ModeShapePlotConfig` when needed, and returns
        ``(config, fig)``.

        Parameters
        ----------
        config : ModeShapePlotConfig or None
        fig : matplotlib.figure.Figure or None
        kwargs : dict
            Remaining keyword arguments (modified in-place).

        Returns
        -------
        config : ModeShapePlotConfig
        fig : matplotlib.figure.Figure or None
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
        if isinstance(modal_data, PogerSSICovRef):
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
        if isinstance(modal_data, VarSSIRef):
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

    def _init_state(self):
        '''Initialise visibility flags and empty plot-object containers.'''
        # bool objects
        self.show_nodes = True
        self.show_lines = True
        self.show_nd_lines = True
        self.show_cn_lines = True
        self.show_parent_childs = True
        self.show_chan_dofs = True
        self.show_axis = True
        self.animated = False
        self.data_animated = False
        # plot objects
        self.patches_objects = {}
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.trace_objects = []
        self.axis_obj = {}
        self.seq_num = 0

    def _setup_figure(self, fig):
        '''Create or validate the matplotlib figure and 3-D subplot.

        Parameters
        ----------
        fig : matplotlib.figure.Figure or None
            When *None*, a new figure is created.
        '''
        if fig is None:
            fig = matplotlib.figure.Figure(dpi=self.dpi, facecolor='#ffffff00')
            # remove all whitespace around the axes
            fig.subplots_adjust(0, 0, 1, 1, 0, 0)
            matplotlib.backend_bases.FigureCanvasBase(fig)
        else:
            if not isinstance(fig, matplotlib.figure.Figure):
                raise TypeError(
                    f"Expected matplotlib.figure.Figure for 'fig', "
                    f"got {type(fig).__name__!r}.")
        self.fig = fig
        self.subplot = fig.subplots(
            subplot_kw=dict(projection='3d', anchor='C',
                            fc='#ffffff00', box_aspect=(1, 1, 1)))
        self.subplot.set_aspect('equal', 'datalim')
        # nasty hack to disable clipping
        self.subplot.patch = fig.patch
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        self.subplot.grid(False)
        self.subplot.set_axis_off()

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
    def _check_color(name, val):
        '''Validate a matplotlib color; raise ValueError otherwise.'''
        if not matplotlib.colors.is_color_like(val):
            raise ValueError(f"Invalid color for {name!r}: {val!r}.")
        return val

    @staticmethod
    def _check_color_or_seq(name, val):
        '''Validate a matplotlib color OR list/tuple/ndarray.'''
        if (not matplotlib.colors.is_color_like(val)
                and not isinstance(val, (list, tuple, np.ndarray))):
            raise ValueError(
                f"{name!r} must be a valid matplotlib color or a "
                f"list/tuple/ndarray, got {val!r}.")
        return val

    @staticmethod
    def _check_linestyle_or_seq(name, val, valid_styles):
        '''Validate a matplotlib linestyle string or sequence.'''
        if val not in valid_styles and not isinstance(val, (list, tuple, np.ndarray)):
            raise ValueError(
                f"{name!r} must be a valid matplotlib linestyle or a "
                f"list/tuple/ndarray, got {val!r}.")
        return val

    @staticmethod
    def _check_marker(name, val, valid_markers):
        '''Validate a matplotlib marker or 3-tuple.'''
        if val not in valid_markers and not (isinstance(val, tuple) and len(val) == 3):
            raise ValueError(
                f"{name!r} must be a valid matplotlib marker or a 3-tuple, "
                f"got {val!r}.")
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

    def _apply_config(self, config):
        '''Validate and apply a :class:`ModeShapePlotConfig` to ``self``.

        Parameters
        ----------
        config : ModeShapePlotConfig
            Configuration object whose fields are validated and stored as
            instance attributes.
        '''
        styles = ['-', '--', '-.', ':', 'None', ' ', '', None]
        markers = list(matplotlib.markers.MarkerStyle.markers.keys())
        self.real = self._check_bool('real', config.real)
        self.scale = self._check_numeric('scale', config.scale)
        self.beamcolor = self._check_color_or_seq('beamcolor', config.beamcolor)
        self.beamstyle = self._check_linestyle_or_seq('beamstyle', config.beamstyle, styles)
        self.nodecolor = self._check_color('nodecolor', config.nodecolor)
        self.nodemarker = self._check_marker('nodemarker', config.nodemarker, markers)
        self.nodesize = self._check_numeric('nodesize', config.nodesize)
        self.dpi = self._check_int('dpi', config.dpi)
        self.amplitude = self._check_numeric('amplitude', config.amplitude)
        self.linewidth = self._check_numeric_or_seq('linewidth', config.linewidth)
        self.callback_fun = self._check_callable_or_none('callback_fun', config.callback_fun)
        self.save_ani_path = self._check_path_or_none('save_ani_path', config.save_ani_path)

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

    def reset_view(self):
        '''
         * restore viewport
         * restore axis' limits
         * reset displacements values for all nodes
        '''
        self.stop_ani()
        # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = persp_transformation
        self.subplot.view_init(30, -60)
        self.subplot.autoscale_view()

        xmin, xmax, ymin, ymax, zmin, zmax = self._compute_node_bounds()
        self.subplot.set_xlim3d(xmin, xmax)
        self.subplot.set_ylim3d(ymin, ymax)
        self.subplot.set_zlim3d(zmin, zmax)

        self.draw_nodes()
        self.draw_lines()
        self.draw_chan_dofs()
        self.draw_parent_childs()
        self.draw_axis()
        if self.mode_index is not None:
            self.draw_msh()
        self.set_equal_aspect()

        self.fig.canvas.draw()


    # Lookup table: named viewport -> (azim, elev, proj_type)
    _NAMED_VIEWPORTS = {
        'X':   (0,   0,  'ortho'),
        'Y':   (-90, 0,  'ortho'),
        'Z':   (0,   90, 'ortho'),
        'ISO': (-60, 30, 'persp'),
    }

    def _setup_viewport_angles(self, viewport):
        '''Resolve *viewport* to ``(elev, azim, roll)`` and set projection type.

        Parameters
        ----------
        viewport : str or sequence
            A named viewport key (``'X'``, ``'Y'``, ``'Z'``, ``'ISO'``), or a
            ``(elev, azim, roll)`` sequence.

        Returns
        -------
        elev, azim, roll : float or None
        '''
        roll = None
        if isinstance(viewport, (list, tuple)):
            elev, azim, roll = viewport
            return elev, azim, roll

        entry = self._NAMED_VIEWPORTS.get(viewport)
        if entry is not None:
            azim, elev, proj_type = entry
            self.subplot.set_proj_type(proj_type)
        else:
            logger.warning(f'viewport not recognized: {viewport}')
            azim, elev = -60, 30
            self.subplot.set_proj_type('persp')
        return elev, azim, roll

    def change_viewport(self, viewport=None):
        '''
         Change the viewport e.g. azimuth and elevation and refresh the canvas

         Parameters
         ----------
             viewport: {'X', 'Y', 'Z', 'ISO'\\, optional
                 The viewport to set.
        '''
        elev, azim, roll = self._setup_viewport_angles(viewport)
        self.subplot.view_init(elev, azim, roll)
        self.fig.canvas.draw()

        if self.animated or self.data_animated:
            for line in self.lines_objects:
                line.set_visible(False)
            for line in self.nd_lines_objects:
                line.set_visible(False)
            for line in self.cn_lines_objects.values():
                line.set_visible(False)
            self.line_ani._setup_blit()

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

        self.draw_msh()

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
            self.draw_msh()

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
        self.draw_msh()

    def save_plot(self, path=None):
        '''
        Save the curently displayed frame as a graphics file

        Parameters
        ----------
            path: str (valid filepath), optional
                The full path, including the extension, where to save
                the graphic.
        '''

        if path:
            self.fig.canvas.print_figure(path, dpi=self.dpi)


    def add_node(self, x, y, z, i):
        '''
        Adds a node to the internal node table and initializes zero-value
        displacements for this node to the internal displacements table.
        Draws a single point at the coordinates and annotates it with
        its number. Stores the two plot objects in a table and removes
        any objects that might be in the table at the desired place
        to avoid duplicate nodes.

        Parameters
        ----------
            x,y,z: float
                3D-coordinates of the node
            i: integer
                Index of the node, must be previously determined
        '''
        # leave present value if there is any else put 0
        self.disp_nodes[i] = self.disp_nodes.get(i, [0, 0, 0])

        x, y, z = x + self.disp_nodes[i][0], y + self.disp_nodes[i][1], z + \
            self.disp_nodes[i][2]  # draw displaced nodes

        patch = self.subplot.scatter(
            x,
            y,
            z,
            color=self.nodecolor,
            marker=self.nodemarker,
            s=self.nodesize,
            visible=self.show_nodes)

        text = self.subplot.text(x, y, z, i, visible=self.show_nodes)

        if self.patches_objects.get(i) is not None:
            if isinstance(self.patches_objects[i], (tuple, list)):
                for obj in self.patches_objects[i]:
                    try:
                        obj.remove()
                    except BaseException:
                        pass

        self.patches_objects[i] = (patch, text)

        self.fig.canvas.draw_idle()

    def _resolve_beam_style(self, i):
        '''Return per-element beam style attributes for line index *i*.

        When a style attribute is a sequence, the element at position *i* is
        returned; otherwise the scalar attribute is returned unchanged.

        Parameters
        ----------
        i : int
            Index of the line in the lines table.

        Returns
        -------
        beamcolor, beamstyle, linewidth
        '''
        beamcolor = (self.beamcolor[i]
                     if isinstance(self.beamcolor, (list, tuple, np.ndarray))
                     else self.beamcolor)
        beamstyle = (self.beamstyle[i]
                     if isinstance(self.beamstyle, (list, tuple, np.ndarray))
                     else self.beamstyle)
        linewidth = (self.linewidth[i]
                     if isinstance(self.linewidth, (list, tuple, np.ndarray))
                     else self.linewidth)
        return beamcolor, beamstyle, linewidth

    def add_line(self, line, i):
        '''
        Add a line by adding the start node and end node to the internal
        line table and draws that line between the two nodes. Stores the
        line object in a table and removes any objects that might be in
        the table at the desired place, i.e. avoid duplicate lines

        Parameters
        ----------
            line: 2-tuple of integer
                The indices of the start- and end-node of the line
            i: integer
                Index of the line, must be previously determined

        '''
        beamcolor, beamstyle, linewidth = self._resolve_beam_style(i)

        line_object = self.subplot.plot(
            [self.geometry_data.nodes[node][0]
             +self.disp_nodes[node][0] for node in line],
            [self.geometry_data.nodes[node][1]
             +self.disp_nodes[node][1] for node in line],
            [self.geometry_data.nodes[node][2]
             +self.disp_nodes[node][2] for node in line],
            color=beamcolor,
            linestyle=beamstyle,
            visible=self.show_lines,
            linewidth=linewidth)[0]

        while len(self.lines_objects) < i + 1:
            self.lines_objects.append(None)
        if self.lines_objects[i] is not None:
            try:
                self.lines_objects[i].remove()
            except ValueError:
                pass
        self.lines_objects[i] = line_object

        self.fig.canvas.draw_idle()

    def add_nd_line(self, line, i):
        '''
        Add a non-displaced line, which acts as a mesh-reference for the
        displaced lines. Works analogously to self.add_line

        Parameters
        ----------
            line: 2-tuple of integer
                The indices of the start- and end-node of the line
            i: integer
                Index of the line, must be previously determined

        '''
        beamcolor, _, _ = self._resolve_beam_style(i)
        beamstyle = 'dotted'

        line_object = self.subplot.plot(
            [self.geometry_data.nodes[node][0] for node in line],
            [self.geometry_data.nodes[node][1] for node in line],
            [self.geometry_data.nodes[node][2] for node in line],
            color=beamcolor,
            linestyle=beamstyle,
            linewidth=1,
            visible=self.show_lines)[0]

        while len(self.nd_lines_objects) < i + 1:
            self.nd_lines_objects.append(None)
        if self.nd_lines_objects[i] is not None:
            try:
                self.nd_lines_objects[i].remove()
            except ValueError:
                pass
                # del self.nd_lines_objects[i]
        self.nd_lines_objects[i] = line_object

        self.fig.canvas.draw_idle()

    def add_cn_line(self, i):
        '''
        Draws a line between the displaced and the undisplaced node.

        Parameters
        ----------
            i: integer
                Index of the node
        '''

        beamcolor = 'lightgray'

        beamstyle = 'dotted'
        node = self.geometry_data.nodes[i]
        disp_node = self.disp_nodes.get(node, [0, 0, 0])

        line_object = self.subplot.plot(
            [node[0], node[0] + disp_node[0]],
            [node[1], node[1] + disp_node[1]],
            [node[2], node[2] + disp_node[2]],
            color=beamcolor,
            linestyle=beamstyle,
            linewidth=1,
            visible=self.show_cn_lines)[0]

        if self.cn_lines_objects.get(i, None) is not None:
            try:
                self.cn_lines_objects[i].remove()
            except ValueError:
                pass
        self.cn_lines_objects[i] = line_object

        self.fig.canvas.draw_idle()


    def add_parent_child(self, *, i, parent, child):
        '''
        Takes parent-child definitions and adds these definitions to the
        internal parent-child table. Draws an arrow indicating the DOF
        at each node of parent and child. Arrows at equal positions and
        direction will be offset to avoid overlapping. Stores the two
        arrow objects in a table and removes any objects that might be
        in the table at the desired index i.e. avoid duplicate arrows.

        Parameters
        ----------
            i : int
                Table index for the plot objects.
            parent : NodeCoords
                Parent node coordinates (node_index, x, y, z).
            child : NodeCoords
                Child node coordinates (node_index, x, y, z).
        '''
        i_m, x_m, y_m, z_m = parent.node_index, parent.x, parent.y, parent.z
        i_sl, x_sl, y_sl, z_sl = child.node_index, child.x, child.y, child.z

        def offset_arrows(verts3d_new, all_arrows_list):
            '''
            avoid overlapping arrows as they are hard to distinguish
            therefore loop through all arrow object and compare their
            coordinates and directions (but ignore length) with the
            arrow to be newly created if there is an overlapping then
            offset the coordinates of the new arrow by 5 % of the
            length (hardcoded) in each direction (which should actually
            only be in the perpendicular plane)
            '''
            ((x_s, x_e), (y_s, y_e), (z_s, z_e)) = verts3d_new
            start_point = (x_s, y_s, z_s)
            length = x_e ** 2 + y_e ** 2 + z_e ** 2
            dir_norm = (x_e / length, y_e / length, z_e / length)
            while True:
                for arrow in itertools.chain.from_iterable(all_arrows_list):
                    (x, y, z, dx, dy, dz) = arrow._verts3d
                    (x_a, x_b) = x, x + dx
                    (y_a, y_b) = y, y + dy
                    (z_a, z_b) = z, z + dz
                    # (x_a, x_b), (y_a, y_b), (z_a, z_b) = arrow._verts3d
                    # transform from position vector to direction vector
                    x_c, y_c, z_c = (x_b - x_a), (y_b - y_a), (z_b - z_a)
                    this_start_point = (x_a, y_a, z_b)
                    this_length = x_c ** 2 + y_c ** 2 + z_c ** 2
                    if this_length == 0:
                        continue
                    this_dir_norm = (
                        x_c / this_length,
                        y_c / this_length,
                        z_c / this_length)
                    if start_point != this_start_point:  # starting point equal
                        continue
                    if this_dir_norm != dir_norm:  # direction equal
                        continue
                    # offset hardcoded
                    x_s, y_s, z_s = [
                        coord + 0.05 * this_length for coord in start_point]
                    # lazy offset, it should actually be in the plane
                    # perpendicular to the vector
                    start_point = (x_s, y_s, z_s)
                    length = x_e ** 2 + y_e ** 2 + z_e ** 2
                    dir_norm = (x_e / length, y_e / length, z_e / length)
                    break
                else:
                    break
            return ((x_s, x_e), (y_s, y_e), (z_s, z_e))

        color = "bgrcmyk"[int(np.fmod(i, 7))]  # equal colors for both arrows

        x_s, y_s, z_s = self.geometry_data.nodes[i_m]
        ((x_s, x_m), (y_s, y_m), (z_s, z_m)) = offset_arrows(
            ((x_s, x_m), (y_s, y_m), (z_s, z_m)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_m = LabeledArrow3D(x_s, y_s, z_s, x_m, y_m, z_m,
                                 mutation_scale=5, lw=1, arrowstyle="-|>",
                                 color=color, visible=self.show_parent_childs)
        arrow_m = self.subplot.add_artist(arrow_m)

        x_s, y_s, z_s = self.geometry_data.nodes[i_sl]
        ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)) = offset_arrows(
            ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_sl = LabeledArrow3D(x_s, y_s, z_s, x_sl, y_sl, z_sl,
                                  mutation_scale=5, lw=1, arrowstyle="-|>",
                                  color=color, visible=self.show_parent_childs)

        arrow_sl = self.subplot.add_artist(arrow_sl)

        while len(self.arrows_objects) < i + 1:
            self.arrows_objects.append(None)
        if self.arrows_objects[i] is not None:
            for obj in self.arrows_objects[i]:
                obj.remove()
        self.arrows_objects[i] = (arrow_m, arrow_sl)

        self.fig.canvas.draw_idle()


    def add_chan_dof(self, chan, node, az, elev, chan_name, i):
        '''
        Draws an arrow indicating a channel-DOF assignment. Annotates the
        arrow with the the channel name. Stores the two plot objects in a
        table and removes any objects that might be in the table at the
        desired index i.e. avoid duplicate arrows/texts.

        Parameters
        ----------
            chan: integer
                Index of the channel.
            node: integer
                Index of the node in the internal node table
            az, elev: float
                Azimuth and elevation of the DOF assignment
            chan_name: str
                Name of the channel to annotate
            i: integer
                Table index for the plot objects.

        .. TODO::
            * arrow lengths do not scale with the total dimension of the plot
        '''

        x_s, y_s, z_s = self.geometry_data.nodes[node]

        x_m, y_m, z_m = calc_xyz(
            az / 180 * np.pi, elev / 180 * np.pi, r=self.scale)

        # point the arrow towards the resulting direction
        arrow = LabeledArrow3D(x_s, y_s, z_s, x_m, y_m, z_m,
                               mutation_scale=5, lw=1, arrowstyle="-|>",
                               visible=self.show_chan_dofs)
        arrow = self.subplot.add_artist(arrow)

        arrow.add_label(chan_name, visible=self.show_chan_dofs)
        arrow.set_clip_path(None)

        while len(self.channels_objects) < i + 1:
            self.channels_objects.append(None)
        if self.channels_objects[i] is not None:
            self.channels_objects[i].remove()

        self.channels_objects[i] = arrow

        self.fig.canvas.draw_idle()

    def _find_and_remove_patch(self, x, y, z, node, d_x, d_y, d_z):
        '''Search ``self.patches_objects`` for a matching patch and remove it.

        Parameters
        ----------
        x, y, z : float
            Node coordinates (centre of the tolerance box).
        node : int
            Node index (searched first for fast lookup).
        d_x, d_y, d_z : float
            Absolute displacement tolerances.

        Returns
        -------
        bool
            *True* if a patch was found and removed, *False* otherwise.
        '''
        for j in [node] + list(range(max(len(self.patches_objects), node))):
            if self.patches_objects.get(j) is None:
                continue
            # ._offsets3d = ([x],[y],np.ndarray([z]))
            x_, y_, z_ = [float(val[0]) for val in self.patches_objects[j][0]._offsets3d]
            if x - d_x <= x_ <= x + d_x and y - d_y <= y_ <= y + d_y and z - d_z <= z_ <= z + d_z:
                for obj in self.patches_objects[j]:
                    obj.remove()
                del self.patches_objects[j]
                return True
        return False

    def take_node(self, x, y, z, node):
        '''
        Remove a node at given coordinates and all objects connected to
        this node first (there should not be any). Remove the patch
        objects from the plot and remove the coordinates from the node
        and displacement tables.

        Parameters
        ----------
            x,y,z: float
                Coordinates of the node
            node: integer
                Index of the node

        .. TODO::
            * Function presumably breaks in the second for loop, because
              geometry_data and the internal tables become out of sync.

        '''
        d_x, d_y, d_z = self.disp_nodes.get(node, [0, 0, 0])
        d_x, d_y, d_z = abs(d_x), abs(d_y), abs(d_z)

        if not self._find_and_remove_patch(x, y, z, node, d_x, d_y, d_z):
            if self.patches_objects:
                logging.warning('patches_object not found')

        for j in [node] + list(range(max(len(self.geometry_data.nodes), node))):
            if self.geometry_data.nodes.get(j) == [x, y, z]:
                del self.disp_nodes[j]
                break
        else:  # executed when for loop runs through without break
            if self.patches_objects:
                logging.warning('node not found')

        self.fig.canvas.draw_idle()

    @staticmethod
    def _endpoints_in_tolerance(actuals, centers, tolerances):
        '''Return True if every actual value lies within center ± tolerance.

        Parameters
        ----------
        actuals : sequence of float
            Observed coordinate values.
        centers : sequence of float
            Expected coordinate values.
        tolerances : sequence of float
            Absolute tolerance for each axis.

        Returns
        -------
        bool
        '''
        return all(c - t <= v <= c + t
                   for v, c, t in zip(actuals, centers, tolerances))

    @staticmethod
    def _line_coords_match(line_obj, start, end, disp_s, disp_e):
        '''Return True if *line_obj* matches the given (possibly displaced) endpoints.

        Both forward and reversed orientations are checked.

        Parameters
        ----------
        line_obj : matplotlib Line3D
            A line object whose ``_verts3d`` attribute is compared.
        start : tuple of float
            ``(x_s, y_s, z_s)`` start-node coordinates.
        end : tuple of float
            ``(x_e, y_e, z_e)`` end-node coordinates.
        disp_s : tuple of float
            ``(d_x_s, d_y_s, d_z_s)`` absolute displacement tolerances at start.
        disp_e : tuple of float
            ``(d_x_e, d_y_e, d_z_e)`` absolute displacement tolerances at end.

        Returns
        -------
        bool
        '''
        (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = line_obj._verts3d
        centers = (*start, *end)
        tols = (*disp_s, *disp_e)
        fwd = ModeShapePlot._endpoints_in_tolerance(
            (x_s_, y_s_, z_s_, x_e_, y_e_, z_e_), centers, tols)
        rev = ModeShapePlot._endpoints_in_tolerance(
            (x_e_, y_e_, z_e_, x_s_, y_s_, z_s_), centers, tols)
        return fwd or rev

    def _find_and_remove_displaced_line(self, objects_list, start, end,
                                        disp_s, disp_e, warn_name):
        '''Search *objects_list* for a matching line and remove it in-place.

        Parameters
        ----------
        objects_list : list
            List of line objects to search.
        start : tuple of float
            ``(x_s, y_s, z_s)`` start-node undisplaced coordinates.
        end : tuple of float
            ``(x_e, y_e, z_e)`` end-node undisplaced coordinates.
        disp_s : tuple of float
            ``(d_x_s, d_y_s, d_z_s)`` absolute displacement tolerances at start.
        disp_e : tuple of float
            ``(d_x_e, d_y_e, d_z_e)`` absolute displacement tolerances at end.
        warn_name : str
            Label used in the warning if no match is found.
        '''
        for j in range(len(objects_list)):
            if self._line_coords_match(objects_list[j], start, end, disp_s, disp_e):
                objects_list[j].remove()
                del objects_list[j]
                return
        if objects_list:
            logging.warning(f'{warn_name} not found')

    def take_line(self, line):
        '''
        Remove a line between to nodes. If the plot objects are already
        in their displaced state, the comparison between the actual
        coordinates and these objects have to account for  displacement
        by comparing to an interval of coordinates. Remove the non-displaced
        lines, too.

        Parameters
        ----------
            line: 2-tuple of integers
                Tuple containg the indices of the start- and end-nodes

        '''
        if not isinstance(line, (tuple, list)):
            raise TypeError(f"Expected tuple or list for 'line', got {type(line).__name__!r}.")
        if len(line) != 2:
            raise ValueError(f"Expected sequence of length 2 for 'line', got {len(line)}.")

        start = tuple(self.geometry_data.nodes[line[0]])
        end = tuple(self.geometry_data.nodes[line[1]])

        d_node_s = self.disp_nodes.get(line[0], [0, 0, 0])
        d_node_e = self.disp_nodes.get(line[1], [0, 0, 0])
        disp_s = (abs(d_node_s[0]), abs(d_node_s[1]), abs(d_node_s[2]))
        disp_e = (abs(d_node_e[0]), abs(d_node_e[1]), abs(d_node_e[2]))

        self._find_and_remove_displaced_line(
            self.lines_objects, start, end, disp_s, disp_e, 'line_object')
        self._find_and_remove_displaced_line(
            self.nd_lines_objects, start, end, disp_s, disp_e, 'nd_line_object')
        self.fig.canvas.draw_idle()

    def take_parent_child(self, *, parent, child):
        '''
        Remove the two arrows associated with the parent-child definition.

        Parameters
        ----------
            parent : NodeCoords
                Parent node coordinates (node_index, x, y, z).
            child : NodeCoords
                Child node coordinates (node_index, x, y, z).
        '''
        i_m, x_m, y_m, z_m = parent.node_index, parent.x, parent.y, parent.z
        i_sl, x_sl, y_sl, z_sl = child.node_index, child.x, child.y, child.z

        x_s_m, y_s_m, z_s_m = self.geometry_data.nodes[i_m]
        length_m = x_m ** 2 + y_m ** 2 + z_m ** 2
        parent_spec = _ArrowSpec(x_s_m, y_s_m, z_s_m, x_m, y_m, z_m, length_m)

        x_s_sl, y_s_sl, z_s_sl = self.geometry_data.nodes[i_sl]
        length_sl = x_sl ** 2 + y_sl ** 2 + z_sl ** 2
        child_spec = _ArrowSpec(x_s_sl, y_s_sl, z_s_sl, x_sl, y_sl, z_sl, length_sl)

        for j in range(len(self.arrows_objects)):
            if self._arrows_match_parent_child(
                    self.arrows_objects[j], parent_spec, child_spec):
                for arrow in self.arrows_objects[j]:
                    arrow.remove()
                del self.arrows_objects[j]
                break
        else:
            if self.arrows_objects:
                logging.warning('arrows_object not found')

        self.fig.canvas.draw_idle()

    @staticmethod
    def _arrow_matches_spec(arrow, spec):
        '''Return True if *arrow* matches the given :class:`_ArrowSpec`.

        Parameters
        ----------
        arrow : LabeledArrow3D
        spec : _ArrowSpec

        Returns
        -------
        bool
        '''
        (x_s, x_e), (y_s, y_e), (z_s, z_e) = arrow._verts3d
        dx, dy, dz = (x_e - x_s), (y_e - y_s), (z_e - z_s)
        tol = 0.05 * spec.length
        return (x_s - tol <= spec.x_s <= x_s + tol and
                y_s - tol <= spec.y_s <= y_s + tol and
                z_s - tol <= spec.z_s <= z_s + tol and
                dx == spec.x_e and dy == spec.y_e and dz == spec.z_e)

    @staticmethod
    def _arrows_match_parent_child(arrow_pair, parent_spec, child_spec):
        '''Return True if *arrow_pair* corresponds to the given parent-child specs.

        Parameters
        ----------
        arrow_pair : sequence of LabeledArrow3D
            The two arrows stored for one parent-child pair.
        parent_spec : _ArrowSpec
            Specification for the parent arrow.
        child_spec : _ArrowSpec
            Specification for the child arrow.

        Returns
        -------
        bool
        '''
        found_m = any(ModeShapePlot._arrow_matches_spec(a, parent_spec)
                      for a in arrow_pair)
        found_sl = any(ModeShapePlot._arrow_matches_spec(a, child_spec)
                       for a in arrow_pair)
        return found_m and found_sl

    def _find_chan_dof_index(self, x_s, y_s, z_s, x_e, y_e, z_e):
        '''Search ``self.channels_objects`` for the arrow matching given endpoints.

        Parameters
        ----------
        x_s, y_s, z_s : float
            Expected arrow start coordinates.
        x_e, y_e, z_e : float
            Expected arrow end coordinates.

        Returns
        -------
        int or None
            Index in ``self.channels_objects`` if found, else *None*.
        '''
        for j, chan_obj in enumerate(self.channels_objects):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = chan_obj[0]._verts3d
            if (nearly_equal(x_s_, x_s, 2) and nearly_equal(x_e_, x_e, 2) and
                    nearly_equal(y_s_, y_s, 2) and nearly_equal(y_e_, y_e, 2) and
                    nearly_equal(z_s_, z_s, 2) and nearly_equal(z_e_, z_e, 2)):
                return j
        return None

    def take_chan_dof(self, chan, node, dof):
        '''
        Remove the arrow and text objects associated with the channel -
        DOF assignment.

        Parameters
        ----------
            chan: integer
                Index of the channel.
            node: integer
                Index of the node in the internal node table
            dof: 3-tuple {az,elev,chan_name}
                az, elev: float
                    Azimuth and elevation of the DOF assignment
                chan_name: str
                    Name of the channel to annotate

        '''
        if not isinstance(node, int):
            raise TypeError(f"Expected int for 'node', got {type(node).__name__!r}.")
        if not isinstance(dof, (tuple, list)):
            raise TypeError(f"Expected tuple or list for 'dof', got {type(dof).__name__!r}.")
        if len(dof) != 3:
            raise ValueError(f"Expected sequence of length 3 for 'dof', got {len(dof)}.")

        x_s, y_s, z_s = self.geometry_data.nodes[node]
        x_e, y_e, z_e = dof[0] + x_s, dof[1] + y_s, dof[2] + z_s

        j = self._find_chan_dof_index(x_s, y_s, z_s, x_e, y_e, z_e)
        if j is not None:
            for obj in self.channels_objects[j]:
                obj.remove()
            del self.channels_objects[j]
        elif self.channels_objects:
            logging.warning('chandof_object not found')

        self.fig.canvas.draw_idle()

    def draw_axis(self):
        '''
        Draw the axis arrows. Length is based on the current data limits.
        Removes the current arrows if the exist.
        '''

        for axis in ['X', 'Y', 'Z']:
            if axis in self.axis_obj:
                try:
                    self.axis_obj[axis].remove()
                    del self.axis_obj[axis]
                except ValueError:
                    continue

        axis = self.subplot.add_artist(
            LabeledArrow3D(0, 0, 0, self.scale, 0, 0,
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="r", visible=self.show_axis))
        axis.add_label('X', color='r', visible=self.show_axis)

#         text = self.subplot.text(
#             self.scale * 1.1,
#             0,
#             0,
#             'X',
#             zdir=None,
#             color='r',
#             visible=self.show_axis)
        self.axis_obj['X'] = axis

        axis = self.subplot.add_artist(
            LabeledArrow3D(0, 0, 0, 0, self.scale, 0,
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="g", visible=self.show_axis))
        axis.add_label('Y', color='g', visible=self.show_axis)

#         text = self.subplot.text(
#             0,
#             self.scale * 1.1,
#             0,
#             'Y',
#             zdir=None,
#             color='g',
#             visible=self.show_axis)
        self.axis_obj['Y'] = axis

        axis = self.subplot.add_artist(
            LabeledArrow3D(0, 0, 0, 0, 0, self.scale,
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="b", visible=self.show_axis))
        axis.add_label('Z', color='b', visible=self.show_axis)

#         text = self.subplot.text(
#             0,
#             0,
#             self.scale * 1.1,
#             'Z',
#             zdir=None,
#             color='b',
#             visible=self.show_axis)
        self.axis_obj['Z'] = axis

        self.fig.canvas.draw_idle()

    def refresh_axis(self, visible=None):
        '''
        Refresh the axis arrows and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the axis arrows

        '''
        visible = bool(visible)

        if visible is not None:
            self.show_axis = visible

        for axis in self.axis_obj.values():
            axis.set_visible(self.show_axis)
        self.fig.canvas.draw()

    def draw_nodes(self):
        ''''
        Draws nodes from the node list of PreProcessingTools.GeometryData
        The currently stored displacement values are used for moving the
        nodes.
        '''
        for key, node in self.geometry_data.nodes.items():
            self.add_node(*node, i=key)

    def refresh_nodes(self, visible=None):
        '''
        Refresh the nodes and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the nodes

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_nodes = visible

        for key in self.geometry_data.nodes.keys():
            node = self.geometry_data.nodes[key]
            disp_node = self.disp_nodes.get(key, [0, 0, 0])
            phase_node = self.phi_nodes.get(key, [0, 0, 0])
            patch = self.patches_objects.get(key, None)
            if isinstance(patch, (tuple, list)):
                for obj in patch:
                    obj.set_visible(self.show_nodes)
                x = node[0] + disp_node[0] * \
                    np.cos(self.seq_num / 25 * 2 * np.pi + phase_node[0])
                y = node[1] + disp_node[1] * \
                    np.cos(self.seq_num / 25 * 2 * np.pi + phase_node[1])
                z = node[2] + disp_node[2] * \
                    np.cos(self.seq_num / 25 * 2 * np.pi + phase_node[2])
                # print('in refresh nodes', x,y,z)
                # if 'PIV' in key:
                #    print(key, disp_node, phase_node)

                patch[0].set_offsets([x, y])
                patch[0].set_3d_properties(z, 'z')

                patch[1].set_position([x, y])
                patch[1].set_3d_properties(z, None)

        self.fig.canvas.draw_idle()

    def draw_lines(self):
        '''
        Draws all line from the line list of PreProcessingTools.GeometryProcessor
        The currently stored displacement values are used for moving the
        nodes.
        '''
        for i, line in enumerate(self.geometry_data.lines):
            self.add_line(line, i)
            self.add_nd_line(line, i)
            self.refresh_lines()
            self.refresh_nd_lines()

        # self.lines_objects[-1].remove()
        # del self.lines_objects[-1]

        # node = line[0]
        # self.lines_objects.append(
        #     self.subplot.plot(
        #     [self.geometry_data.nodes[node][0]
        #      + self.disp_nodes[node][0]],
        #     [self.geometry_data.nodes[node][1]
        #      + self.disp_nodes[node][1]],
        #     [self.geometry_data.nodes[node][2]
        #      + self.disp_nodes[node][2] ],
        #     color=self.beamcolor,
        #     marker='o', markersize=6,
        #     visible=self.show_lines,)[0])

        for i in self.geometry_data.nodes.keys():
            self.add_cn_line(i)

    def refresh_lines(self, visible=None):
        '''
        Refresh the lines and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the lines

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_lines = visible

        for line, line_node in zip(
                self.lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + self.phi_nodes[node][0])
                 for node in line_node]
            y = [self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + self.phi_nodes[node][1])
                 for node in line_node]
            z = [self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + self.phi_nodes[node][2])
                 for node in line_node]
            line.set_visible(self.show_lines)
            line.set_data_3d([x, y, z])
            # line.set_3d_properties(z)

        for key in self.geometry_data.nodes.keys():
            node = self.geometry_data.nodes[key]
            disp_node = self.disp_nodes.get(key, [0, 0, 0])
            phi_node = self.phi_nodes.get(key, [0, 0, 0])
            line = self.cn_lines_objects.get(key, None)
            if line is None:
                continue

            x = [node[0], node[0] + disp_node[0]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[0])]
            y = [node[1], node[1] + disp_node[1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[1])]
            z = [node[2], node[2] + disp_node[2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[2])]
            line.set_visible(self.show_cn_lines)
            line.set_data_3d([x, y, z])
            # line.set_3d_properties(z)

        self.fig.canvas.draw_idle()

    def refresh_nd_lines(self, visible=None):
        '''
        Refresh the non-displaced lines and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the non-displaced lines

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_nd_lines = visible

        for line, line_node in zip(
                self.nd_lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[node][0]
                 for node in line_node]
            y = [self.geometry_data.nodes[node][1]
                 for node in line_node]
            z = [self.geometry_data.nodes[node][2]
                 for node in line_node]
            line.set_visible(self.show_nd_lines)
            line.set_data_3d([x, y, z])
            # line.set_3d_properties(z)

        self.fig.canvas.draw_idle()

    def refresh_cn_lines(self, visible=None):
        '''
        Refresh the connecting lines and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the non-displaced lines

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_cn_lines = visible

        for key, node in self.geometry_data.nodes.items():
            disp_node = self.disp_nodes.get(key, [0, 0, 0])
            phi_node = self.phi_nodes.get(key, [0, 0, 0])
            line = self.cn_lines_objects.get(key, None)
            if line is not None:
                x = [node[0], node[0] + disp_node[0]
                     * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[0])]
                y = [node[1], node[1] + disp_node[1]
                     * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[1])]
                z = [node[2], node[2] + disp_node[2]
                     * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[2])]
                line.set_visible(self.show_cn_lines)
                line.set_data_3d([x, y, z])
                # line.set_3d_properties(z)

        self.fig.canvas.draw_idle()

    def draw_parent_childs(self):
        '''
        Draw arrows for all parent-child definitions stored in the
        internal parent-child definition table.
        '''
        for i, (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl) in enumerate(
                self.geometry_data.parent_childs):
            self.add_parent_child(
                i=i,
                parent=NodeCoords(i_m, x_m * self.scale, y_m * self.scale, z_m * self.scale),
                child=NodeCoords(i_sl, x_sl * self.scale, y_sl * self.scale, z_sl * self.scale))

    def refresh_parent_childs(self, visible=None):
        '''
        Refresh the parent-child arrows and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Will not be shown in displaced mode (modeshape)

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the parent-child arrows

        '''
        if visible is not None:
            visible = bool(visible)
            self.show_parent_childs = visible

        for patch in self.arrows_objects:
            for obj in patch:
                obj.set_visible(self.show_parent_childs)
        self.fig.canvas.draw_idle()

    def draw_chan_dofs(self):
        '''
        Draw arrows and numbers for all channel-DOF assignments stored
        in the channel - DOF assignment table of PreProcessingTools.GeometrProcessor
        '''
        for i, chan_dof in enumerate(self.chan_dofs):

            chan, node, az, elev, chan_name = chan_dof[0:4] + chan_dof[-1:]
            if node is None:
                continue
            if node not in self.geometry_data.nodes.keys():
                continue
            self.add_chan_dof(chan, node, az, elev, chan_name, i)

    def refresh_chan_dofs(self, visible=None):
        '''
        Refresh the arrows indicating the channel-dof assignments
        and make them visible/invisible, e.g. after programmatically
        changing visibility flags.

        Will not be shown in displaced mode (modeshape)

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the channel-dof assignment arrows

        '''
        if visible is not None:
            visible = bool(visible)
            self.show_chan_dofs = visible

        for patch in self.channels_objects:
            patch.set_visible(self.show_chan_dofs)
        self.fig.canvas.draw_idle()

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

    def draw_msh(self):
        '''
        Draw mode shapes by assigning displacement values to the
        nodes based on the channel - DOF assignments and the parent -
        child definitions. Draws the displaced nodes and beams.

        .. Todo::
            * The computation of resulting magnitude and phase angles for
              displacements based on parent-child definitions is currently
              more or less broken. It should be possible, even in 3D to
              compute exact solutions.
        '''
        mode_shape = self.mode_shapes[:, self.mode_index[1], self.mode_index[0]]
        mode_shape = ModalBase.rescale_mode_shape(mode_shape)
        ampli = self.amplitude

        self.disp_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}
        self.phi_nodes = {i: [0, 0, 0] for i in self.geometry_data.nodes.keys()}

        self._compute_chan_dof_displacements(mode_shape, ampli)
        self._compute_parent_child_displacements()

        self.refresh_nodes()
        self.refresh_lines()
        self.refresh_chan_dofs(False)
        self.refresh_parent_childs(False)
        if self.animated:
            self.stop_ani()
            self.animate()
        self.set_equal_aspect()

        self.fig.canvas.draw()

    def set_equal_aspect(self):

        minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()
        dx, dy, dz = (maxx - minx), (maxy - miny), (maxz - minz)

        if dx != dy or dx != dz:
            midx = 0.5 * (minx + maxx)
            midy = 0.5 * (miny + maxy)
            midz = 0.5 * (minz + maxz)

            hrange = max(dy, dy, dz) * 0.5
            self.subplot.set_xlim3d(midx - hrange, midx + hrange)
            self.subplot.set_ylim3d(midy - hrange, midy + hrange)
            self.subplot.set_zlim3d(midz - hrange, midz + hrange)

    def stop_ani(self):
        '''
        Convenience method to stop the animation and restore the still plot
        '''
        if self.animated or self.data_animated:
            self.seq_num = next(self.line_ani.frame_seq)
            self.line_ani._stop()
            if self.trace_objects:
                for i in range(len(self.trace_objects) - 1, -1, -1):
                    try:
                        self.trace_objects[i].remove()
                    except BaseException as e:
                        print(e)

                    del self.trace_objects[i]
            # self.draw_trace = False
            self.animated = False
            self.data_animated = False
            for c in self.connect_handles:
                self.fig.canvas.mpl_disconnect(c)
            self.draw_nodes()
            self.refresh_nodes()
            self.draw_lines()
            self.refresh_lines()
            self.refresh_nd_lines()
            self.refresh_parent_childs()
            self.refresh_chan_dofs()
            # self.draw_msh()


    def _animate_draw_traces(self):
        '''Draw trace ellipses for all moving nodes (helper for ``_animate_init_lines``).'''
        if not self.trace_objects:
            return
        for i in range(len(self.trace_objects) - 1, -1, -1):
            try:
                self.trace_objects[i].remove()
            except BaseException:
                pass
            del self.trace_objects[i]
        # assemble the list of moving nodes; parent-child not accounted for
        moving_nodes = {
            chan_dof[1]
            for chan_dof in self.chan_dofs
            if chan_dof[1] is not None and chan_dof[1] in self.geometry_data.nodes
        }
        clist = itertools.cycle(['darkgray'] * len(moving_nodes))
        angles = np.arange(0, 2 * np.pi, np.pi / 180)
        for node in moving_nodes:
            self.trace_objects.append(
                self.subplot.plot(
                    xs=self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
                    * np.cos(angles + self.phi_nodes[node][0]),
                    ys=self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                    * np.cos(angles + self.phi_nodes[node][1]),
                    zs=self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                    * np.cos(angles + self.phi_nodes[node][2]),
                    color=next(clist), linewidth=1, linestyle=(0, (1, 1)))[0])

    def _animate_init_lines(self):
        '''Initialize line objects for modal animation (``init_func`` callback).'''
        minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()

        for i, line in enumerate(self.lines_objects):
            line.set_visible(False)
            beamcolor = (self.beamcolor[i]
                         if isinstance(self.beamcolor, (list, tuple, np.ndarray))
                         else self.beamcolor)
            beamstyle = (self.beamstyle[i]
                         if isinstance(self.beamstyle, (list, tuple, np.ndarray))
                         else self.beamstyle)
            line.set_color(beamcolor)
            line.set_linestyle(beamstyle)

        for line in self.nd_lines_objects:
            line.set_visible(False)

        for line in self.cn_lines_objects.values():
            line.set_visible(False)

        self.fig.canvas.draw()
        self.subplot.set_xlim3d(minx, maxx)
        self.subplot.set_ylim3d(miny, maxy)
        self.subplot.set_zlim3d(minz, maxz)

        if self.show_cn_lines:
            self._animate_draw_traces()

        return (self.lines_objects + self.nd_lines_objects + self.trace_objects
                + list(self.cn_lines_objects.values()))

    def _animate_apply_line_positions(self, num):
        '''Update displaced line positions for animation frame *num*.'''
        phase = num / 25 * 2 * np.pi
        for line, line_node in zip(self.lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[n][0] + self.disp_nodes[n][0]
                 * np.cos(phase + self.phi_nodes[n][0]) for n in line_node]
            y = [self.geometry_data.nodes[n][1] + self.disp_nodes[n][1]
                 * np.cos(phase + self.phi_nodes[n][1]) for n in line_node]
            z = [self.geometry_data.nodes[n][2] + self.disp_nodes[n][2]
                 * np.cos(phase + self.phi_nodes[n][2]) for n in line_node]
            line.set_visible(self.show_lines)
            line.set_data_3d([x, y, z])

    def _maybe_save_animation_frame(self, num):
        """Save animation frame to disk if a save path is configured."""
        if self.save_ani_path and num <= 25:
            self.fig.savefig(
                self.save_ani_path
                / f'{self.select_modes.index(self.mode_index)}'
                / f'ani_{num}.pdf')

    def _animate_update_lines(self, num):
        '''Update all animated objects for frame *num* (``func`` callback).'''
        self._animate_apply_line_positions(num)
        rets = [self.lines_objects]

        if self.nd_lines_objects[0].get_visible() != self.show_nd_lines:
            for line in self.nd_lines_objects:
                line.set_visible(self.show_nd_lines)
            rets.append(self.nd_lines_objects)

        for trace_obj in self.trace_objects:
            trace_obj.set_visible(self.show_cn_lines)
            rets.append([trace_obj])

        if self.axis_obj['X'].get_visible() != self.show_axis:
            for axis in self.axis_obj.values():
                axis.set_visible(self.show_axis)
            rets.append(self.axis_obj.values())

        self._maybe_save_animation_frame(num)
        return list(itertools.chain.from_iterable(rets))

    def animate(self):
        '''
        Create necessary objects to animate the currently displayed
        deformed structure.

        If self.save_ani_path is given, the animation will be saved to that
        folder. The **numbering** of the **files**
        follows the order in which the modes were selected in the
        stabilization diagram.
        '''
        if self.animated:
            return self.stop_ani()
        if self.data_animated:
            self.stop_ani()
        self.animated = True

        c1 = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        c2 = self.fig.canvas.mpl_connect('button_press_event', self._button_press)
        c3 = self.fig.canvas.mpl_connect(
            'button_release_event', self._button_release)
        self.connect_handles = [c1, c2, c3]
        self.button_pressed = None

        self.line_ani = matplotlib.animation.FuncAnimation(
            fig=self.fig,
            func=self._animate_update_lines,
            init_func=self._animate_init_lines,
            interval=50,
            save_count=50,
            blit=True)

        self.fig.canvas.draw()


    def _data_animate_init_lines(self):
        '''Initialize lines for data animation (``init_func`` callback).'''
        minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()

        self.subplot.cla()
        self.draw_lines()
        for line in self.lines_objects:
            line.set_visible(False)
        for line in self.nd_lines_objects:
            line.set_visible(False)
        for line in self.cn_lines_objects.values():
            line.set_visible(False)

        self.subplot.set_xlim3d(minx, maxx)
        self.subplot.set_ylim3d(miny, maxy)
        self.subplot.set_zlim3d(minz, maxz)

        return (self.lines_objects + self.nd_lines_objects
                + list(self.cn_lines_objects.values()))

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

    def _data_animate_update_lines(self, num):
        '''Update all animated objects for data-animation frame *num*.'''
        self.callback(f'{num/self.prep_signals.sampling_rate:.4f}')
        disp_nodes = self._compute_data_disp_nodes(num)

        for line, line_node in zip(self.lines_objects, self.geometry_data.lines):
            coords = [[self.geometry_data.nodes[n][k] + disp_nodes[n][k]
                       for n in line_node] for k in range(3)]
            line.set_visible(self.show_lines)
            line.set_data_3d(coords)
            line.set_color('b')

        for line in self.nd_lines_objects:
            line.set_visible(self.show_nd_lines)

        for key in self.geometry_data.nodes.keys():
            node_coords = self.geometry_data.nodes[key]
            disp_node = disp_nodes.get(key, [0, 0, 0])
            cn_line = self.cn_lines_objects.get(key, None)
            if cn_line is not None:
                coords = [[node_coords[k], node_coords[k] + disp_node[k]] for k in range(3)]
                cn_line.set_data_3d(coords)
                cn_line.set_visible(self.show_cn_lines)

        return (self.lines_objects + self.nd_lines_objects
                + list(self.cn_lines_objects.values()))

    def filter_and_animate_data(self, callback=None):
        '''
        Animate the acquired vibration data to check the real vibration
        displacement of the structure against the identified modes.
        '''
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.axis_obj = {}

        if self.data_animated:
            return self.stop_ani()
        if self.animated:
            self.stop_ani()
        self.data_animated = True

        c1 = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        c2 = self.fig.canvas.mpl_connect('button_press_event', self._button_press)
        c3 = self.fig.canvas.mpl_connect(
            'button_release_event', self._button_release)
        self.connect_handles = [c1, c2, c3]
        self.button_pressed = None

        if callback is not None:
            self.callback = callback
        self.line_ani = matplotlib.animation.FuncAnimation(
            fig=self.fig,
            func=self._data_animate_update_lines,
            frames=range(self.prep_signals.signals_filtered.shape[0]),
            init_func=self._data_animate_init_lines,
            interval=1 / self.prep_signals.sampling_rate,
            save_count=0,
            blit=True)

        self.fig.canvas.draw()

    def _button_press(self, event):
        if event.inaxes == self.subplot:
            self.button_pressed = event.button

    def _button_release(self, event):
        self.button_pressed = None

    def _on_move(self, event):
        if not self.button_pressed:
            return

        for line in self.lines_objects:
            line.set_visible(False)
        for line in self.nd_lines_objects:
            line.set_visible(False)
        for line in self.cn_lines_objects.values():
            line.set_visible(False)
        # self.fig.canvas.draw()
        self.line_ani._setup_blit()
        # self.line_ani._start()


class LabeledArrow3D(matplotlib.patches.FancyArrowPatch):
    '''
    credit goes to (don't know the original author):
    http://pastebin.com/dWvFxb1Q
    draw an arrow in 3D space
    '''

    def __init__(self, *pos, **kwargs):
        '''
        inherit from matplotlib.patches.FancyArrowPatch
        and set self._verts3d class variable
        dx,dy,dz is understood as fractions of the axis'limits

        Parameters
        ----------
        *pos : float
            Positional args: x, y, z, dx, dy, dz [, extra FancyArrowPatch args].
        **kwargs :
            Keyword arguments forwarded to FancyArrowPatch.
        '''
        x, y, z, dx, dy, dz = pos[:6]
        rest = pos[6:]
        self.text = None
        self._verts3d = (x, y, z, dx, dy, dz)
        super().__init__((x, x + dx), (y, y + dy), *rest, **kwargs)

    def set_visible(self, b):

        if self.text is not None:
            self.text.set_visible(b)
        super().set_visible(b)

    def add_label(self, text, color=None, visible=True):

        if self.axes is None:
            logging.warning('The arrow must be added to an axes, before a label can be added.')

        (x, y, z, dx, dy, dz) = self._verts3d

        self.text = self.axes.text(
            x + dx,
            y + dy,
            z + dz,
            text,
            color=color,
            visible=visible)

    def draw(self, renderer):
        '''
        get the projection from the 3D point to 2D point to draw the arrow
        '''

        # scale and direction of the arrow as fractions of axis limits
        x, y, z, dx, dy, dz = self._verts3d

        minx, maxx, miny, maxy, minz, maxz = self.axes.get_w_lims()
        lx, ly, lz = (maxx - minx), (maxy - miny), (maxz - minz)

        # rescale arrow to fraction axis limits
        xs3d = [x, x + lx * dx]
        ys3d = [y, y + ly * dy]
        zs3d = [z, z + lz * dz]
        xs, ys, _zs = mpl_toolkits.mplot3d.axes3d.proj3d.proj_transform(
            xs3d, ys3d, zs3d, self.axes.M)
        if self.text:
            self.text.set_position_3d((xs3d[1], ys3d[1], zs3d[1]))
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1, dx, dy, dz = self._verts3d
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = mpl_toolkits.mplot3d.axes3d.proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


if __name__ == "__main__":
    pass
