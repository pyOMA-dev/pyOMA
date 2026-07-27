# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""pyvista/VTK rendering backends for 3-D mode-shape visualisation.

Two read-only backends reproduce the geometry and animation of the
matplotlib reference backend
:class:`~pyOMA.core.PlotMSH.ModeShapePlot`:

* :class:`ModeShapePlotPVQt` embeds a :class:`pyvistaqt.QtInteractor`
  (a ``QWidget``) and animates server-side with two chained
  :class:`vtkWarpVector` filters, so a frame costs two scale-factor
  assignments and no Python point math.
* :class:`ModeShapePlotPVJupyter` targets notebooks, where the scene is
  serialised to the browser once and server-side VTK filters therefore do
  not run.  It precomputes a full harmonic cycle of point positions
  instead.

Both derive their meshes from :attr:`disp_nodes` / :attr:`phi_nodes`,
which :class:`~pyOMA.core.ModeShapeBase.ModeShapeBase` owns; the meshes
are never written back to.

Neither pyvista nor VTK is imported at module level, so this module can
be imported (and its :class:`ImportError` message read) without the
``pyOMA[pyvista]`` extra installed.

Notes
-----
Mode animation follows

.. math:: x(t) = x_0 + \\Re(\\varphi)\\cos(2 \\pi t) - \\Im(\\varphi)\\sin(2 \\pi t)

with :math:`\\Re(\\varphi) = d \\cos \\phi` and
:math:`\\Im(\\varphi) = d \\sin \\phi` per node and axis, taken from
:attr:`disp_nodes` (``d``) and :attr:`phi_nodes` (``phi``).  This is the
angle-addition form of the matplotlib backend's
``x0 + d * cos(2 * pi * t + phi)``, so both backends trace the same path.
The amplitude scaling is *already contained* in ``disp_nodes`` -- the base
class multiplies it in when the displacements are computed -- so the warp
scale factors are pure ``cos``/``-sin`` and must not be scaled again.
"""

import numpy as np

from .ModeShapeBase import ModeShapeBase
from .PreProcessingTools import GeometryProcessor
from .Helpers import calc_xyz

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

__all__ = ['ModeShapePlotPVQt', 'ModeShapePlotPVJupyter']

_INSTALL_HINT = (
    "This backend requires pyvista and VTK. "
    "Install them with 'pip install pyOMA[pyvista]'.")

#: Point-data array names carrying the real and imaginary mode components.
_ARRAY_RE = 'mode_re'
_ARRAY_IM = 'mode_im'

#: Frames per harmonic cycle and timer interval of the Qt animation; chosen
#: to match the matplotlib backend's ``num / 25 * 2 * pi`` at 50 ms/frame.
_QT_FRAMES_PER_CYCLE = 25
_QT_INTERVAL_MS = 50


def _import_pyvista():
    """Import pyvista and the VTK pieces used here.

    Returns
    -------
    pv : module
        The :mod:`pyvista` module.
    vtk_warp_vector : type
        :class:`vtkmodules.vtkFiltersGeneral.vtkWarpVector`.
    vtk_data_object : type
        :class:`vtkmodules.vtkCommonDataModel.vtkDataObject`.

    Raises
    ------
    ImportError
        If pyvista or VTK is unavailable, with an actionable message.
    """
    try:
        import pyvista as pv
        from vtkmodules.vtkFiltersGeneral import vtkWarpVector
        from vtkmodules.vtkCommonDataModel import vtkDataObject
    except ImportError as e:
        raise ImportError(f'{_INSTALL_HINT} (missing: {e.name})') from e
    return pv, vtkWarpVector, vtkDataObject


class _WarpChain:
    """Two chained ``vtkWarpVector`` filters realising the harmonic motion.

    The first filter displaces the base points by ``scale_re * mode_re``,
    the second adds ``scale_im * mode_im`` on top.  Point data passes
    through both filters unchanged, so the second still sees ``mode_im``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh carrying the ``mode_re`` and ``mode_im`` point arrays.
    vtk_warp_vector : type
        The ``vtkWarpVector`` class.
    vtk_data_object : type
        The ``vtkDataObject`` class, for the field-association enum.
    """

    def __init__(self, mesh, vtk_warp_vector, vtk_data_object):
        points = vtk_data_object.FIELD_ASSOCIATION_POINTS

        self._warp_re = vtk_warp_vector()
        self._warp_re.SetInputData(mesh)
        self._warp_re.SetInputArrayToProcess(0, 0, 0, points, _ARRAY_RE)

        self._warp_im = vtk_warp_vector()
        self._warp_im.SetInputConnection(self._warp_re.GetOutputPort())
        self._warp_im.SetInputArrayToProcess(0, 0, 0, points, _ARRAY_IM)

        self.set_phase(0.0)

    def set_phase(self, t):
        """Move the output to fraction *t* of a harmonic cycle.

        Parameters
        ----------
        t : float
            Cycle fraction; ``t = 1`` is one full period.
        """
        self._warp_re.SetScaleFactor(np.cos(2 * np.pi * t))
        self._warp_im.SetScaleFactor(-np.sin(2 * np.pi * t))
        self._warp_im.Update()

    @property
    def output(self):
        """vtkPolyData : the warped output, updated in place by :meth:`set_phase`."""
        return self._warp_im.GetOutput()

    @property
    def output_port(self):
        """vtkAlgorithmOutput : the pipeline port a mapper should connect to."""
        return self._warp_im.GetOutputPort()


class PyVistaMeshMixin:
    """Builds the VTK meshes of a geometry and keeps them in sync with a mode.

    The node-name to vtk-point-id map is established once in
    :meth:`_build_node_index` and reused by every mesh (beams, nodes,
    connecting lines, surfaces and arrows).  ``geometry_data.nodes`` is
    keyed by string node names while VTK addresses points by integer id,
    and rebuilding that correspondence per artist is the single most
    error-prone thing these backends could do.
    """

    # ── Node index: the one place names and point ids are related ────────────

    def _build_node_index(self):
        """Establish the node-name <-> vtk-point-id correspondence.

        Sets :attr:`node_names` (point id -> name), :attr:`node_ids`
        (name -> point id) and :attr:`base_points`, the undeformed
        coordinates in point-id order.
        """
        names = [str(name) for name in self.geometry_data.nodes.keys()]

        self.node_names = names
        self.node_ids = {name: i for i, name in enumerate(names)}
        # dict keys are unique, but the str() above could in principle
        # collapse two distinct keys onto one name.
        if len(self.node_ids) != len(names):
            raise ValueError(
                'Node names are not unique after conversion to str; '
                'the node-name to vtk-point-id map would not be a bijection.')

        self.base_points = np.array(
            [self.geometry_data.nodes[name] for name in
             self.geometry_data.nodes.keys()], dtype=float).reshape(-1, 3)

    def _node_id(self, name):
        """Return the vtk point id of node *name*, or *None* with a warning."""
        node_id = self.node_ids.get(str(name))
        if node_id is None:
            logger.warning(f'Node {name} is not part of the geometry. Skipping.')
        return node_id

    # ── Cell arrays ──────────────────────────────────────────────────────────

    def _line_cells(self):
        """Return the beam connectivity as a flat VTK line-cell array."""
        cells = []
        for line in self.geometry_data.lines:
            start, end = self._node_id(line[0]), self._node_id(line[1])
            if start is None or end is None:
                continue
            cells += [2, start, end]
        return np.array(cells, dtype=np.int64)

    def _face_cells(self):
        """Return the surface connectivity as a flat VTK face-cell array."""
        cells = []
        for surface in getattr(self.geometry_data, 'surfaces', []):
            ids = [self._node_id(name) for name in surface]
            if any(i is None for i in ids):
                continue
            cells += [len(ids)] + ids
        return np.array(cells, dtype=np.int64)

    # ── Mode vectors ─────────────────────────────────────────────────────────

    def _mode_vectors(self):
        """Return the real and imaginary mode components in point-id order.

        Returns
        -------
        mode_re, mode_im : ndarray, shape (n_nodes, 3)
            ``disp * cos(phi)`` and ``disp * sin(phi)`` respectively.  The
            amplitude factor is already contained in ``disp_nodes``.
        """
        n_nodes = len(self.node_names)
        mode_re = np.zeros((n_nodes, 3))
        mode_im = np.zeros((n_nodes, 3))

        for name in self.geometry_data.nodes.keys():
            node_id = self.node_ids[str(name)]
            disp = self.disp_nodes.get(name, [0, 0, 0])
            phi = self.phi_nodes.get(name, [0, 0, 0])
            for axis in range(3):
                mode_re[node_id, axis] = disp[axis] * np.cos(phi[axis])
                mode_im[node_id, axis] = disp[axis] * np.sin(phi[axis])

        return mode_re, mode_im

    def mode_points(self, t):
        """Return the deformed node coordinates at cycle fraction *t*.

        Parameters
        ----------
        t : float
            Cycle fraction; ``t = 1`` is one full period.

        Returns
        -------
        ndarray, shape (n_nodes, 3)
        """
        mode_re, mode_im = self._mode_vectors()
        return (self.base_points
                + mode_re * np.cos(2 * np.pi * t)
                - mode_im * np.sin(2 * np.pi * t))

    # ── Mesh construction ────────────────────────────────────────────────────

    #: Whether :meth:`_build_meshes` should attach ``vtkWarpVector`` chains.
    #: Client-side rendering cannot run server-side filters, so the Jupyter
    #: backend turns them off and swaps precomputed points instead.
    _USES_WARP = True

    def _build_meshes(self):
        """Create the deformable meshes and store them in ``self.meshes``.

        The meshes all share :attr:`base_points` and the same node index;
        they differ only in the cells they carry, so that beams, nodes,
        connecting lines and surfaces can be shown and hidden separately.
        """
        pv = self._pv
        points = self.base_points
        n_nodes = points.shape[0]

        meshes = {}

        meshes['beams'] = pv.PolyData(points.copy(), lines=self._line_cells())
        meshes['nodes'] = pv.PolyData(points.copy())

        faces = self._face_cells()
        if faces.size:
            meshes['surfaces'] = pv.PolyData(points.copy(), faces=faces)

        # Connecting lines run from the undeformed to the deformed position
        # of each node, so the mesh holds both: the first n_nodes points are
        # pinned (zero mode vectors), the second n_nodes move.
        cn_cells = np.empty((n_nodes, 3), dtype=np.int64)
        cn_cells[:, 0] = 2
        cn_cells[:, 1] = np.arange(n_nodes)
        cn_cells[:, 2] = np.arange(n_nodes) + n_nodes
        meshes['cn_lines'] = pv.PolyData(
            np.vstack([points, points]), lines=cn_cells.ravel())

        # The undeformed wireframe never warps.
        self.nd_mesh = pv.PolyData(points.copy(), lines=self._line_cells())

        self.meshes = meshes
        self._update_mode_arrays()

        self.warps = {
            key: _WarpChain(mesh, self._vtk_warp_vector, self._vtk_data_object)
            for key, mesh in meshes.items()} if self._USES_WARP else {}

    def _update_mode_arrays(self):
        """Write the current mode into the ``mode_re``/``mode_im`` point arrays."""
        mode_re, mode_im = self._mode_vectors()
        zeros = np.zeros_like(mode_re)

        for key, mesh in self.meshes.items():
            if key == 'cn_lines':
                # pinned half first, moving half second
                mesh.point_data[_ARRAY_RE] = np.vstack([zeros, mode_re])
                mesh.point_data[_ARRAY_IM] = np.vstack([zeros, mode_im])
            else:
                mesh.point_data[_ARRAY_RE] = mode_re
                mesh.point_data[_ARRAY_IM] = mode_im

    def set_phase(self, t):
        """Move every deformable mesh to cycle fraction *t*."""
        for warp in self.warps.values():
            warp.set_phase(t)

    def warped_points(self, key='beams'):
        """Return the current warped points of mesh *key*.

        Parameters
        ----------
        key : str, optional
            One of the keys of :attr:`meshes`.  Default is ``'beams'``.

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        return np.asarray(self._pv.wrap(self.warps[key].output).points)

    # ── Static overlays ──────────────────────────────────────────────────────

    def _arrow_length(self):
        """Return the arrow length used for the axis triad and DOF arrows."""
        xmin, xmax, ymin, ymax, zmin, zmax = self._compute_node_bounds()
        return self.scale * max(xmax - xmin, ymax - ymin, zmax - zmin)

    def _axis_arrows(self):
        """Return ``{'X': PolyData, 'Y': ..., 'Z': ...}`` for the axis triad."""
        pv = self._pv
        length = self._arrow_length()
        directions = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}
        return {name: pv.Arrow(start=(0, 0, 0), direction=direction,
                               scale=length)
                for name, direction in directions.items()}

    def _chan_dof_arrows(self):
        """Return one arrow per channel-DOF assignment, at the undeformed nodes.

        Channel-DOF arrows are drawn on the undeformed geometry, matching
        the matplotlib backend, which does not animate them either.
        """
        pv = self._pv
        length = self._arrow_length()
        arrows = []
        for chan_dof in self.chan_dofs:
            _chan, node, az, elev = chan_dof[0:4]
            node_id = self.node_ids.get(str(node))
            if node_id is None:
                continue
            direction = calc_xyz(az * np.pi / 180, elev * np.pi / 180, r=1)
            arrows.append(pv.Arrow(start=self.base_points[node_id],
                                   direction=direction, scale=length))
        return arrows

    def _parent_child_arrows(self):
        """Return the parent and child DOF arrows of all parent-child relations."""
        pv = self._pv
        length = self._arrow_length() * 0.8
        arrows = []
        for parent, x_m, y_m, z_m, child, x_sl, y_sl, z_sl in \
                self.geometry_data.parent_childs:
            for node, direction in ((parent, (x_m, y_m, z_m)),
                                    (child, (x_sl, y_sl, z_sl))):
                node_id = self.node_ids.get(str(node))
                if node_id is None or np.allclose(direction, 0):
                    continue
                arrows.append(pv.Arrow(start=self.base_points[node_id],
                                       direction=direction, scale=length))
        return arrows


class _PyVistaModeShapeBase(PyVistaMeshMixin, ModeShapeBase):
    """Constructor and shared state of the pyvista mode-shape backends.

    Subclasses build the plotter and implement the animation; everything
    up to the meshes is identical.  See
    :class:`~pyOMA.core.PlotMSH.ModeShapePlot` for the meaning of the
    constructor arguments and the merging-routine table.
    """

    def __init__(self, geometry_data, stabil_calc=None, modal_data=None,
                 prep_signals=None, merged_data=None, config=None, **kwargs):
        '''
        Parameters
        ----------
            geometry_data : PreProcessingTools.GeometryProcessor, required
                    Object containing all the necessary geometry information.

            stabil_calc : StabilDiagram.StabilCalc, optional
                    Object containing the information, which modes were
                    selected from modal_data.

            modal_data : ModalBase.ModalBase, optional
                    Object of one of the classes derived from
                    ModalBase.ModalBase.

            prep_signals : PreProcessingTools.PreProcessSignals, optional
                    Object containing the signals data.

            merged_data : PostProcessingTools.MergePoSER, optional
                    Object containing the merged data.

            config : ModeShapeBase.ModeShapePlotConfig, optional
                    Visual-style configuration object.  The matplotlib-only
                    style fields (marker, linestyle) are ignored.

            **kwargs :
                    Accepts ``selected_mode`` and the deprecated individual
                    style parameters.
        '''
        self._pv, self._vtk_warp_vector, self._vtk_data_object = _import_pyvista()

        kwargs.pop('selected_mode', None)  # accepted but not used
        config, _fig = self._resolve_config(config, kwargs.pop('fig', None), kwargs)

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

        self._build_node_index()
        self._build_meshes()

        if not self.select_modes:
            self.mode_index = None
        else:
            self.mode_index = self.select_modes[0]

    def _apply_config(self, config):
        '''Apply the backend-independent fields of *config* to ``self``.

        The matplotlib-specific style fields (``nodemarker``,
        ``beamstyle``) have no pyvista equivalent and are ignored; colours
        and sizes are mapped onto their pyvista counterparts.
        '''
        self.real = self._check_bool('real', config.real)
        self.scale = self._check_numeric('scale', config.scale)
        self.amplitude = self._check_numeric('amplitude', config.amplitude)
        self.nodesize = self._check_numeric('nodesize', config.nodesize)
        self.linewidth = self._check_numeric_or_seq('linewidth', config.linewidth)
        self.callback_fun = self._check_callable_or_none('callback_fun', config.callback_fun)
        self.save_ani_path = self._check_path_or_none('save_ani_path', config.save_ani_path)
        self.beamcolor = config.beamcolor
        self.nodecolor = config.nodecolor
        self.dpi = self._check_int('dpi', config.dpi)

    def _init_state(self):
        '''Allocate the actor containers on top of the base-class flags.'''
        super()._init_state()
        self.actors = {}
        self.show_surfaces = True

    @property
    def widget(self):
        '''Backend widget hosting the rendering.'''
        raise NotImplementedError

    # ── Rendering hooks ──────────────────────────────────────────────────────

    def draw_msh(self):
        '''Recompute the current mode and push it into the meshes.'''
        self.compute_mode_displacements()
        self._update_mode_arrays()
        self.set_phase(0.0)
        self.render()

    def _render_mode(self):
        '''Render the currently selected mode (backend hook override).'''
        self.draw_msh()

    def _render_amplitude(self):
        '''Re-render after an amplitude change (backend hook override).'''
        self.draw_msh()

    def _render_part(self):
        '''Re-render after a real/complex-part change (backend hook override).'''
        self.draw_msh()

    def render(self):
        '''Ask the plotter to redraw.  Overridden by the concrete backends.'''


class ModeShapePlotPVQt(_PyVistaModeShapeBase):
    """pyvista mode-shape backend rendering into a Qt6 widget.

    Wraps a :class:`pyvistaqt.QtInteractor`, which is itself a
    ``QWidget`` and can therefore be dropped into any Qt layout via the
    :attr:`widget` property.  Camera interaction (rotate, pan, zoom) is
    handled by the interactor, so no event handlers need connecting.

    Animation runs entirely inside VTK: two chained ``vtkWarpVector``
    filters per mesh are driven by setting their scale factors on a
    ``QTimer``, so a frame costs no Python point math.

    Parameters
    ----------
    geometry_data : PreProcessingTools.GeometryProcessor
        Geometry to display.
    off_screen : bool, optional
        Render without a window, into a plain :class:`pyvista.Plotter`.
        Required for headless runs; see :meth:`_make_plotter` for why the
        Qt interactor cannot serve that case.  In this mode there is no
        Qt widget and :attr:`widget` raises.  Default is *False*.
    plotter : pyvistaqt.QtInteractor or pyvista.Plotter, optional
        Existing plotter to draw into.  A new one is created by default.
    **kwargs
        Passed to :class:`_PyVistaModeShapeBase`.

    Examples
    --------
    >>> msh = ModeShapePlotPVQt(geometry_data, stabil_calc, modal_data)  # doctest: +SKIP
    >>> layout.addWidget(msh.widget)                                    # doctest: +SKIP
    >>> msh.change_mode(index=0)                                        # doctest: +SKIP
    >>> msh.animate()                                                   # doctest: +SKIP
    """

    def __init__(self, geometry_data, *args, off_screen=False, plotter=None,
                 **kwargs):
        super().__init__(geometry_data, *args, **kwargs)

        if plotter is None:
            plotter = self._make_plotter(off_screen)
        self.plotter = plotter

        self._frame = 0
        self._timer = None

        self._add_actors()
        self.reset_view()
        self._realize_off_screen_window()

    def _realize_off_screen_window(self):
        '''Force the off-screen render window to be created and drawn once.

        An off-screen :class:`pyvista.Plotter` builds its render window
        lazily on the first ``show``/``screenshot``.  Until that has
        happened, ``render()`` has nothing to draw into and every
        screenshot returns the same first frame, which silently freezes
        the animation.
        '''
        if hasattr(self.plotter, 'winId'):  # a Qt widget draws itself
            return
        self.plotter.show(auto_close=False)

    def _make_plotter(self, off_screen):
        '''Create the plotter backing this backend.

        On-screen use gets a :class:`pyvistaqt.QtInteractor`.  Off-screen
        use gets a plain :class:`pyvista.Plotter`, because the interactor
        renders into a ``QWidget``'s native surface and Qt's ``offscreen``
        platform plugin provides none, leaving VTK without a GL context.
        A plain plotter creates its own off-screen context and is what
        headless runs and CI need.

        Parameters
        ----------
        off_screen : bool

        Returns
        -------
        pyvista.Plotter or pyvistaqt.QtInteractor
        '''
        if off_screen:
            return self._pv.Plotter(off_screen=True)
        try:
            from pyvistaqt import QtInteractor
        except ImportError as e:
            raise ImportError(f'{_INSTALL_HINT} (missing: {e.name})') from e
        return QtInteractor()

    # ── Widget & actors ──────────────────────────────────────────────────────

    @property
    def widget(self):
        '''pyvistaqt.QtInteractor : the ``QWidget`` hosting the rendering.

        Raises
        ------
        RuntimeError
            If this backend was built off-screen, where no Qt widget exists.
        '''
        # QWidget has winId(); pyvista.Plotter does not.  Checking for it
        # avoids importing Qt just to answer this.
        if not hasattr(self.plotter, 'winId'):
            raise RuntimeError(
                'This ModeShapePlotPVQt renders off-screen into a '
                'pyvista.Plotter and has no Qt widget. Construct it with '
                'off_screen=False to obtain one.')
        return self.plotter

    def _add_actors(self):
        '''Create one actor per mesh and connect them to the warp pipelines.

        The mappers are attached to the filters' output *ports* rather than
        to a snapshot of their data, so re-executing the pipeline after a
        scale-factor change is enough to move the actors.
        '''
        pv = self._pv

        style = {
            'beams': dict(color=self.beamcolor, line_width=self._line_width(),
                          render_lines_as_tubes=False),
            'nodes': dict(color=self.nodecolor, point_size=self.nodesize ** 0.5 * 2,
                          render_points_as_spheres=True, style='points'),
            'cn_lines': dict(color='lightgrey', line_width=1),
            'surfaces': dict(color='tan', opacity=0.35, show_edges=False),
        }

        for key, warp in self.warps.items():
            actor = self.plotter.add_mesh(pv.wrap(warp.output),
                                          **style.get(key, {}))
            actor.GetMapper().SetInputConnection(warp.output_port)
            self.actors[key] = actor

        self.actors['nd_lines'] = self.plotter.add_mesh(
            self.nd_mesh, color='lightgrey', line_width=1, style='wireframe')

        for name, arrow in self._axis_arrows().items():
            self.actors[f'axis_{name}'] = self.plotter.add_mesh(
                arrow, color={'X': 'r', 'Y': 'g', 'Z': 'b'}[name])

        for i, arrow in enumerate(self._chan_dof_arrows()):
            self.actors[f'chan_dof_{i}'] = self.plotter.add_mesh(
                arrow, color='darkorange')

        for i, arrow in enumerate(self._parent_child_arrows()):
            self.actors[f'parent_child_{i}'] = self.plotter.add_mesh(
                arrow, color='purple')

        self.refresh_chan_dofs(False)
        self.refresh_parent_childs(False)

    def _line_width(self):
        '''Return a scalar line width, collapsing the per-beam sequence form.'''
        if isinstance(self.linewidth, (list, tuple, np.ndarray)):
            return float(np.mean(self.linewidth))
        return float(self.linewidth)

    def _set_visible(self, prefix, visible):
        '''Show or hide every actor whose key equals or starts with *prefix*.'''
        for key, actor in self.actors.items():
            if key == prefix or key.startswith(f'{prefix}_'):
                actor.SetVisibility(bool(visible))

    # ── Visibility toggles (mirroring the matplotlib backend's API) ──────────

    def refresh_axis(self, visible=None):
        '''Show/hide the axis triad.'''
        if visible is not None:
            self.show_axis = bool(visible)
        self._set_visible('axis', self.show_axis)
        self.render()

    def refresh_nodes(self, visible=None):
        '''Show/hide the node markers.'''
        if visible is not None:
            self.show_nodes = bool(visible)
        self._set_visible('nodes', self.show_nodes)
        self.render()

    def refresh_lines(self, visible=None):
        '''Show/hide the displaced beams.'''
        if visible is not None:
            self.show_lines = bool(visible)
        self._set_visible('beams', self.show_lines)
        self.render()

    def refresh_nd_lines(self, visible=None):
        '''Show/hide the undeformed wireframe.'''
        if visible is not None:
            self.show_nd_lines = bool(visible)
        self._set_visible('nd_lines', self.show_nd_lines)
        self.render()

    def refresh_cn_lines(self, visible=None):
        '''Show/hide the lines connecting undeformed and deformed nodes.'''
        if visible is not None:
            self.show_cn_lines = bool(visible)
        self._set_visible('cn_lines', self.show_cn_lines)
        self.render()

    def refresh_surfaces(self, visible=None):
        '''Show/hide the translucent surfaces.'''
        if visible is not None:
            self.show_surfaces = bool(visible)
        self._set_visible('surfaces', self.show_surfaces)
        self.render()

    def refresh_chan_dofs(self, visible=None):
        '''Show/hide the channel-DOF arrows.'''
        if visible is not None:
            self.show_chan_dofs = bool(visible)
        self._set_visible('chan_dof', self.show_chan_dofs)
        self.render()

    def refresh_parent_childs(self, visible=None):
        '''Show/hide the parent-child DOF arrows.'''
        if visible is not None:
            self.show_parent_childs = bool(visible)
        self._set_visible('parent_child', self.show_parent_childs)
        self.render()

    def refresh_traces(self, visible=None):
        '''Accepted for API compatibility; traces are not drawn by this backend.'''
        if visible is not None:
            self.show_traces = bool(visible)
        logger.info('Node traces are not implemented in the pyvista backend.')

    # ── Camera ───────────────────────────────────────────────────────────────

    def reset_view(self):
        '''Stop any animation, restore the isometric view and redraw the mode.'''
        self.stop_ani()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        if self.mode_index is not None:
            self.draw_msh()
        else:
            self.render()

    #: Named viewport -> the ``QtInteractor`` method that applies it.
    _NAMED_VIEWPORTS = {'X': 'view_yz', 'Y': 'view_xz', 'Z': 'view_xy',
                        'ISO': 'view_isometric'}

    def change_viewport(self, viewport=None):
        '''Set a named viewport.

        Parameters
        ----------
        viewport : {'X', 'Y', 'Z', 'ISO'}, optional
        '''
        if viewport is None:
            return
        method = self._NAMED_VIEWPORTS.get(str(viewport).upper())
        if method is None:
            logger.warning(f'Unknown viewport {viewport}.')
            return
        getattr(self.plotter, method)()
        self.plotter.reset_camera()
        self.render()

    def save_plot(self, path=None):
        '''Save a screenshot of the current view to *path*.'''
        if path is None:
            return
        self.plotter.screenshot(str(path))

    def render(self):
        '''Redraw the scene.'''
        self.plotter.render()

    # ── Animation ────────────────────────────────────────────────────────────

    def animate(self):
        '''Start (or stop, if already running) the harmonic mode animation.

        Frames are produced by a ``QTimer`` that only assigns the two warp
        scale factors; the point math stays inside VTK.
        '''
        if self.animated:
            return self.stop_ani()

        if not hasattr(self.plotter, 'add_callback'):
            raise RuntimeError(
                'Timer-driven animation needs a pyvistaqt.QtInteractor and a '
                'running Qt event loop. This backend was built off-screen; '
                'step the animation with set_phase(t) instead.')

        self.animated = True
        self._frame = self.seq_num
        self._timer = self.plotter.add_callback(
            self._advance_frame, interval=_QT_INTERVAL_MS)

    def _advance_frame(self):
        '''Advance the animation by one frame (``QTimer`` callback).'''
        self._frame += 1
        self.seq_num = self._frame
        self.set_phase(self._frame / _QT_FRAMES_PER_CYCLE)
        self._maybe_save_animation_frame(self._frame)
        self.render()

    def _maybe_save_animation_frame(self, num):
        '''Write the frame to ``save_ani_path`` if one was configured.'''
        if self.save_ani_path and num <= _QT_FRAMES_PER_CYCLE:
            self.plotter.screenshot(str(
                self.save_ani_path
                / f'{self.select_modes.index(self.mode_index)}'
                / f'ani_{num}.png'))

    def stop_ani(self):
        '''Stop the animation and restore the still plot.'''
        if not self.animated:
            return
        self.animated = False
        if hasattr(self.plotter, 'clear_callbacks'):
            self.plotter.clear_callbacks()
        self._timer = None
        self.seq_num = 0
        self.set_phase(0.0)
        self.render()

    def close(self):
        '''Release the render window.'''
        self.stop_ani()
        self.plotter.close()


class ModeShapePlotPVJupyter(_PyVistaModeShapeBase):
    """pyvista mode-shape backend for Jupyter notebooks.

    Client-side rendering serialises the scene to the browser once, and
    server-side VTK filters do not run there, so this backend cannot use
    the warp chain that :class:`ModeShapePlotPVQt` relies on.  It
    precomputes the point positions of one full harmonic cycle instead
    and cycles through them.

    Two ways to view the result are offered, and they differ in where the
    animation loop runs:

    * :attr:`widget` returns an :mod:`ipywidgets` ``Play``/slider next to
      a client-side pyvista view.  Camera interaction is client-side, but
      each *frame* is applied by a kernel callback that re-pushes the
      geometry, so frames cost a server round-trip.  This is unavoidable:
      ``PyVistaLocalView.update`` re-serialises the scene and publishes
      it over the trame protocol.
    * :meth:`export_html` writes a self-contained page with every frame
      embedded and the loop running in JavaScript.  No kernel is involved
      and no round-trip occurs, at the cost of a fixed camera model and a
      wireframe-only rendering.

    Parameters
    ----------
    geometry_data : PreProcessingTools.GeometryProcessor
        Geometry to display.
    n_frames : int, optional
        Number of precomputed frames per harmonic cycle.  Default is 32.
    off_screen : bool, optional
        Build the plotter off-screen.  Default is *False*.
    **kwargs
        Passed to :class:`_PyVistaModeShapeBase`.

    Examples
    --------
    >>> msh = ModeShapePlotPVJupyter(geometry_data, stabil_calc, modal_data)  # doctest: +SKIP
    >>> msh.change_mode(index=0)                                             # doctest: +SKIP
    >>> msh.widget                                                           # doctest: +SKIP
    """

    #: Client-side rendering cannot execute server-side VTK filters.
    _USES_WARP = False

    def __init__(self, geometry_data, *args, n_frames=32, off_screen=False,
                 **kwargs):
        super().__init__(geometry_data, *args, **kwargs)

        if not isinstance(n_frames, int) or n_frames < 2:
            raise ValueError(
                f'n_frames must be an integer >= 2, got {n_frames!r}.')
        self.n_frames = n_frames

        self.plotter = self._pv.Plotter(notebook=True, off_screen=off_screen)
        self._add_actors()
        self.plotter.view_isometric()
        self.plotter.reset_camera()

        self._frames = None
        self.compute_frames()

    # ── Actors ───────────────────────────────────────────────────────────────

    def _add_actors(self):
        '''Add one actor per mesh, bound directly to the mesh points.

        Unlike the Qt backend there is no warp pipeline to connect to;
        animation replaces the point coordinates in place.
        '''
        style = {
            'beams': dict(color=self.beamcolor, line_width=2),
            'nodes': dict(color=self.nodecolor, point_size=6,
                          render_points_as_spheres=True, style='points'),
            'cn_lines': dict(color='lightgrey', line_width=1),
            'surfaces': dict(color='tan', opacity=0.35),
        }
        for key, mesh in self.meshes.items():
            self.actors[key] = self.plotter.add_mesh(mesh, **style.get(key, {}))

        self.actors['nd_lines'] = self.plotter.add_mesh(
            self.nd_mesh, color='lightgrey', line_width=1, style='wireframe')

    # ── Precomputed cycle ────────────────────────────────────────────────────

    def compute_frames(self):
        '''Precompute the node positions of one full harmonic cycle.

        Returns
        -------
        ndarray, shape (n_frames, n_nodes, 3)
            Frame *i* holds the positions at ``t = i / n_frames``.
        '''
        self._frames = np.stack(
            [self.mode_points(i / self.n_frames) for i in range(self.n_frames)])
        return self._frames

    @property
    def frames(self):
        '''ndarray : the precomputed cycle, shape ``(n_frames, n_nodes, 3)``.'''
        if self._frames is None:
            self.compute_frames()
        return self._frames

    def show_frame(self, index):
        '''Move every mesh to precomputed frame *index*.

        Parameters
        ----------
        index : int
            Frame number; taken modulo :attr:`n_frames`.
        '''
        points = self.frames[int(index) % self.n_frames]
        for key, mesh in self.meshes.items():
            if key == 'cn_lines':
                mesh.points = np.vstack([self.base_points, points])
            else:
                mesh.points = points
        self.render()

    def draw_msh(self):
        '''Recompute the current mode, refresh the cycle and show frame 0.'''
        self.compute_mode_displacements()
        self._update_mode_arrays()
        self.compute_frames()
        self.show_frame(0)

    def render(self):
        '''Push the current geometry to the client-side view, if one exists.

        This is the round-trip: :meth:`pyvista.trame.ui.Viewer.update`
        re-serialises the scene and publishes it over the trame protocol.
        Before :attr:`widget` has been requested there is no view to push
        to and this is a no-op.
        '''
        viewer = getattr(self, '_viewer', None)
        if viewer is None:
            return
        try:
            viewer.update()
        except Exception as e:  # the trame server may not be running yet
            logger.debug(f'Could not push geometry to the trame view: {e!r}')

    # ── Widget ───────────────────────────────────────────────────────────────

    @property
    def widget(self):
        '''ipywidgets.Widget : a client-side view with Play/slider controls.

        The returned container holds the pyvista client-side view and an
        :mod:`ipywidgets` ``Play``/``IntSlider`` pair driving
        :meth:`show_frame`.

        Notes
        -----
        Camera interaction runs in the browser, but every frame change is
        applied by this kernel and re-pushed to the client.  For an
        animation that runs without the kernel, use :meth:`export_html`.
        '''
        try:
            import ipywidgets
        except ImportError as e:
            raise ImportError(
                "The notebook backend's widget requires ipywidgets; install "
                "it with 'pip install pyOMA[jupyter]'.") from e

        from pyvista.trame.ui import get_viewer

        self._view = self.plotter.show(jupyter_backend='client', return_viewer=True)
        # show() hands back an iframe wrapper; the object that can push
        # geometry to the client is the plotter's trame viewer.
        self._viewer = get_viewer(self.plotter)

        play = ipywidgets.Play(value=0, min=0, max=self.n_frames - 1, step=1,
                               interval=1000 // self.n_frames,
                               description='Animate')
        slider = ipywidgets.IntSlider(value=0, min=0, max=self.n_frames - 1,
                                      description='Frame')
        ipywidgets.jslink((play, 'value'), (slider, 'value'))
        slider.observe(lambda change: self.show_frame(change['new']),
                       names='value')

        return ipywidgets.VBox([self._view,
                                ipywidgets.HBox([play, slider])])

    # ── Self-contained export ────────────────────────────────────────────────

    def export_html(self, path, width=800, height=600):
        '''Write a self-contained animated page of the precomputed cycle.

        Every frame is embedded in the file and the animation loop runs in
        JavaScript, so the page needs neither a kernel nor a network
        connection.  Rendering is a wireframe of the beams, the nodes and
        the undeformed geometry, drawn on a canvas with mouse-drag
        rotation; the translucent surfaces of the Qt backend are not
        reproduced.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write.
        width, height : int, optional
            Canvas size in pixels.

        Returns
        -------
        pathlib.Path
            The path written.
        '''
        import json
        from pathlib import Path

        cells = self._line_cells()
        segments = (cells.reshape(-1, 3)[:, 1:].tolist() if cells.size else [])

        payload = {
            'frames': np.round(self.frames, 6).tolist(),
            'base': np.round(self.base_points, 6).tolist(),
            'segments': segments,
            'interval': 1000 // self.n_frames,
        }

        path = Path(path)
        path.write_text(
            _HTML_TEMPLATE
            .replace('__WIDTH__', str(int(width)))
            .replace('__HEIGHT__', str(int(height)))
            .replace('__SETUP__', str(self.setup_name or 'mode shape'))
            .replace('__DATA__', json.dumps(payload)),
            encoding='utf-8')
        return path


#: Self-contained page used by :meth:`ModeShapePlotPVJupyter.export_html`.
#: The frames are embedded and the loop is client-side, so no server is
#: contacted; the renderer is deliberately dependency-free.
_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pyOMA - __SETUP__</title>
<style>
 body{margin:0;font:13px sans-serif;background:#fff;color:#222}
 #wrap{padding:8px} canvas{border:1px solid #ddd;cursor:grab;touch-action:none}
 #bar{margin-top:6px;display:flex;gap:10px;align-items:center}
</style></head><body><div id="wrap">
<canvas id="c" width="__WIDTH__" height="__HEIGHT__"></canvas>
<div id="bar">
 <button id="play">Pause</button>
 <input id="frame" type="range" min="0" value="0" style="flex:1">
 <span id="lbl"></span>
</div></div>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), cx = cv.getContext('2d');
const slider = document.getElementById('frame'), lbl = document.getElementById('lbl');
const btn = document.getElementById('play');
slider.max = D.frames.length - 1;

// Centre and scale so the whole cycle stays inside the canvas.
let lo = [1e30,1e30,1e30], hi = [-1e30,-1e30,-1e30];
for (const f of D.frames) for (const p of f) for (let k=0;k<3;k++){
  if (p[k]<lo[k]) lo[k]=p[k]; if (p[k]>hi[k]) hi[k]=p[k]; }
const mid = [0,1,2].map(k => (lo[k]+hi[k])/2);
const span = Math.max(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]) || 1;

let yaw = -0.6, pitch = 0.5, zoom = 0.75, frame = 0, playing = true;

function project(p){
  const x = p[0]-mid[0], y = p[1]-mid[1], z = p[2]-mid[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const X = cy*x + sy*y;               // rotate about the vertical axis
  const Y = -sy*x + cy*y;
  const Z = z;
  const sx = X;                        // then tilt towards the viewer
  const sy2 = -(cp*Z - sp*Y);
  const s = zoom * Math.min(cv.width, cv.height) / span;
  return [cv.width/2 + sx*s, cv.height/2 + sy2*s];
}

function strokeSegments(pts, style, w){
  cx.strokeStyle = style; cx.lineWidth = w; cx.beginPath();
  for (const [a,b] of D.segments){
    const p = project(pts[a]), q = project(pts[b]);
    cx.moveTo(p[0], p[1]); cx.lineTo(q[0], q[1]);
  }
  cx.stroke();
}

function draw(){
  cx.clearRect(0,0,cv.width,cv.height);
  strokeSegments(D.base, '#d5d5d5', 1);          // undeformed reference
  const pts = D.frames[frame];
  strokeSegments(pts, '#444', 1.6);              // deformed shape
  cx.fillStyle = '#333';
  for (let i=0;i<pts.length;i++){
    const p = project(pts[i]), q = project(D.base[i]);
    cx.strokeStyle = '#e2e2e2'; cx.lineWidth = 1;
    cx.beginPath(); cx.moveTo(q[0],q[1]); cx.lineTo(p[0],p[1]); cx.stroke();
    cx.beginPath(); cx.arc(p[0], p[1], 2.5, 0, 6.2832); cx.fill();
  }
  lbl.textContent = 'frame ' + (frame+1) + ' / ' + D.frames.length;
  slider.value = frame;
}

setInterval(() => { if (playing){ frame = (frame+1) % D.frames.length; draw(); } },
            D.interval);
btn.onclick = () => { playing = !playing; btn.textContent = playing?'Pause':'Play'; };
slider.oninput = () => { playing = false; btn.textContent='Play';
                         frame = +slider.value; draw(); };

let drag = null;
cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY];
                                          cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointerup',   () => { drag = null; });
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  yaw   += (e.clientX - drag[0]) * 0.01;
  pitch += (e.clientY - drag[1]) * 0.01;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  drag = [e.clientX, e.clientY]; draw();
});
cv.addEventListener('wheel', e => { e.preventDefault();
  zoom *= e.deltaY < 0 ? 1.1 : 0.9; draw(); }, {passive:false});

draw();
</script></body></html>
"""
