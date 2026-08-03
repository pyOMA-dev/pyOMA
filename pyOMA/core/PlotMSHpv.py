# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2026  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
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

import os
import sys

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


def _ensure_qt_embeddable_platform():
    '''Steer Qt onto an X platform, or explain why the Qt backend cannot run.

    pyvistaqt embeds VTK into a native widget through
    ``QVTKRenderWindowInteractor``, which needs a real X window.  Under the
    Qt ``wayland`` platform plugin the widget's native handle is a Wayland
    surface, so VTK's X render window (:class:`vtkXOpenGLRenderWindow`)
    aborts the whole process with an uncatchable ``BadWindow`` X error on
    ``X_ConfigureWindow`` the moment the interactor is created.  Xwayland's
    ``xcb`` plugin provides a real X window and works.

    Two situations arise, because the platform is fixed once the
    ``QApplication`` is constructed:

    * No ``QApplication`` exists yet -- request ``xcb`` for the process,
      unless the user pinned ``QT_QPA_PLATFORM`` themselves.
    * A ``wayland`` ``QApplication`` is already running -- it is too late to
      switch, so raise an actionable error rather than crash.

    Only Linux is affected; other platforms return immediately.
    '''
    if sys.platform != 'linux':
        return
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return

    app = QApplication.instance()
    if app is None:
        wants_override = (os.environ.get('WAYLAND_DISPLAY')
                          and os.environ.get('DISPLAY')
                          and not os.environ.get('QT_QPA_PLATFORM'))
        if wants_override:
            logger.info(
                "Setting QT_QPA_PLATFORM=xcb: pyvistaqt needs an X window, "
                "which the Qt 'wayland' platform plugin does not provide.")
            os.environ['QT_QPA_PLATFORM'] = 'xcb'
        return

    if app.platformName().startswith('wayland'):
        raise RuntimeError(
            "ModeShapePlotPVQt embeds VTK through pyvistaqt, which needs an X "
            "window, but the running Qt application already uses the 'wayland' "
            "platform plugin -- VTK would abort the process with a BadWindow X "
            "error. Set QT_QPA_PLATFORM=xcb before the first QApplication is "
            "created (e.g. at the very top of your script, before importing "
            "pyOMA) so Qt runs through Xwayland.")


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

        # Surfaces are not warp-driven: they carry a scalar field ('displacement')
        # that must stay the active point scalars, and the warp's mode_re/mode_im
        # arrays would take that slot.  They are animated by re-setting their
        # points and scalar directly (see _update_surface).
        self.warps = {
            key: _WarpChain(mesh, self._vtk_warp_vector, self._vtk_data_object)
            for key, mesh in meshes.items()
            if key != 'surfaces'} if self._USES_WARP else {}

    def _update_mode_arrays(self):
        """Write the current mode into the ``mode_re``/``mode_im`` point arrays.

        Skips the surface mesh, whose active point scalars are reserved for the
        displacement-magnitude colouring (see :meth:`_update_surface`).
        """
        mode_re, mode_im = self._mode_vectors()
        zeros = np.zeros_like(mode_re)

        for key, mesh in self.meshes.items():
            if key == 'surfaces':
                continue
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

    # ── Text-label anchors (mirror the arrow generators above) ───────────────

    def _axis_labels(self):
        """Return ``(points, texts)`` for the axis-triad labels X/Y/Z."""
        length = self._arrow_length()
        directions = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}
        points = [np.array(d, dtype=float) * length for d in directions.values()]
        return points, list(directions.keys())

    def _chan_dof_labels(self):
        """Return ``(points, texts)`` for channel-DOF labels (channel names).

        Uses the same node resolution and skip logic as
        :meth:`_chan_dof_arrows`, and anchors each label at that arrow's *tip*
        (start + unit direction * length) so the name sits at the arrow head
        rather than on the node, matching the matplotlib backend.
        """
        length = self._arrow_length()
        points, texts = [], []
        for chan_dof in self.chan_dofs:
            _chan, node, az, elev = chan_dof[0:4]
            chan_name = chan_dof[-1]
            node_id = self.node_ids.get(str(node))
            if node_id is None:
                continue
            direction = calc_xyz(az * np.pi / 180, elev * np.pi / 180, r=1)
            points.append(self.base_points[node_id] + np.asarray(direction) * length)
            texts.append(str(chan_name))
        return points, texts

    def _parent_child_labels(self):
        """Return ``(points, texts)`` for parent-child labels (node names).

        Anchored at each arrow's tip, mirroring :meth:`_parent_child_arrows`
        (same ``0.8`` length factor and direction normalisation).
        """
        length = self._arrow_length() * 0.8
        points, texts = [], []
        for parent, x_m, y_m, z_m, child, x_sl, y_sl, z_sl in \
                self.geometry_data.parent_childs:
            for node, direction in ((parent, (x_m, y_m, z_m)),
                                    (child, (x_sl, y_sl, z_sl))):
                node_id = self.node_ids.get(str(node))
                if node_id is None or np.allclose(direction, 0):
                    continue
                unit = np.asarray(direction, dtype=float)
                unit = unit / np.linalg.norm(unit)
                points.append(self.base_points[node_id] + unit * length)
                texts.append(str(node))
        return points, texts

    # ── 3-D text glyphs (for backends that cannot render 2-D label actors) ────

    def _label_height(self):
        """Cap height for 3-D text labels: ~4 % of the model's largest extent."""
        xmin, xmax, ymin, ymax, zmin, zmax = self._compute_node_bounds()
        return 0.04 * max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-9)

    def _text3d_mesh(self, points, texts, height=None, normal=(1, 1, 1)):
        """Merge one 3-D text glyph per ``(point, text)`` into a single PolyData.

        Each glyph is a :func:`pyvista.Text3D` of cap height *height* (default
        :meth:`_label_height`), centred on its point and lying in the plane
        whose normal is *normal* -- by default ``(1, 1, 1)``, so the text faces
        the default isometric camera.  Unlike the 2-D ``add_point_labels``
        actors, real polydata serialises to vtk.js, so the client-side notebook
        view can render it; the trade-off is that the text does not billboard
        when the camera is rotated away from *normal*.

        Returns *None* if there is nothing to draw.
        """
        pv = self._pv
        points = np.asarray(points, dtype=float)
        if not len(points) or not len(texts):
            return None
        if height is None:
            height = self._label_height()

        glyphs = []
        for point, text in zip(points, texts):
            label = str(text)
            if not label:
                continue
            glyphs.append(pv.Text3D(label, depth=0.0, height=height,
                                    center=point, normal=normal))
        if not glyphs:
            return None
        return glyphs[0] if len(glyphs) == 1 else pv.merge(glyphs)


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
        self._view_limits = None

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

    def redraw(self):
        '''Repaint the current scene (neutral-contract entry point).'''
        self.render()

    # ── Static overlays & visibility (shared by both pyvista backends) ───────
    #
    # Arrows and text labels sit on the *undeformed* geometry and are identical
    # for the Qt and Jupyter backends; only the deformable meshes differ (warp
    # pipeline vs. precomputed frame swaps).  Each backend's ``_add_actors``
    # therefore adds just its deformable meshes and then calls
    # :meth:`_add_static_overlays`.

    #: Axis-triad arrow colours by name.
    _AXIS_ARROW_COLORS = {'X': 'r', 'Y': 'g', 'Z': 'b'}

    def _add_static_overlays(self):
        '''Add the undeformed wireframe, the axis triad, the channel-DOF and
        parent-child arrows, and all text labels (node names, axis, arrows).'''
        self.actors['nd_lines'] = self.plotter.add_mesh(
            self.nd_mesh, color='lightgrey', line_width=1, style='wireframe')

        for name, arrow in self._axis_arrows().items():
            self.actors[f'axis_{name}'] = self.plotter.add_mesh(
                arrow, color=self._AXIS_ARROW_COLORS[name])

        for i, arrow in enumerate(self._chan_dof_arrows()):
            self.actors[f'chan_dof_{i}'] = self.plotter.add_mesh(
                arrow, color='darkorange')

        for i, arrow in enumerate(self._parent_child_arrows()):
            self.actors[f'parent_child_{i}'] = self.plotter.add_mesh(
                arrow, color='purple')

        # Text labels. Each is keyed with its arrows' prefix so the
        # prefix-based _set_visible toggles labels together with the arrows;
        # node labels follow 'Show nodes'. Mirrors the matplotlib backend.
        self._add_label_actor('axis_labels', *self._axis_labels(), color='black')
        self._add_label_actor('chan_dof_labels', *self._chan_dof_labels(),
                              color='darkorange')
        self._add_label_actor('parent_child_labels', *self._parent_child_labels(),
                              color='purple')
        self._place_node_labels(self.base_points)

        self.refresh_chan_dofs(False)
        self.refresh_parent_childs(False)

    def _add_label_actor(self, key, points, texts, color):
        '''Add (or replace) a point-labels actor stored under *key*.'''
        old = self.actors.pop(key, None)
        if old is not None:
            try:
                self.plotter.remove_actor(old)
            except Exception:  # pragma: no cover - actor already gone
                pass
        if not len(points):
            return
        self.actors[key] = self.plotter.add_point_labels(
            np.asarray(points, dtype=float), list(texts),
            show_points=False, font_size=12, text_color=color, shape=None,
            always_visible=True, reset_camera=False, render=False)

    def _place_node_labels(self, points):
        '''Anchor the node-name labels at *points*, matching 'Show nodes'.'''
        self._add_label_actor('nodes_labels', points,
                              [str(name) for name in self.node_names],
                              color='black')
        actor = self.actors.get('nodes_labels')
        if actor is not None:
            actor.SetVisibility(bool(self.show_nodes))

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

    # ── Surface displacement colouring (shared) ──────────────────────────────
    #
    # Surfaces are the only mesh with faces, so they are the natural place to
    # show a scalar field.  We colour them by the per-node modal-displacement
    # magnitude, a native pyvista scalar/colormap, and update it every frame.

    #: Name of the per-node scalar array used to colour the surfaces.
    _SURFACE_ARRAY = 'displacement'
    #: Colormap for the surface displacement scalar.
    _SURFACE_CMAP = 'viridis'
    #: Surface opacity.  Fully opaque: vtk.js (the notebook backend) has no
    #: order-independent transparency, so a translucent surface blends with the
    #: light background and the viridis colours wash out to a flat grey.  Toggle
    #: the surface off (Show surfaces) to see the wireframe/nodes behind it.
    _SURFACE_OPACITY = 1.0

    def _surface_scalar_env(self):
        '''Per-node modal-displacement *envelope* magnitude sqrt(re^2+im^2).

        Phase-independent; used for the still view and to set the colour range.
        '''
        mode_re, mode_im = self._mode_vectors()
        return np.sqrt((mode_re ** 2 + mode_im ** 2).sum(axis=1))

    def _surface_scalar_at(self, t):
        '''Per-node *instantaneous* displacement magnitude at cycle fraction *t*.'''
        mode_re, mode_im = self._mode_vectors()
        inst = mode_re * np.cos(2 * np.pi * t) - mode_im * np.sin(2 * np.pi * t)
        return np.sqrt((inst ** 2).sum(axis=1))

    def _surface_clim(self):
        '''Colour limits ``[0, max envelope]`` for the surface scalar.'''
        env = self._surface_scalar_env()
        top = float(env.max()) if env.size else 0.0
        return [0.0, top if top > 1e-12 else 1.0]

    def _update_surface(self, t):
        '''Warp the surface mesh to cycle fraction *t* and recolour it.

        The surface is not part of the warp pipeline (see :meth:`_build_meshes`),
        so its points are moved here directly, and the instantaneous
        displacement magnitude is written as the *active* point scalars so the
        mapper (which colours by the active array) shows it.  No-op without a
        surface.
        '''
        mesh = self.meshes.get('surfaces')
        if mesh is None:
            return
        mesh.points = self.mode_points(t)
        mesh.point_data[self._SURFACE_ARRAY] = self._surface_scalar_at(t)
        mesh.set_active_scalars(self._SURFACE_ARRAY)

    def _add_surface_actor(self, mesh):
        '''Add the translucent surface actor coloured by displacement magnitude.

        Colours by the ``displacement`` point scalar (kept as the mesh's active
        array); the surface is animated by :meth:`_update_surface`, not the warp
        pipeline, so no filter connection is needed.
        '''
        mesh.point_data[self._SURFACE_ARRAY] = self._surface_scalar_env()
        mesh.set_active_scalars(self._SURFACE_ARRAY)
        return self.plotter.add_mesh(
            mesh, scalars=self._SURFACE_ARRAY, cmap=self._SURFACE_CMAP,
            clim=self._surface_clim(), opacity=self._SURFACE_OPACITY,
            show_edges=False, scalar_bar_args={'title': 'Displacement'})

    def _refresh_surface_clim(self):
        '''Re-range the surface colour map after a mode change.'''
        actor = self.actors.get('surfaces')
        if actor is None:
            return
        try:
            actor.mapper.scalar_range = self._surface_clim()
        except Exception:  # pragma: no cover - VTK version differences
            pass

    def set_phase(self, t):
        '''Move the deformable meshes to cycle fraction *t* and recolour surfaces.'''
        super().set_phase(t)
        self._update_surface(t)

    # ── Animation frame export ───────────────────────────────────────────────

    #: Per-frame export formats: PNG (raster screenshot) and PDF (vector).
    _EXPORT_FORMATS = ('png', 'pdf')

    def _write_frame(self, path, fmt):
        '''Write the current view to *path* as PNG (raster) or PDF (vector).'''
        if fmt == 'png':
            self.plotter.screenshot(str(path))
        else:
            self.plotter.save_graphic(str(path))

    def export_animation_frames(self, directory, fmt='png', n_frames=None):
        '''Write one image per animation frame of the current mode to *directory*.

        Renders a full harmonic cycle as *n_frames* numbered files
        (``ani_000.<fmt>`` ...), then restores the still view.

        Parameters
        ----------
        directory : str or pathlib.Path
            Output folder; created if missing.
        fmt : {'png', 'pdf'}, optional
            PNG (raster screenshot) or PDF (vector, via ``save_graphic``).
        n_frames : int, optional
            Frames per cycle; defaults to the backend's own frame count.

        Returns
        -------
        list of pathlib.Path
        '''
        from pathlib import Path
        fmt = str(fmt).lower().lstrip('.')
        if fmt not in self._EXPORT_FORMATS:
            raise ValueError(
                f"fmt must be one of {self._EXPORT_FORMATS}, got {fmt!r}.")
        if n_frames is None:
            n_frames = getattr(self, 'n_frames', _QT_FRAMES_PER_CYCLE)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        paths = []
        for i in range(int(n_frames)):
            self.set_phase(i / n_frames)
            self.render()
            path = directory / f'ani_{i:03d}.{fmt}'
            self._write_frame(path, fmt)
            paths.append(path)
        self.set_phase(0.0)
        self.render()
        return paths

    # ── Camera (shared: both backends have a pyvista plotter camera) ─────────

    def get_view_angles(self):
        '''Return the camera orientation as ``(elev, azim, roll)`` in degrees.

        Derived from the camera position relative to its focal point.  A VTK
        camera and a matplotlib ``Axes3D`` do not share an exact
        elevation/azimuth/roll convention, so these are meant to be read and
        edited (populating the GUI's angle fields), not matched bit-for-bit.
        '''
        cam = self.plotter.camera
        px, py, pz = cam.position
        fx, fy, fz = cam.focal_point
        dx, dy, dz = px - fx, py - fy, pz - fz
        r = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if r == 0:
            return 0.0, 0.0, 0.0
        elev = float(np.degrees(np.arcsin(np.clip(dz / r, -1.0, 1.0))))
        azim = float(np.degrees(np.arctan2(dy, dx)))
        return elev, azim, float(cam.roll)

    def _set_camera_angles(self, elev, azim, roll):
        '''Orient the camera from ``(elev, azim, roll)`` in degrees.

        The inverse of :meth:`get_view_angles`: keep the camera-to-focal
        distance, place the camera at the requested elevation/azimuth on that
        sphere, then apply roll.
        '''
        cam = self.plotter.camera
        fx, fy, fz = cam.focal_point
        px, py, pz = cam.position
        dist = float(np.sqrt((px - fx) ** 2 + (py - fy) ** 2
                             + (pz - fz) ** 2)) or 1.0
        e, a = np.radians(float(elev)), np.radians(float(azim))
        cam.up = (0.0, 0.0, 1.0) if abs(float(elev)) < 89.9 else (0.0, 1.0, 0.0)
        cam.position = (fx + dist * np.cos(e) * np.cos(a),
                        fy + dist * np.cos(e) * np.sin(a),
                        fz + dist * np.sin(e))
        cam.focal_point = (fx, fy, fz)
        cam.roll = float(roll)
        self.plotter.reset_camera_clipping_range()
        self.render()

    #: Named viewport -> the pyvista Plotter method that applies it.
    _NAMED_VIEWPORTS = {'X': 'view_yz', 'Y': 'view_xz', 'Z': 'view_xy',
                        'ISO': 'view_isometric'}

    def change_viewport(self, viewport=None):
        '''Set a named viewport or an ``(elev, azim, roll)`` orientation.

        Parameters
        ----------
        viewport : {'X', 'Y', 'Z', 'ISO'} or sequence of float, optional
            A named viewport, or an ``(elev, azim, roll)`` triple in degrees
            (as the GUI sends when the angle fields are edited).
        '''
        if viewport is None:
            return
        if isinstance(viewport, (list, tuple, np.ndarray)):
            self._set_camera_angles(*viewport)
            return
        method = self._NAMED_VIEWPORTS.get(str(viewport).upper())
        if method is None:
            logger.warning(f'Unknown viewport {viewport}.')
            return
        getattr(self.plotter, method)()
        self.plotter.reset_camera()
        self.render()


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

    #: A VTK camera cannot clip to per-axis data limits, but the GUI's limit
    #: fields and zoom buttons are honoured by framing the camera on the
    #: requested box; see :meth:`set_view_limits`.
    supports_axis_limits = True

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
        if self.is_interactive:  # a Qt widget draws itself
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
        _ensure_qt_embeddable_platform()
        try:
            from pyvistaqt import QtInteractor
        except ImportError as e:
            raise ImportError(f'{_INSTALL_HINT} (missing: {e.name})') from e
        # auto_update=False disables QtInteractor's own 5 Hz render timer.
        # This backend drives rendering itself, and leaving both timers
        # running races two render passes against a VTK pipeline that
        # ``set_phase`` is re-executing, which segfaults after a few
        # hundred animation frames.
        return QtInteractor(auto_update=False)

    # ── Widget & actors ──────────────────────────────────────────────────────

    @property
    def is_interactive(self):
        '''bool : whether the plotter is a Qt widget rather than off-screen.

        ``QWidget`` has ``winId()`` and :class:`pyvista.Plotter` does not,
        so this answers the question without importing Qt.
        '''
        return hasattr(self.plotter, 'winId')

    @property
    def widget(self):
        '''pyvistaqt.QtInteractor : the ``QWidget`` hosting the rendering.

        Raises
        ------
        RuntimeError
            If this backend was built off-screen, where no Qt widget exists.
        '''
        if not self.is_interactive:
            raise RuntimeError(
                'This ModeShapePlotPVQt renders off-screen into a '
                'pyvista.Plotter and has no Qt widget. Construct it with '
                'off_screen=False to obtain one.')
        return self.plotter

    def _add_actors(self):
        '''Create the warp-connected deformable-mesh actors, then the static
        overlays (arrows, labels, undeformed wireframe).

        The mappers are attached to the filters' output *ports* rather than
        to a snapshot of their data, so re-executing the pipeline after a
        scale-factor change is enough to move the actors.  The static overlays
        are added by the shared :meth:`_add_static_overlays`.
        '''
        pv = self._pv

        style = {
            'beams': dict(color=self.beamcolor, line_width=self._line_width(),
                          render_lines_as_tubes=False),
            # render_points_as_spheres uses a point-sprite impostor shader that
            # silently renders nothing on several Mesa/VTK builds (verified on
            # Mesa llvmpipe and Mesa-on-NVIDIA here), so the nodes vanish. Flat
            # points render everywhere and still warp with the mode.
            'nodes': dict(color=self.nodecolor, point_size=self.nodesize ** 0.5 * 2,
                          render_points_as_spheres=False, style='points'),
            'cn_lines': dict(color='lightgrey', line_width=1),
        }

        for key, warp in self.warps.items():
            actor = self.plotter.add_mesh(pv.wrap(warp.output),
                                          **style.get(key, {}))
            actor.GetMapper().SetInputConnection(warp.output_port)
            self.actors[key] = actor

        # Surfaces are not warp-driven (their active scalars carry the
        # displacement colouring); animated directly via _update_surface.
        if 'surfaces' in self.meshes:
            self.actors['surfaces'] = self._add_surface_actor(self.meshes['surfaces'])

        self._add_static_overlays()

    def _line_width(self):
        '''Return a scalar line width, collapsing the per-beam sequence form.'''
        if isinstance(self.linewidth, (list, tuple, np.ndarray)):
            return float(np.mean(self.linewidth))
        return float(self.linewidth)

    # ── Camera ───────────────────────────────────────────────────────────────

    def reset_view(self):
        '''Stop any animation, restore the isometric view and redraw the mode.'''
        self.stop_ani()
        self._view_limits = None
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        if self.mode_index is not None:
            self.draw_msh()
        else:
            self.render()

    def draw_msh(self):
        '''Draw the mode, then re-anchor node labels to the rest positions.

        The node markers warp with the mode, so the static node-name labels
        are re-placed at the current (phase-0) node positions to stay on
        their markers, mirroring how the matplotlib backend redraws node text.
        '''
        super().draw_msh()
        self._refresh_surface_clim()
        self._place_node_labels(self.mode_points(0.0))
        self.render()

    def rebuild_geometry(self):
        '''Rebuild meshes and actors from the (possibly edited) geometry_data.

        Re-runs the node index, meshes, warp chains and actors so added or
        removed nodes/lines/surfaces/parent-childs appear.  The camera is
        left where it is, so an edit does not jump the view.
        '''
        for actor in list(self.actors.values()):
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # pragma: no cover - actor already gone
                pass
        self.actors = {}
        self._build_node_index()
        self._build_meshes()
        self._add_actors()
        if self.mode_index is not None:
            self.draw_msh()
        else:
            self.render()

    def redraw_geometry(self):
        '''Neutral-contract entry point; see :meth:`ModeShapeBase.redraw_geometry`.'''
        self.rebuild_geometry()

    def draw_draft_chan_dof(self, channel, node, az, elev, chan_name):
        '''Draw one highlighted draft channel-DOF arrow and label at *node*.'''
        node_id = self.node_ids.get(str(node))
        if node_id is None:
            return
        direction = calc_xyz(az * np.pi / 180, elev * np.pi / 180, r=1)
        arrow = self._pv.Arrow(start=self.base_points[node_id],
                               direction=direction, scale=self._arrow_length())
        old = self.actors.pop('draft_chan_dof', None)
        if old is not None:
            try:
                self.plotter.remove_actor(old)
            except Exception:  # pragma: no cover - actor already gone
                pass
        self.actors['draft_chan_dof'] = self.plotter.add_mesh(arrow, color='red')
        tip = self.base_points[node_id] + np.asarray(direction) * self._arrow_length()
        self._add_label_actor('draft_chan_dof_label',
                              [tip], [str(chan_name)], color='red')
        self.render()

    def get_view_limits(self):
        '''Return the box the view frames, ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

        Defaults to the node bounding cube until :meth:`set_view_limits` is
        called; :meth:`reset_view` clears it back to that default.
        '''
        if self._view_limits is None:
            self._view_limits = list(self._compute_node_bounds())
        return tuple(self._view_limits)

    def set_view_limits(self, xmin, xmax, ymin, ymax, zmin, zmax):
        '''Frame the camera on the given box.

        A VTK camera has no per-axis data limits to *clip* to the way a
        matplotlib ``Axes3D`` does, so this frames (zooms/pans to) the box
        while keeping the current view direction rather than cropping the
        geometry outside it.  The stabilisation-diagram GUI always passes an
        equal-sided cube, which is what the matplotlib backend zooms to for
        the same controls, so the on-screen result matches.
        '''
        self._view_limits = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.plotter.reset_camera(bounds=[xmin, xmax, ymin, ymax, zmin, zmax])
        self.render()

    def save_plot(self, path=None):
        '''Save a screenshot of the current view to *path* (PNG).'''
        if path is None:
            return
        from pathlib import Path
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.png')
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

        if not self.is_interactive:
            raise RuntimeError(
                'Timer-driven animation needs a pyvistaqt.QtInteractor and a '
                'running Qt event loop. This backend was built off-screen; '
                'step the animation with set_phase(t) instead.')

        # qtpy resolves to whichever binding the host application already
        # imported, which is how pyvistaqt itself picks one.  Note that
        # ``add_callback`` lives on pyvistaqt's BackgroundPlotter, not on
        # QtInteractor, so the timer is created here rather than delegated.
        from qtpy.QtCore import QTimer

        self.animated = True
        self._frame = self.seq_num
        self._timer = QTimer(self.plotter)
        self._timer.timeout.connect(self._advance_frame)
        self._timer.start(_QT_INTERVAL_MS)

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
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
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

    Rather than the ``vtkWarpVector`` chain that :class:`ModeShapePlotPVQt`
    animates inside VTK, this backend precomputes the node positions of one
    full harmonic cycle and cycles through them by swapping the mesh points,
    which keeps the animation independent of the rendering path.

    Two ways to view the result are offered, and they differ in where the
    animation loop runs:

    * :attr:`widget` returns an :mod:`ipywidgets` ``Play``/slider next to a
      pyvista trame view.  The view renders **client-side** in the browser
      with vtk.js (:attr:`_JUPYTER_BACKEND`), so interacting with the controls
      never triggers an in-kernel VTK render (the ``'server'`` backend does,
      and crashes the kernel here -- see :attr:`_JUPYTER_BACKEND`).  The
      arrows, the colour-mapped surfaces and the scalar bar all render; text
      labels are drawn as 3-D polydata glyphs (:meth:`_add_label_actor`)
      because vtk.js does not render 2-D label actors.  Each frame is applied
      by a kernel callback that re-serialises the scene to the browser.
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

    #: Surfaces stay fully opaque (matches the base): vtk.js has no
    #: order-independent transparency, so a translucent surface would wash the
    #: viridis colouring out to a flat grey in the browser.
    _SURFACE_OPACITY = 1.0

    #: trame ``jupyter_backend`` for the interactive view.  ``'client'``
    #: (vtk.js) renders in the browser: the kernel only serialises the scene,
    #: so the ipywidgets controls never trigger an in-kernel VTK render.  The
    #: ``'server'`` backend renders each frame on the kernel and streams the
    #: image, which crashes the kernel here: a control callback runs on the
    #: kernel thread and calls ``push_image`` -> a VTK GL render that races the
    #: trame server's own render loop.  vtk.js serialises polydata actors and
    #: scalar bars but not the 2-D ``vtkLabelPlacementMapper`` behind
    #: ``add_point_labels``, so this backend draws its text as 3-D polydata
    #: glyphs instead (see :meth:`_add_label_actor`).
    _JUPYTER_BACKEND = 'client'

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
        '''Add the deformable-mesh actors bound directly to the mesh points,
        then the shared static overlays (arrows, labels, wireframe).

        Unlike the Qt backend there is no warp pipeline to connect to;
        animation replaces the point coordinates in place.
        '''
        style = {
            'beams': dict(color=self.beamcolor, line_width=2),
            # See ModeShapePlotPVQt._add_actors: render_points_as_spheres draws
            # nothing on several Mesa/VTK builds, so keep nodes as flat points.
            'nodes': dict(color=self.nodecolor, point_size=6,
                          render_points_as_spheres=False, style='points'),
            'cn_lines': dict(color='lightgrey', line_width=1),
        }
        for key, mesh in self.meshes.items():
            if key == 'surfaces':
                self.actors[key] = self._add_surface_actor(mesh)
                continue
            self.actors[key] = self.plotter.add_mesh(mesh, **style.get(key, {}))

        self._add_static_overlays()

    def _add_label_actor(self, key, points, texts, color):
        '''Render labels as 3-D polydata text instead of 2-D point labels.

        The client-side (vtk.js) view serialises ordinary polydata actors but
        not the ``vtkLabelPlacementMapper`` behind ``add_point_labels`` (see
        :attr:`_JUPYTER_BACKEND`), so node names, axis and DOF labels are drawn
        as merged :func:`pyvista.Text3D` glyphs (see :meth:`_text3d_mesh`).
        The actor is keyed identically to the Qt backend's label actors so the
        same ``_set_visible`` / ``refresh_*`` toggles hide it.
        '''
        old = self.actors.pop(key, None)
        if old is not None:
            try:
                self.plotter.remove_actor(old)
            except Exception:  # pragma: no cover - actor already gone
                pass
        mesh = self._text3d_mesh(points, texts)
        if mesh is None:
            return
        self.actors[key] = self.plotter.add_mesh(
            mesh, color=color, lighting=False)

    def reset_view(self):
        '''Restore the isometric view and redraw the current mode.'''
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        if self.mode_index is not None:
            self.draw_msh()
        else:
            self.render()

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
        index = int(index) % self.n_frames
        points = self.frames[index]
        for key, mesh in self.meshes.items():
            if key == 'surfaces':
                continue  # handled by _update_surface (points + active scalar)
            if key == 'cn_lines':
                mesh.points = np.vstack([self.base_points, points])
            else:
                mesh.points = points
        self._update_surface(index / self.n_frames)
        self.render()

    def draw_msh(self):
        '''Recompute the current mode, refresh the cycle and show frame 0.'''
        self.compute_mode_displacements()
        self._update_mode_arrays()
        self.compute_frames()
        self._refresh_surface_clim()
        self._place_node_labels(self.mode_points(0.0))
        self.show_frame(0)

    def export_animation_frames(self, directory, fmt='png', n_frames=None):
        '''Write one image per animation frame of the current mode to *directory*.

        The notebook plotter renders client-side and has no reliable
        on-demand render window to screenshot, so a throwaway off-screen
        :class:`ModeShapePlotPVQt` that shares this mode does the rasterising
        (giving the same arrows/labels/coloured surfaces as the Qt viewer).
        See :meth:`export_html` for the self-contained in-notebook alternative.
        '''
        if n_frames is None:
            n_frames = self.n_frames
        off = ModeShapePlotPVQt(
            self.geometry_data, self.stabil_calc, self.modal_data,
            prep_signals=self.prep_signals, merged_data=self.merged_data,
            off_screen=True)
        try:
            off.amplitude = self.amplitude
            off.real = self.real
            if self.mode_index is not None and self.mode_index in off.select_modes:
                off.change_mode(mode_index=self.mode_index)
            else:
                off.disp_nodes = dict(self.disp_nodes)
                off.phi_nodes = dict(self.phi_nodes)
                off._update_mode_arrays()
                off.set_phase(0.0)
            return off.export_animation_frames(directory, fmt, n_frames)
        finally:
            off.close()

    def render(self):
        '''Push the current geometry to the trame view, if one exists.

        With the client-side backend (:attr:`_JUPYTER_BACKEND`),
        :meth:`pyvista.trame.ui.Viewer.update` re-serialises the scene to the
        browser (vtk.js does the actual rendering); it does not render in the
        kernel.  Before :attr:`widget` has been requested there is no view to
        push to and this is a no-op.
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
        '''ipywidgets.Widget : an interactive view with Play/slider controls.

        The returned container holds the pyvista trame view (rendered
        client-side with vtk.js, see :attr:`_JUPYTER_BACKEND`; arrows, coloured
        surfaces, the scalar bar and 3-D text labels show) and an
        :mod:`ipywidgets` ``Play``/``IntSlider`` pair driving
        :meth:`show_frame`.

        Notes
        -----
        Every frame change is applied by this kernel and re-pushed to the
        view.  For an animation that runs without the kernel, use
        :meth:`export_html`.
        '''
        try:
            import ipywidgets
        except ImportError as e:
            raise ImportError(
                "The notebook backend's widget requires ipywidgets; install "
                "it with 'pip install pyOMA[jupyter]'.") from e

        from pyvista.trame.ui import get_viewer

        self._view = self.plotter.show(
            jupyter_backend=self._JUPYTER_BACKEND, return_viewer=True)
        # show() hands back an iframe wrapper; the object that can push
        # geometry to the view is the plotter's trame viewer.
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
