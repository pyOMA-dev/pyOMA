"""
Unit tests for the pyvista mode-shape backends in pyOMA.core.PlotMSHpv.

The tests that matter most here guard the node-name to vtk-point-id map:
``geometry_data.nodes`` is keyed by string node names while VTK addresses
points by integer id, and every mesh (beams, nodes, connecting lines,
surfaces) is built through that one map.

All rendering happens off-screen.  ``ModeShapePlotPVQt`` uses a plain
``pyvista.Plotter`` in that mode, because a ``QtInteractor`` draws into a
``QWidget``'s native surface and Qt's ``offscreen`` platform plugin
provides none, leaving VTK without a GL context.
"""
import subprocess
import sys

import numpy as np
import pytest

pv = pytest.importorskip('pyvista', reason='requires the pyOMA[pyvista] extra')

from pyOMA.core.PlotMSHpv import (  # noqa: E402
    ModeShapePlotPVJupyter, ModeShapePlotPVQt)
from pyOMA.core.PreProcessingTools import GeometryProcessor  # noqa: E402

pv.OFF_SCREEN = True

pytestmark = pytest.mark.gui


@pytest.fixture
def msh_pv(geometry_data):
    """Off-screen pyvista backend over the shared test geometry."""
    plot = ModeShapePlotPVQt(geometry_data=geometry_data, off_screen=True)
    yield plot
    plot.close()


@pytest.fixture
def msh_pv_deformed(msh_pv):
    """As *msh_pv*, with a reproducible pseudo-mode written into the meshes.

    The displacements are injected into ``disp_nodes``/``phi_nodes``
    directly, which is exactly the interface ``ModeShapeBase`` fills in,
    so the meshes are exercised without needing a modal identification.
    """
    rng = np.random.default_rng(20260727)
    for node in msh_pv.disp_nodes:
        msh_pv.disp_nodes[node] = list(rng.normal(scale=0.5, size=3))
        msh_pv.phi_nodes[node] = list(rng.uniform(0, 2 * np.pi, size=3))
    msh_pv._update_mode_arrays()
    return msh_pv


class TestNodeIndexBijection:
    """The name <-> vtk-point-id map must be a bijection, both ways."""

    def test_every_node_maps_to_exactly_one_point(self, msh_pv, geometry_data):
        assert len(msh_pv.node_ids) == len(geometry_data.nodes)
        assert set(msh_pv.node_ids) == {str(n) for n in geometry_data.nodes}

    def test_point_ids_are_a_contiguous_range(self, msh_pv, geometry_data):
        assert sorted(msh_pv.node_ids.values()) == list(range(len(geometry_data.nodes)))

    def test_map_round_trips_in_both_directions(self, msh_pv):
        for name, point_id in msh_pv.node_ids.items():
            assert msh_pv.node_names[point_id] == name
        for point_id, name in enumerate(msh_pv.node_names):
            assert msh_pv.node_ids[name] == point_id

    def test_base_points_match_the_geometry_coordinates(self, msh_pv, geometry_data):
        for name, point_id in msh_pv.node_ids.items():
            np.testing.assert_allclose(msh_pv.base_points[point_id],
                                       geometry_data.nodes[name])

    def test_duplicate_coordinates_stay_distinct_nodes(self):
        """Two nodes at the same location must not collapse onto one point."""
        geo = GeometryProcessor()
        geo.add_node('a', [0.0, 0.0, 0.0])
        geo.add_node('b', [0.0, 0.0, 0.0])
        plot = ModeShapePlotPVQt(geometry_data=geo, off_screen=True)
        try:
            assert plot.node_ids['a'] != plot.node_ids['b']
            assert plot.base_points.shape == (2, 3)
        finally:
            plot.close()


class TestBeamTopology:
    def test_line_cells_equal_the_geometry_lines(self, msh_pv, geometry_data):
        beams = msh_pv.meshes['beams']
        assert beams.n_cells == len(geometry_data.lines)

        cells = beams.lines.reshape(-1, 3)
        assert (cells[:, 0] == 2).all()
        as_names = [(msh_pv.node_names[a], msh_pv.node_names[b])
                    for _, a, b in cells]
        assert as_names == [(str(a), str(b)) for a, b in geometry_data.lines]

    def test_node_mesh_has_one_point_per_node(self, msh_pv, geometry_data):
        assert msh_pv.meshes['nodes'].n_points == len(geometry_data.nodes)

    def test_connecting_lines_join_pinned_and_moving_copies(self, msh_pv, geometry_data):
        n_nodes = len(geometry_data.nodes)
        cn_lines = msh_pv.meshes['cn_lines']
        assert cn_lines.n_points == 2 * n_nodes
        cells = cn_lines.lines.reshape(-1, 3)
        np.testing.assert_array_equal(cells[:, 1], np.arange(n_nodes))
        np.testing.assert_array_equal(cells[:, 2], np.arange(n_nodes) + n_nodes)


class TestWarpCorrectness:
    """The VTK warp must reproduce x0 + Re*cos(2*pi*t) - Im*sin(2*pi*t)."""

    @pytest.mark.parametrize('t', [0.0, 0.25, 0.5])
    def test_warped_points_match_the_closed_form(self, msh_pv_deformed, t):
        msh = msh_pv_deformed
        msh.set_phase(t)

        expected = np.empty_like(msh.base_points)
        for name, point_id in msh.node_ids.items():
            disp = msh.disp_nodes[name]
            phi = msh.phi_nodes[name]
            for axis in range(3):
                re = disp[axis] * np.cos(phi[axis])
                im = disp[axis] * np.sin(phi[axis])
                expected[point_id, axis] = (msh.base_points[point_id, axis]
                                            + re * np.cos(2 * np.pi * t)
                                            - im * np.sin(2 * np.pi * t))

        np.testing.assert_allclose(msh.warped_points('beams'), expected, atol=1e-12)

    @pytest.mark.parametrize('t', [0.0, 0.125, 0.25, 0.5, 0.75])
    def test_warp_agrees_with_the_matplotlib_cosine_form(self, msh_pv_deformed, t):
        """disp*cos(2*pi*t + phi) is what the matplotlib backend draws."""
        msh = msh_pv_deformed
        msh.set_phase(t)

        expected = np.empty_like(msh.base_points)
        for name, point_id in msh.node_ids.items():
            for axis in range(3):
                expected[point_id, axis] = (
                    msh.base_points[point_id, axis]
                    + msh.disp_nodes[name][axis]
                    * np.cos(2 * np.pi * t + msh.phi_nodes[name][axis]))

        np.testing.assert_allclose(msh.warped_points('beams'), expected, atol=1e-12)

    def test_half_cycle_mirrors_the_displacement(self, msh_pv_deformed):
        msh = msh_pv_deformed
        msh.set_phase(0.0)
        at_zero = msh.warped_points('beams') - msh.base_points
        msh.set_phase(0.5)
        at_half = msh.warped_points('beams') - msh.base_points
        np.testing.assert_allclose(at_half, -at_zero, atol=1e-12)

    def test_pinned_half_of_cn_lines_never_moves(self, msh_pv_deformed):
        msh = msh_pv_deformed
        n_nodes = len(msh.node_names)
        for t in (0.0, 0.3, 0.7):
            msh.set_phase(t)
            np.testing.assert_allclose(
                msh.warped_points('cn_lines')[:n_nodes], msh.base_points, atol=1e-12)

    def test_amplitude_is_not_applied_twice(self, msh_pv):
        """``disp_nodes`` already carries the amplitude; the warp must not rescale."""
        node = next(iter(msh_pv.disp_nodes))
        msh_pv.disp_nodes[node] = [2.0, 0.0, 0.0]
        msh_pv.phi_nodes[node] = [0.0, 0.0, 0.0]
        msh_pv.amplitude = 7.0  # must have no effect without recomputation
        msh_pv._update_mode_arrays()
        msh_pv.set_phase(0.0)

        point_id = msh_pv.node_ids[node]
        displacement = (msh_pv.warped_points('beams')[point_id]
                        - msh_pv.base_points[point_id])
        np.testing.assert_allclose(displacement, [2.0, 0.0, 0.0], atol=1e-12)


class TestSurfaces:
    @pytest.fixture
    def msh_with_surfaces(self, geometry_data):
        names = list(geometry_data.nodes)[:4]
        geometry_data.add_surface(tuple(names))
        plot = ModeShapePlotPVQt(geometry_data=geometry_data, off_screen=True)
        yield plot, names
        plot.close()
        geometry_data.take_surface(tuple(names))

    def test_surface_mesh_uses_the_shared_node_index(self, msh_with_surfaces):
        plot, names = msh_with_surfaces
        faces = plot.meshes['surfaces'].faces
        assert faces[0] == len(names)
        assert [plot.node_names[i] for i in faces[1:]] == names

    def test_no_surface_mesh_when_geometry_has_none(self, msh_pv):
        assert 'surfaces' not in msh_pv.meshes


class TestFromMesh:
    def test_cube_yields_nodes_lines_and_quad_surfaces(self):
        geo = GeometryProcessor.from_mesh(pv.Cube())
        assert len(geo.nodes) == 8
        assert len(geo.surfaces) == 6
        assert all(len(surface) == 4 for surface in geo.surfaces)
        assert len(geo.lines) == 12

    def test_all_referenced_nodes_exist(self):
        geo = GeometryProcessor.from_mesh(pv.Cube())
        for surface in geo.surfaces:
            assert all(node in geo.nodes for node in surface)
        for line in geo.lines:
            assert all(node in geo.nodes for node in line)

    def test_without_merge_points_the_corners_stay_split(self):
        geo = GeometryProcessor.from_mesh(pv.Cube(), merge_points=False)
        assert len(geo.nodes) == pv.Cube().n_points

    def test_supplied_node_names_are_used(self):
        names = [f'N{i}' for i in range(8)]
        geo = GeometryProcessor.from_mesh(pv.Cube(), node_names=names)
        assert set(geo.nodes) == set(names)

    def test_wrong_number_of_node_names_is_rejected(self):
        with pytest.raises(ValueError, match='node_names has'):
            GeometryProcessor.from_mesh(pv.Cube(), node_names=['only-one'])

    def test_imported_mesh_renders(self):
        geo = GeometryProcessor.from_mesh(pv.Cube())
        plot = ModeShapePlotPVQt(geometry_data=geo, off_screen=True)
        try:
            assert plot.meshes['surfaces'].n_cells == 6
            assert plot.meshes['beams'].n_cells == 12
        finally:
            plot.close()


class TestLazyImport:
    def test_plotmsh_alone_pulls_in_neither_pyvista_nor_vtk(self):
        """The matplotlib backend must stay usable without the extra."""
        code = (
            'import sys; import pyOMA.core.PlotMSH; '
            "leaked = sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('pyvista', 'vtk', 'vtkmodules', 'pyvistaqt')); "
            'print(leaked); sys.exit(1 if leaked else 0)')
        result = subprocess.run([sys.executable, '-c', code],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, f'leaked modules: {result.stdout.strip()}'

    def test_importing_plotmshpv_does_not_import_pyvista_either(self):
        """Only constructing a backend may pull the extra in."""
        code = (
            'import sys; import pyOMA.core.PlotMSHpv; '
            "leaked = sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('pyvista', 'vtk', 'vtkmodules', 'pyvistaqt')); "
            'print(leaked); sys.exit(1 if leaked else 0)')
        result = subprocess.run([sys.executable, '-c', code],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, f'leaked modules: {result.stdout.strip()}'


class TestScreenshotSmoke:
    def test_drawn_mode_produces_a_non_blank_image(self, msh_pv_deformed):
        msh_pv_deformed.set_phase(0.0)
        msh_pv_deformed.render()
        img = msh_pv_deformed.plotter.screenshot(return_img=True)
        assert img.ndim == 3
        assert img.std() > 1.0, 'rendered image is blank'

    def test_frames_of_a_cycle_differ(self, msh_pv_deformed):
        msh_pv_deformed.set_phase(0.0)
        msh_pv_deformed.render()
        first = msh_pv_deformed.plotter.screenshot(return_img=True)
        msh_pv_deformed.set_phase(0.25)
        msh_pv_deformed.render()
        second = msh_pv_deformed.plotter.screenshot(return_img=True)
        assert not np.array_equal(first, second)


class TestOffScreenWidget:
    def test_widget_raises_a_clear_error_off_screen(self, msh_pv):
        with pytest.raises(RuntimeError, match='no Qt widget'):
            _ = msh_pv.widget

    def test_animate_raises_a_clear_error_off_screen(self, msh_pv):
        with pytest.raises(RuntimeError, match='set_phase'):
            msh_pv.animate()


class TestGuiFacingRegressions:
    """Fixes for issues found driving the Qt backend through PlotMSHGUI."""

    def test_nodes_render_as_flat_points(self, msh_pv_deformed):
        # render_points_as_spheres uses a point-sprite shader that draws
        # nothing on several Mesa/VTK builds, so the nodes vanished. Flat
        # points must be visible and the sphere mode must stay off.
        msh_pv_deformed.set_phase(0.0)
        msh_pv_deformed.refresh_nodes(True)
        on = msh_pv_deformed.plotter.screenshot(return_img=True)
        msh_pv_deformed.refresh_nodes(False)
        off = msh_pv_deformed.plotter.screenshot(return_img=True)
        changed = int((np.abs(on.astype(int) - off.astype(int)).sum(axis=2) > 8).sum())
        assert changed > 0, 'node markers are not drawn'
        assert not msh_pv_deformed.actors['nodes'].GetProperty().GetRenderPointsAsSpheres()

    def test_save_plot_writes_a_png(self, msh_pv, tmp_path):
        out = tmp_path / 'shot.png'
        msh_pv.save_plot(out)
        assert out.exists() and out.stat().st_size > 0
        assert out.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'

    def test_save_plot_appends_png_when_extension_missing(self, msh_pv, tmp_path):
        msh_pv.save_plot(tmp_path / 'shot')
        assert (tmp_path / 'shot.png').exists()

    def test_axis_limits_are_supported(self, msh_pv):
        assert msh_pv.supports_axis_limits
        assert len(msh_pv.get_view_limits()) == 6

    def test_set_view_limits_frames_the_camera(self, msh_pv):
        before = np.array(msh_pv.plotter.camera_position[0], dtype=float)
        xmin, xmax, ymin, ymax, zmin, zmax = msh_pv.get_view_limits()
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        h = (xmax - xmin) / 6
        box = (cx - h, cx + h, cy - h, cy + h, cz - h, cz + h)
        msh_pv.set_view_limits(*box)
        after = np.array(msh_pv.plotter.camera_position[0], dtype=float)
        assert np.linalg.norm(after - before) > 0, 'camera did not reframe'
        assert tuple(msh_pv.get_view_limits()) == box

    def test_reset_view_restores_default_limits(self, msh_pv):
        node_cube = tuple(msh_pv._compute_node_bounds())
        msh_pv.set_view_limits(0, 1, 0, 1, 0, 1)
        msh_pv.reset_view()
        assert tuple(msh_pv.get_view_limits()) == node_cube

    def test_view_angles_round_trip(self, msh_pv):
        # get_view_angles reports (elev, azim, roll); change_viewport accepts
        # the same triple (the GUI sends it when an angle field is edited).
        assert len(msh_pv.get_view_angles()) == 3
        msh_pv.change_viewport((25.0, 60.0, 0.0))
        elev, azim, _roll = msh_pv.get_view_angles()
        assert abs(elev - 25.0) < 1e-3 and abs(azim - 60.0) < 1e-3

    def test_named_and_tuple_viewports_do_not_warn(self, msh_pv, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger='pyOMA.core.PlotMSHpv'):
            msh_pv.change_viewport('ISO')
            msh_pv.change_viewport((10.0, 20.0, 0.0))
        assert 'Unknown viewport' not in caplog.text

    def test_node_and_axis_labels_exist_and_follow_visibility(self, msh_pv):
        assert 'nodes_labels' in msh_pv.actors
        assert 'axis_labels' in msh_pv.actors
        msh_pv.refresh_nodes(False)
        assert not msh_pv.actors['nodes_labels'].GetVisibility()
        msh_pv.refresh_nodes(True)
        assert msh_pv.actors['nodes_labels'].GetVisibility()

    def test_rebuild_geometry_reflects_an_added_node(self, msh_pv, geometry_data):
        n0 = len(msh_pv.node_names)
        geometry_data.add_node('99991', [30.0, 30.0, 30.0])
        msh_pv.rebuild_geometry()
        assert len(msh_pv.node_names) == n0 + 1
        assert '99991' in msh_pv.node_ids

    def test_draw_draft_chan_dof_adds_a_highlight(self, msh_pv):
        node = msh_pv.node_names[0]
        msh_pv.draw_draft_chan_dof(0, node, 45.0, 30.0, 'ch0')
        assert 'draft_chan_dof' in msh_pv.actors
        assert 'draft_chan_dof_label' in msh_pv.actors


class TestArrowLabelTips:
    """Arrow labels sit at the arrow tip (start + unit dir * length)."""

    def test_parent_child_labels_are_at_the_arrow_tip(self, msh_pv):
        length = msh_pv._arrow_length() * 0.8  # mirrors _parent_child_arrows
        points, texts = msh_pv._parent_child_labels()
        assert points, 'the test geometry defines parent-child arrows'
        for point, node in zip(points, texts):
            nid = msh_pv.node_ids[str(node)]
            dist = np.linalg.norm(np.asarray(point) - msh_pv.base_points[nid])
            assert abs(dist - length) < 1e-6

    def test_chan_dof_labels_are_at_the_arrow_tip(self, msh_pv):
        node = msh_pv.node_names[0]
        msh_pv.chan_dofs = [(0, node, 30.0, 20.0, 'ch0')]
        length = msh_pv._arrow_length()
        (point,), (text,) = msh_pv._chan_dof_labels()
        nid = msh_pv.node_ids[node]
        dist = np.linalg.norm(np.asarray(point) - msh_pv.base_points[nid])
        assert abs(dist - length) < 1e-6
        assert text == 'ch0'


class TestJupyterParity:
    """The notebook backend now has the Qt backend's overlays and toggles."""

    def test_arrow_and_label_actors_exist(self, msh_jupyter):
        keys = set(msh_jupyter.actors)
        assert any(k.startswith('axis_') for k in keys)
        assert any(k.startswith('parent_child_') for k in keys)
        assert {'axis_labels', 'nodes_labels', 'parent_child_labels'} <= keys

    def test_refresh_toggles_node_labels(self, msh_jupyter):
        msh_jupyter.refresh_nodes(False)
        assert not msh_jupyter.actors['nodes_labels'].GetVisibility()
        msh_jupyter.refresh_nodes(True)
        assert msh_jupyter.actors['nodes_labels'].GetVisibility()

    def test_refresh_parent_childs_shows_the_arrows(self, msh_jupyter):
        msh_jupyter.refresh_parent_childs(True)
        pc = [a for k, a in msh_jupyter.actors.items()
              if k.startswith('parent_child_')]
        assert pc and all(a.GetVisibility() for a in pc)

    def test_reset_view_is_available(self, msh_jupyter):
        msh_jupyter.reset_view()  # must not raise

    def test_camera_methods_are_available(self, msh_jupyter):
        # get_view_angles/change_viewport are shared on the pyvista base, so the
        # notebook panel (which reads them) does not crash with a NoneType.
        angles = msh_jupyter.get_view_angles()
        assert angles is not None and len(angles) == 3
        msh_jupyter.change_viewport('X')
        msh_jupyter.change_viewport((20.0, 40.0, 0.0))  # must not raise


def _seed_mode(plot, seed=7):
    """Inject a reproducible pseudo-mode straight into disp/phi tables."""
    rng = np.random.default_rng(seed)
    for node in plot.disp_nodes:
        plot.disp_nodes[node] = list(rng.normal(size=3))
        plot.phi_nodes[node] = list(rng.uniform(0, 2 * np.pi, size=3))
    plot._update_mode_arrays()


class TestSurfaceColouring:
    """Surfaces carry a per-node displacement scalar, updated every frame."""

    @pytest.fixture
    def geo_with_surface(self, geometry_data):
        names = list(geometry_data.nodes)[:4]
        geometry_data.add_surface(tuple(names))
        yield geometry_data
        geometry_data.take_surface(tuple(names))

    def test_qt_surface_scalar_changes_with_phase(self, geo_with_surface):
        plot = ModeShapePlotPVQt(geometry_data=geo_with_surface, off_screen=True)
        try:
            assert 'surfaces' in plot.actors
            _seed_mode(plot)
            plot.set_phase(0.0)
            s0 = plot.meshes['surfaces'].point_data[plot._SURFACE_ARRAY].copy()
            plot.set_phase(0.25)
            s1 = plot.meshes['surfaces'].point_data[plot._SURFACE_ARRAY].copy()
            assert not np.allclose(s0, s1)
        finally:
            plot.close()

    def test_jupyter_surface_scalar_changes_between_frames(self, geo_with_surface):
        plot = ModeShapePlotPVJupyter(geometry_data=geo_with_surface,
                                      off_screen=True, n_frames=16)
        assert 'surfaces' in plot.actors
        _seed_mode(plot)
        plot.compute_frames()
        plot.show_frame(0)
        s0 = plot.meshes['surfaces'].point_data[plot._SURFACE_ARRAY].copy()
        plot.show_frame(4)
        s4 = plot.meshes['surfaces'].point_data[plot._SURFACE_ARRAY].copy()
        assert not np.allclose(s0, s4)

    def test_mapper_colours_by_the_displacement_scalar_not_the_mode_vector(
            self, geo_with_surface):
        # Regression: _update_mode_arrays used to write mode_re onto the surface
        # mesh, which became the active scalars, so the mapper coloured by the
        # mode vector (one flat colour) instead of the displacement magnitude.
        plot = ModeShapePlotPVQt(geometry_data=geo_with_surface, off_screen=True)
        try:
            _seed_mode(plot)
            plot.set_phase(0.0)
            mapper = plot.actors['surfaces'].GetMapper()
            mapper.Update()
            active = mapper.GetInput().GetPointData().GetScalars()
            assert active is not None
            assert active.GetName() == plot._SURFACE_ARRAY
            values = np.asarray(plot.meshes['surfaces'].point_data[plot._SURFACE_ARRAY])
            assert float(values.max() - values.min()) > 0  # varies per node
        finally:
            plot.close()


class TestAnimationExport:
    """export_animation_frames writes one numbered file per cycle frame."""

    @pytest.mark.parametrize('fmt', ['png', 'pdf'])
    def test_qt_export_writes_frames(self, msh_pv_deformed, tmp_path, fmt):
        paths = msh_pv_deformed.export_animation_frames(
            tmp_path / fmt, fmt=fmt, n_frames=4)
        assert len(paths) == 4
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)
        assert all(p.suffix == f'.{fmt}' for p in paths)

    def test_export_rejects_unknown_format(self, msh_pv_deformed, tmp_path):
        with pytest.raises(ValueError, match='fmt'):
            msh_pv_deformed.export_animation_frames(tmp_path, fmt='gif')

    def test_jupyter_export_delegates_to_an_offscreen_qt(self, msh_jupyter, tmp_path):
        paths = msh_jupyter.export_animation_frames(tmp_path, fmt='png', n_frames=4)
        assert len(paths) == 4
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)


class TestBackendResolver:
    """resolve_mode_shape_backend honours the env var and module override."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setattr(core, 'MSH_BACKEND', None)
        monkeypatch.delenv('PYOMA_MSH_BACKEND', raising=False)

    def test_auto_prefers_pyvista_when_installed(self):
        from pyOMA.core import resolve_mode_shape_backend
        assert resolve_mode_shape_backend() is ModeShapePlotPVQt

    def test_auto_notebook_prefers_the_jupyter_backend(self):
        from pyOMA.core import resolve_mode_shape_backend
        assert resolve_mode_shape_backend('notebook') is ModeShapePlotPVJupyter

    def test_env_forces_matplotlib(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setenv('PYOMA_MSH_BACKEND', 'matplotlib')
        assert resolve_backend() is core.ModeShapePlot

    def test_env_matplotlib_is_context_independent(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setenv('PYOMA_MSH_BACKEND', 'matplotlib')
        assert core.resolve_mode_shape_backend('notebook') is core.ModeShapePlot

    def test_env_pyvista_notebook_picks_the_jupyter_backend(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setenv('PYOMA_MSH_BACKEND', 'pyvista')
        assert core.resolve_mode_shape_backend('notebook') is ModeShapePlotPVJupyter
        assert core.resolve_mode_shape_backend() is ModeShapePlotPVQt

    def test_module_override_beats_env(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setenv('PYOMA_MSH_BACKEND', 'matplotlib')
        monkeypatch.setattr(core, 'MSH_BACKEND', 'pyvista')
        assert resolve_backend() is ModeShapePlotPVQt

    def test_class_override(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setattr(core, 'MSH_BACKEND', core.ModeShapePlot)
        assert resolve_backend() is core.ModeShapePlot

    def test_class_override_ignores_context(self, monkeypatch):
        import pyOMA.core as core
        monkeypatch.setattr(core, 'MSH_BACKEND', ModeShapePlotPVQt)
        assert core.resolve_mode_shape_backend('notebook') is ModeShapePlotPVQt


def resolve_backend():
    from pyOMA.core import resolve_mode_shape_backend
    return resolve_mode_shape_backend()


# ── Notebook backend ─────────────────────────────────────────────────────────

@pytest.fixture
def msh_jupyter(geometry_data):
    """Off-screen notebook backend with a reproducible pseudo-mode."""
    plot = ModeShapePlotPVJupyter(geometry_data=geometry_data, n_frames=32,
                                  off_screen=True)
    rng = np.random.default_rng(20260727)
    for node in plot.disp_nodes:
        plot.disp_nodes[node] = list(rng.normal(scale=0.5, size=3))
        plot.phi_nodes[node] = list(rng.uniform(0, 2 * np.pi, size=3))
    plot._update_mode_arrays()
    plot.compute_frames()
    return plot


class TestPrecomputedCycle:
    def test_frame_count_matches_the_configured_n_frames(self, msh_jupyter):
        assert msh_jupyter.n_frames == 32
        assert msh_jupyter.frames.shape[0] == 32

    def test_frames_have_one_row_per_node(self, msh_jupyter, geometry_data):
        assert msh_jupyter.frames.shape == (32, len(geometry_data.nodes), 3)

    def test_half_cycle_negates_the_displacement(self, msh_jupyter):
        base = msh_jupyter.base_points
        at_zero = msh_jupyter.frames[0] - base
        at_half = msh_jupyter.frames[msh_jupyter.n_frames // 2] - base
        np.testing.assert_allclose(at_half, -at_zero, atol=1e-12)

    def test_frames_match_the_closed_form(self, msh_jupyter):
        for i in (0, 7, 16, 31):
            expected = msh_jupyter.mode_points(i / msh_jupyter.n_frames)
            np.testing.assert_allclose(msh_jupyter.frames[i], expected, atol=1e-12)

    def test_frames_agree_with_the_qt_backend_warp(self, geometry_data, msh_jupyter):
        """Both backends must trace the same path for the same mode."""
        qt = ModeShapePlotPVQt(geometry_data=geometry_data, off_screen=True)
        try:
            qt.disp_nodes = dict(msh_jupyter.disp_nodes)
            qt.phi_nodes = dict(msh_jupyter.phi_nodes)
            qt._update_mode_arrays()
            for i in (0, 8, 16, 24):
                qt.set_phase(i / msh_jupyter.n_frames)
                np.testing.assert_allclose(qt.warped_points('beams'),
                                           msh_jupyter.frames[i], atol=1e-12)
        finally:
            qt.close()

    def test_no_warp_filters_are_built(self, msh_jupyter):
        """Client-side rendering cannot run server-side VTK filters."""
        assert msh_jupyter.warps == {}

    def test_show_frame_moves_every_mesh(self, msh_jupyter):
        msh_jupyter.show_frame(8)
        np.testing.assert_allclose(msh_jupyter.meshes['beams'].points,
                                   msh_jupyter.frames[8], atol=1e-12)
        np.testing.assert_allclose(msh_jupyter.meshes['nodes'].points,
                                   msh_jupyter.frames[8], atol=1e-12)

    def test_show_frame_wraps_around(self, msh_jupyter):
        msh_jupyter.show_frame(msh_jupyter.n_frames + 3)
        np.testing.assert_allclose(msh_jupyter.meshes['beams'].points,
                                   msh_jupyter.frames[3], atol=1e-12)

    def test_show_frame_keeps_the_pinned_half_of_cn_lines(self, msh_jupyter):
        n_nodes = len(msh_jupyter.node_names)
        msh_jupyter.show_frame(5)
        points = msh_jupyter.meshes['cn_lines'].points
        np.testing.assert_allclose(points[:n_nodes], msh_jupyter.base_points, atol=1e-12)
        np.testing.assert_allclose(points[n_nodes:], msh_jupyter.frames[5], atol=1e-12)

    @pytest.mark.parametrize('bad', [1, 0, -4, 2.5, 'many'])
    def test_invalid_n_frames_is_rejected(self, geometry_data, bad):
        with pytest.raises(ValueError, match='n_frames'):
            ModeShapePlotPVJupyter(geometry_data=geometry_data, n_frames=bad,
                                   off_screen=True)


class TestHtmlExport:
    def test_export_is_self_contained_and_holds_every_frame(self, msh_jupyter, tmp_path):
        import json
        import re

        path = msh_jupyter.export_html(tmp_path / 'msh.html')
        text = path.read_text(encoding='utf-8')

        payload = json.loads(re.search(r'const D = (\{.*?\});\n', text, re.S).group(1))
        frames = np.array(payload['frames'])
        assert frames.shape == msh_jupyter.frames.shape
        np.testing.assert_allclose(frames, msh_jupyter.frames, atol=1e-5)

        # A page that fetched anything at view time would not be round-trip free.
        assert not re.findall(r'(?:src|href)=["\']([^"\']+)', text)
        assert 'http://' not in text and 'https://' not in text

    def test_export_carries_the_beam_topology(self, msh_jupyter, geometry_data, tmp_path):
        import json
        import re

        path = msh_jupyter.export_html(tmp_path / 'msh.html')
        payload = json.loads(
            re.search(r'const D = (\{.*?\});\n', path.read_text(), re.S).group(1))

        assert len(payload['segments']) == len(geometry_data.lines)
        as_names = [(msh_jupyter.node_names[a], msh_jupyter.node_names[b])
                    for a, b in payload['segments']]
        assert as_names == [(str(a), str(b)) for a, b in geometry_data.lines]


# ── Phase 2: the GUI drives the backend through the neutral contract ─────────

class TestBackendContract:
    def test_capability_flags(self, msh_pv):
        # Axis limits are honoured by framing the camera; time-history
        # animation and matplotlib picking remain unsupported.
        assert msh_pv.supports_axis_limits is True
        assert msh_pv.supports_data_animation is False
        assert msh_pv.supports_picking is False

    def test_view_hooks(self, msh_pv):
        # Both limits and angles are reported now: the camera frames a box and
        # its orientation maps to an (elev, azim, roll) triple.
        assert len(msh_pv.get_view_limits()) == 6
        assert len(msh_pv.get_view_angles()) == 3
        msh_pv.set_view_limits(0, 1, 0, 1, 0, 1)  # must not raise

    def test_no_camera_callbacks_to_connect(self, msh_pv):
        assert msh_pv.connect_view_change(lambda event: None) == []

    def test_editors_accept_this_backend(self, msh_pv):
        # The geometry / channel-DOF editors now drive any backend through the
        # neutral redraw/edit contract, so the pyvista backend exposes the
        # hooks they call.
        for name in ('redraw', 'redraw_geometry', 'draw_draft_chan_dof',
                     'rebuild_geometry'):
            assert callable(getattr(msh_pv, name))


class TestModeShapeGUIIntegration:
    """PlotMSHGUI must drive the pyvista backend without matplotlib internals."""

    @pytest.fixture
    def gui(self, qapp, geometry_data):  # noqa: ARG002 - qapp starts Qt
        from pyOMA.GUI.PlotMSHGUI import ModeShapeGUI

        plot = ModeShapePlotPVQt(geometry_data=geometry_data)
        window = ModeShapeGUI(plot)
        yield window
        window.close()
        plot.close()

    def test_the_interactor_replaces_the_matplotlib_canvas(self, gui):
        assert gui.canvas is gui.mode_shape_plot.widget
        assert hasattr(gui.canvas, 'winId')  # it is a QWidget

    def test_axis_limit_controls_are_enabled(self, gui):
        # The backend frames the camera on the limit box, so the GUI's limit
        # fields and zoom buttons are live rather than greyed out.
        for edit in (gui.x_limits_min_edit, gui.y_limits_min_edit,
                     gui.z_limits_min_edit):
            assert edit.isEnabled()
        assert gui.zoom_plus_button.isEnabled()

    def test_data_animation_button_is_disabled(self, gui):
        assert not gui.ani_data_button.isEnabled()

    def test_visibility_checkboxes_reach_the_backend(self, gui):
        gui.mode_shape_plot.refresh_nodes(False)
        assert gui.mode_shape_plot.show_nodes is False
        gui.mode_shape_plot.refresh_nodes(True)
        assert gui.mode_shape_plot.show_nodes is True


class TestQtAnimationWiring:
    """Regressions from driving a real Qt window (see the VNC session notes).

    Neither bug is reachable off-screen -- the off-screen path uses a plain
    ``pyvista.Plotter`` and never builds a QtInteractor or a QTimer -- but
    both are reachable here, because constructing a QtInteractor works under
    the ``offscreen`` platform plugin even though rendering into it does not.
    """

    @pytest.fixture
    def interactive(self, qapp, geometry_data):  # noqa: ARG002 - qapp starts Qt
        plot = ModeShapePlotPVQt(geometry_data=geometry_data)
        yield plot
        plot.stop_ani()

    def test_interactor_self_render_timer_is_disabled(self, interactive):
        """Two render timers race the VTK pipeline and segfault after ~120 frames.

        QtInteractor starts its own 5 Hz render timer unless auto_update is
        off. This backend renders from its own animation timer, so the
        interactor's must stay stopped.
        """
        assert not interactive.plotter.render_timer.isActive()

    def test_animate_starts_a_running_timer(self, interactive):
        """``add_callback`` is a BackgroundPlotter method, not a QtInteractor one.

        Driving frames through it raised RuntimeError against a real
        QtInteractor, so the timer is created directly.
        """
        assert interactive.is_interactive
        interactive.animate()
        assert interactive.animated
        assert interactive._timer is not None
        assert interactive._timer.isActive()
        assert interactive._timer.interval() == 50

    def test_stop_ani_stops_the_timer_and_resets_the_phase(self, interactive):
        interactive.animate()
        timer = interactive._timer
        interactive.stop_ani()
        assert not interactive.animated
        assert not timer.isActive()
        assert interactive._timer is None
        assert interactive.seq_num == 0

    def test_animate_toggles_off_when_called_twice(self, interactive):
        interactive.animate()
        interactive.animate()
        assert not interactive.animated

    def test_advance_frame_walks_the_cycle(self, interactive):
        """The timer callback must move the warp without needing a render."""
        node = next(iter(interactive.disp_nodes))
        interactive.disp_nodes[node] = [1.0, 0.0, 0.0]
        interactive.phi_nodes[node] = [0.0, 0.0, 0.0]
        interactive._update_mode_arrays()

        point_id = interactive.node_ids[node]
        interactive._frame = 0
        seen = []
        for _ in range(4):
            interactive._advance_frame()
            seen.append(interactive.warped_points('beams')[point_id, 0]
                        - interactive.base_points[point_id, 0])
        expected = [np.cos(2 * np.pi * i / 25) for i in range(1, 5)]
        np.testing.assert_allclose(seen, expected, atol=1e-12)
