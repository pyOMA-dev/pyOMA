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
    def test_capability_flags_are_conservative(self, msh_pv):
        assert msh_pv.supports_axis_limits is False
        assert msh_pv.supports_data_animation is False
        assert msh_pv.supports_picking is False

    def test_view_limit_hooks_are_inert(self, msh_pv):
        assert msh_pv.get_view_limits() is None
        assert msh_pv.get_view_angles() is None
        msh_pv.set_view_limits(0, 1, 0, 1, 0, 1)  # must not raise

    def test_no_camera_callbacks_to_connect(self, msh_pv):
        assert msh_pv.connect_view_change(lambda event: None) == []

    def test_editors_refuse_this_backend(self, msh_pv):
        from pyOMA.core.ModeShapeBase import require_picking_backend

        with pytest.raises(TypeError, match='matplotlib picking'):
            require_picking_backend(msh_pv, 'ChanDofEditorGUI')


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

    def test_axis_limit_controls_are_disabled(self, gui):
        for edit in (gui.x_limits_min_edit, gui.y_limits_min_edit,
                     gui.z_limits_min_edit):
            assert not edit.isEnabled()
        assert not gui.zoom_plus_button.isEnabled()

    def test_data_animation_button_is_disabled(self, gui):
        assert not gui.ani_data_button.isEnabled()

    def test_visibility_checkboxes_reach_the_backend(self, gui):
        gui.mode_shape_plot.refresh_nodes(False)
        assert gui.mode_shape_plot.show_nodes is False
        gui.mode_shape_plot.refresh_nodes(True)
        assert gui.mode_shape_plot.show_nodes is True
