"""
Characterization tests for the *backend-agnostic* behaviour of
:class:`pyOMA.core.PlotMSH.ModeShapePlot`.

These tests pin down the pure-computation contract of ``ModeShapePlot`` -
merging-mode detection, mode lookup, the chan_dof -> node displacement math,
the visibility/amplitude/part state machine and the data-animation
accumulation - *without asserting on any matplotlib artist*.  They exist so
that a second rendering backend (planned pyvista) can be proven to agree with
the matplotlib backend on everything that is not rendering.

They must pass unchanged both before and after the extraction of
``ModeShapeBase``.  None of them is marked ``@pytest.mark.gui``; they run in
the plain ``tests.yml`` job.
"""
import copy

import numpy as np
import pytest

from pyOMA.core.PlotMSH import ModeShapePlot
from pyOMA.core.PreProcessingTools import GeometryProcessor, PreProcessSignals
from pyOMA.core.SSICovRef import PogerSSICovRef
from pyOMA.core.StabilDiagram import StabilCalc, StabilCluster
from pyOMA.core.PostProcessingTools import MergePoSER
from pyOMA.core.Helpers import calc_xyz


# ── Small helpers ─────────────────────────────────────────────────────────────

def _simple_geometry(coords):
    """Build a :class:`GeometryProcessor` from ``{name: (x, y, z)}``."""
    geo = GeometryProcessor()
    for name, xyz in coords.items():
        geo.add_node(name, list(xyz))
    return geo


def _zero_disp(plot):
    """Reset ``disp_nodes`` / ``phi_nodes`` to all-zero for a fresh assignment."""
    plot.disp_nodes = {i: [0, 0, 0] for i in plot.geometry_data.nodes}
    plot.phi_nodes = {i: [0, 0, 0] for i in plot.geometry_data.nodes}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def stabil_calc_single(modal_data_ssi_cov):
    """StabilCluster over the synthetic single-setup result with a handful of
    finite modes selected.  Provides real MPC/MP/MPD matrices so that
    ``change_mode`` can be exercised end-to-end."""
    sc = StabilCluster(modal_data_ssi_cov)
    sc.calculate_stabilization_masks()
    order = 10
    finite = np.where(np.isfinite(modal_data_ssi_cov.modal_frequencies[order, :]))[0]
    # three modes with well-separated frequencies
    sc.select_modes = [(order, int(finite[0])),
                       (order, int(finite[1])),
                       (order, int(finite[3]))]
    return sc


@pytest.fixture
def single_plot(geometry_data, stabil_calc_single, modal_data_ssi_cov,
                prep_signals_with_corr):
    """Single-setup ModeShapePlot (fresh per test - draw_msh mutates state)."""
    return ModeShapePlot(
        geometry_data=geometry_data,
        stabil_calc=stabil_calc_single,
        modal_data=modal_data_ssi_cov,
        prep_signals=prep_signals_with_corr,
    )


@pytest.fixture(scope='module')
def poger_data(test_files_dir):
    """A PoGER-style ``(modal_data, stabil_calc)`` pair.

    ``tests/files/merged_poger/stabil_data.npz`` is a saved ``StabilCalc``
    state produced from a real 15-setup PoGER merge.  There is no companion
    ``modal_data.npz``, so a lightweight :class:`PogerSSICovRef` carrier is
    fabricated with matching ``setup_name`` and array shapes; loading the
    saved state onto it restores the real ``select_modes`` and MPC/MP/MPD
    matrices used in the PoGER path.
    """
    npz_path = test_files_dir / 'merged_poger' / 'stabil_data.npz'
    saved = np.load(npz_path, allow_pickle=True)
    setup_name = str(saved['self.setup_name'].item())

    n_orders = 100
    n_channels = 6
    rng = np.random.default_rng(0)

    md = PogerSSICovRef()
    md.setup_name = setup_name
    md.start_time = None
    md.max_model_order = n_orders
    md.modal_frequencies = np.abs(rng.normal(size=(n_orders, n_orders))) + 1.0
    md.modal_damping = np.abs(rng.normal(size=(n_orders, n_orders))) * 0.01
    md.eigenvalues = (rng.normal(size=(n_orders, n_orders))
                      + 1j * rng.normal(size=(n_orders, n_orders)))
    md.mode_shapes = (rng.normal(size=(n_channels, n_orders, n_orders))
                      + 1j * rng.normal(size=(n_channels, n_orders, n_orders)))
    md.std_frequencies = None
    md.modal_contributions = None
    md.merged_chan_dofs = [[c, str(c), 1.0, 0.0, f'ch{c}'] for c in range(n_channels)]
    md.merged_num_channels = n_channels
    md.prep_signals = PreProcessSignals(
        rng.normal(size=(256, n_channels)), 128, ref_channels=[0])

    sc = StabilCalc.load_state(npz_path, md)
    return md, sc


@pytest.fixture
def poger_plot(poger_data):
    md, sc = poger_data
    geo = _simple_geometry({str(c): (float(c), 0.0, 0.0)
                            for c in range(md.merged_num_channels)})
    return ModeShapePlot(geometry_data=geo, stabil_calc=sc, modal_data=md)


@pytest.fixture
def poser_merged():
    """A fabricated :class:`MergePoSER` carrier for the PoSER path."""
    n_channels = 3
    n_modes = 2
    md = MergePoSER()
    md.merged_chan_dofs = [[c, str(c), 1.0, 0.0, f'ch{c}'] for c in range(n_channels)]
    md.merged_num_channels = n_channels
    md.mean_frequencies = np.array([1.5, 3.0])
    md.mean_damping = np.array([0.01, 0.02])
    md.merged_mode_shapes = np.full((n_channels, n_modes, 1), 1 + 1j, dtype=complex)
    md.std_frequencies = None
    md.std_damping = None
    md.setup_name = 'poser_test'
    md.start_time = None
    return md


@pytest.fixture
def empty_plot(geometry_data):
    """ModeShapePlot with geometry only - the empty merging path."""
    return ModeShapePlot(geometry_data=geometry_data)


# ── 4.1 Merging-mode detection ────────────────────────────────────────────────

class TestMergingDetection:

    def test_detect_mode_string(self, poser_merged, poger_data):
        md_poger, _ = poger_data
        assert ModeShapePlot._detect_merging_mode(poser_merged, None) == 'PoSER'
        assert ModeShapePlot._detect_merging_mode(None, md_poger) == 'PoGER'
        # both absent -> no merging mode
        assert ModeShapePlot._detect_merging_mode(None, None) is None
        # any non-None, non-Poger modal_data is the single-setup path
        assert ModeShapePlot._detect_merging_mode(None, object()) == 'single'

    def test_single_setup_attrs(self, single_plot, modal_data_ssi_cov,
                                prep_signals_with_corr, stabil_calc_single):
        assert single_plot.chan_dofs == prep_signals_with_corr.chan_dofs
        assert single_plot.num_channels == prep_signals_with_corr.num_analised_channels
        assert single_plot.mode_shapes is modal_data_ssi_cov.mode_shapes
        assert single_plot.select_modes == stabil_calc_single.select_modes
        # std not available for a plain (non-variance) estimator
        assert single_plot.std_frequencies is None

    def test_poger_attrs(self, poger_plot, poger_data):
        md, sc = poger_data
        assert poger_plot.chan_dofs == md.merged_chan_dofs
        assert poger_plot.num_channels == md.merged_num_channels
        assert poger_plot.mode_shapes is md.mode_shapes
        assert poger_plot.select_modes == sc.select_modes
        assert poger_plot.std_frequencies is None

    def test_poser_attrs(self, geometry_data, poser_merged):
        plot = ModeShapePlot(geometry_data=geometry_data, merged_data=poser_merged)
        assert plot.chan_dofs == poser_merged.merged_chan_dofs
        assert plot.num_channels == poser_merged.merged_num_channels
        assert plot.mode_shapes is poser_merged.merged_mode_shapes
        assert plot.modal_frequencies is poser_merged.mean_frequencies
        # select_modes are (mode, 0) pairs, one per mean frequency
        assert plot.select_modes == [(0, 0), (1, 0)]

    def test_empty_with_prep_signals(self, geometry_data, prep_signals):
        plot = ModeShapePlot(geometry_data=geometry_data, prep_signals=prep_signals)
        assert plot.chan_dofs == prep_signals.chan_dofs
        assert plot.num_channels == prep_signals.num_analised_channels
        assert plot.select_modes == []
        assert plot.mode_shapes.shape == (1, 1, 0)
        assert plot.mode_index is None

    def test_empty_without_prep_signals(self, empty_plot):
        assert empty_plot.chan_dofs == []
        assert empty_plot.num_channels == 0
        assert empty_plot.select_modes == []
        assert empty_plot.setup_name == ''
        assert empty_plot.start_time is None


# ── 4.2 Mode lookup ───────────────────────────────────────────────────────────

class TestModeLookup:

    def test_lookup_by_index(self, single_plot, stabil_calc_single):
        for i, expected in enumerate(stabil_calc_single.select_modes):
            assert single_plot._lookup_mode_index(index=i) == expected

    def test_lookup_by_mode_index_passthrough(self, single_plot,
                                              stabil_calc_single):
        target = stabil_calc_single.select_modes[1]
        assert single_plot._lookup_mode_index(mode_index=target) == target

    def test_lookup_by_frequency_nearest(self, single_plot):
        freqs = single_plot.get_frequencies()
        # exact hit
        target = single_plot.select_modes[0]
        f0 = single_plot.modal_frequencies[target[0], target[1]]
        assert single_plot._lookup_mode_index(frequency=f0) == target
        # a value slightly above the largest selected frequency snaps to it
        idx = single_plot._lookup_mode_index(frequency=max(freqs) + 1000.0)
        f_found = single_plot.modal_frequencies[idx[0], idx[1]]
        assert f_found == pytest.approx(max(freqs))

    def test_lookup_no_args_raises(self, single_plot):
        with pytest.raises(RuntimeError):
            single_plot._lookup_mode_index()

    def test_lookup_out_of_range_raises(self, single_plot):
        with pytest.raises(IndexError):
            single_plot._lookup_mode_index(index=999)

    def test_get_frequencies_sorted(self, single_plot):
        freqs = single_plot.get_frequencies()
        assert freqs == sorted(freqs)
        assert len(freqs) == len(single_plot.select_modes)

    def test_change_mode_returns_modal_values(self, single_plot):
        target = single_plot.select_modes[0]
        order, mode, freq, damp, MPC, MP, MPD = single_plot.change_mode(
            mode_index=target)
        assert (mode, order) == (target[0], target[1])
        assert freq == single_plot.modal_frequencies[target[0], target[1]]
        assert damp == single_plot.modal_damping[target[0], target[1]]
        assert single_plot.mode_index == target
        # stabil params available for a StabilCluster-backed plot
        assert MPC is not None and MP is not None and MPD is not None

    def test_change_mode_by_frequency_and_index_agree(self, single_plot):
        by_index = single_plot.change_mode(index=2)
        f = by_index[2]
        by_freq = single_plot.change_mode(frequency=f)
        assert by_freq[:4] == by_index[:4]

    def test_get_stabil_params_none_without_stabil_calc(self, empty_plot):
        assert empty_plot.stabil_calc is None
        assert empty_plot._get_stabil_params((0, 0)) == (None, None, None)

    def test_change_mode_on_poger(self, poger_plot):
        target = poger_plot.select_modes[0]
        res = poger_plot.change_mode(mode_index=target)
        assert res[2] == poger_plot.modal_frequencies[target[0], target[1]]
        assert poger_plot.mode_index == target


# ── 4.3 Displacement computation ──────────────────────────────────────────────

class TestDisplacementMath:
    """Deterministic geometry + hand-written chan_dofs and mode shapes.

    A single node ``'A'`` at the origin is enough for the per-node assignment
    branches; the direction vectors come from ``calc_xyz`` exactly as the code
    under test computes them.
    """

    @pytest.fixture
    def plot(self):
        geo = _simple_geometry({'A': (0.0, 0.0, 0.0),
                                'B': (1.0, 0.0, 0.0),
                                'C': (0.0, 1.0, 0.0)})
        p = ModeShapePlot(geometry_data=geo)
        p.real = False
        _zero_disp(p)
        return p

    def test_single_sensor_real(self, plot):
        # entry = [chan, dir_x, dir_y, dir_z, disp]; az=elev=0 -> +x direction
        plot._assign_single_sensor_disp('A', [0, 1.0, 0.0, 0.0, 2 + 0j], 3.0)
        np.testing.assert_allclose(plot.disp_nodes['A'], [6.0, 0.0, 0.0])
        np.testing.assert_allclose(plot.phi_nodes['A'], [0.0, 0.0, 0.0])

    def test_single_sensor_complex_phase(self, plot):
        plot._assign_single_sensor_disp('A', [0, 1.0, 0.0, 0.0, 1j], 1.0)
        # |1j| = 1, angle = pi/2, applied on x direction only
        np.testing.assert_allclose(plot.disp_nodes['A'], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(plot.phi_nodes['A'],
                                   [np.pi / 2, np.pi / 2, np.pi / 2])

    def test_axis_aligned(self, plot):
        cds = [[0, 1.0, 0.0, 0.0, 2 + 0j],
               [1, 0.0, 1.0, 0.0, 3 + 0j],
               [2, 0.0, 0.0, 1.0, 4 + 0j]]
        plot._assign_axis_aligned_disp('A', cds, 1.0)
        np.testing.assert_allclose(plot.disp_nodes['A'], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(plot.phi_nodes['A'], [0.0, 0.0, 0.0])

    def test_multi_sensor_dispatches_to_axis_aligned(self, plot):
        # one sensor per axis -> active_per_axis <= 1 -> axis-aligned branch
        cds = [[0, 1.0, 0.0, 0.0, 2 + 0j],
               [1, 0.0, 1.0, 0.0, 3 + 0j],
               [2, 0.0, 0.0, 1.0, 4 + 0j]]
        plot._assign_multi_sensor_disp('A', cds, 1.0)
        np.testing.assert_allclose(plot.disp_nodes['A'], [2.0, 3.0, 4.0])

    def test_multi_sensor_dispatches_to_lstsq(self, plot):
        # two sensors share the x axis -> active_per_axis[0] == 2 -> lstsq branch
        cds = [[0, 1.0, 0.0, 0.0, 2 + 0j],
               [1, 0.0, 1.0, 0.0, 3 + 0j],
               [2, 0.0, 0.0, 1.0, 4 + 0j],
               [3, 1.0, 0.0, 0.0, 6 + 0j]]
        plot._assign_multi_sensor_disp('A', cds, 1.0)
        # least squares over [1,0,0],[0,1,0],[0,0,1],[1,0,0]:
        # x = (2 + 6) / 2 = 4, y = 3, z = 4
        np.testing.assert_allclose(plot.disp_nodes['A'], [4.0, 3.0, 4.0])

    def test_lstsq_overdetermined(self, plot):
        cds = [[0, 1.0, 0.0, 0.0, 2 + 0j],
               [1, 0.0, 1.0, 0.0, 3 + 0j],
               [2, 0.0, 0.0, 1.0, 4 + 0j],
               [3, 1.0, 0.0, 0.0, 6 + 0j]]
        plot._assign_lstsq_sensor_disp('A', cds, 2.0)
        np.testing.assert_allclose(plot.disp_nodes['A'], [8.0, 6.0, 8.0])
        np.testing.assert_allclose(plot.phi_nodes['A'], [0.0, 0.0, 0.0])

    def test_compute_chan_dof_displacements_full(self):
        geo = _simple_geometry({'A': (0.0, 0.0, 0.0), 'B': (1.0, 0.0, 0.0)})
        p = ModeShapePlot(geometry_data=geo)
        p.real = False
        _zero_disp(p)
        # A: single x sensor (chan 0); B: two axis-aligned sensors (chan 1 x, chan 2 y)
        p.chan_dofs = [[0, 'A', 0.0, 0.0, 'cA'],
                       [1, 'B', 0.0, 0.0, 'cB1'],
                       [2, 'B', 90.0, 0.0, 'cB2']]
        mode_shape = np.array([2 + 0j, 3 + 0j, 4 + 0j])
        p._compute_chan_dof_displacements(mode_shape, 1.0)
        np.testing.assert_allclose(p.disp_nodes['A'], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(p.disp_nodes['B'], [3.0, 4.0, 0.0], atol=1e-12)

    def test_parent_child_propagation(self):
        geo = _simple_geometry({'P': (0.0, 0.0, 0.0), 'C': (1.0, 0.0, 0.0)})
        # parent DOF along +x, child DOF along +y with amplification factor 2
        geo.add_parent_child(['P', 1.0, 0.0, 0.0, 'C', 0.0, 2.0, 0.0])
        p = ModeShapePlot(geometry_data=geo)
        p.disp_nodes = {'P': [5.0, 0.0, 0.0], 'C': [0.0, 0.0, 0.0]}
        p.phi_nodes = {'P': [0.3, 0.0, 0.0], 'C': [0.0, 0.0, 0.0]}
        p._compute_parent_child_displacements()
        # parent_disp = 5 * 1 = 5, child y += 5 * 2 = 10, phi propagated
        np.testing.assert_allclose(p.disp_nodes['C'], [0.0, 10.0, 0.0])
        np.testing.assert_allclose(p.phi_nodes['C'], [0.0, 0.3, 0.0])

    @pytest.mark.parametrize('disp, expected', [
        (1 + 0j, (0, 1.0)),
        (-1 + 0j, (0, -1.0)),
        (1j, (0, 1.0)),
        (-1j, (0, -1.0)),
    ])
    def test_disp_phase_mag_real(self, plot, disp, expected):
        plot.real = True
        phase, mag = plot._disp_phase_mag(disp)
        assert phase == expected[0]
        assert mag == pytest.approx(expected[1])

    def test_disp_phase_mag_complex(self, plot):
        plot.real = False
        phase, mag = plot._disp_phase_mag(1 + 1j)
        assert phase == pytest.approx(np.pi / 4)
        assert mag == pytest.approx(np.sqrt(2))


# ── 4.4 State machine ─────────────────────────────────────────────────────────

class TestStateMachine:

    def test_default_show_flags(self, empty_plot):
        for flag in ('show_nodes', 'show_lines', 'show_nd_lines',
                     'show_cn_lines', 'show_traces', 'show_parent_childs',
                     'show_chan_dofs', 'show_axis'):
            assert getattr(empty_plot, flag) is True, flag
        assert empty_plot.animated is False
        assert empty_plot.data_animated is False

    def _mode_plot(self):
        """Two-node plot with hand-set mode shapes for amplitude/part tests."""
        geo = _simple_geometry({'A': (0.0, 0.0, 0.0), 'B': (1.0, 0.0, 0.0)})
        p = ModeShapePlot(geometry_data=geo)
        p.chan_dofs = [[0, 'A', 0.0, 0.0, 'cA'], [1, 'B', 0.0, 0.0, 'cB']]
        p.mode_shapes = np.zeros((2, 1, 1), dtype=complex)
        p.mode_shapes[:, 0, 0] = [1 + 0j, 1j]
        p.mode_index = (0, 0)
        p.real = False
        p.amplitude = 1.0
        return p

    def test_change_amplitude_scales_linearly(self):
        p = self._mode_plot()
        p.draw_msh()
        base = copy.deepcopy(p.disp_nodes)
        p.change_amplitude(3.0)
        for node in p.disp_nodes:
            np.testing.assert_allclose(
                p.disp_nodes[node], 3.0 * np.array(base[node]))

    def test_change_amplitude_noop_when_unchanged(self):
        p = self._mode_plot()
        p.draw_msh()
        before = copy.deepcopy(p.disp_nodes)
        assert p.change_amplitude(None) is None
        assert p.change_amplitude(p.amplitude) is None
        assert p.disp_nodes == before

    def test_change_part_flips_and_recomputes(self):
        p = self._mode_plot()
        p.draw_msh()
        # node B carries a genuine phase in complex mode
        assert p.phi_nodes['B'][0] == pytest.approx(np.pi / 2)
        p.change_part(True)
        assert p.real is True
        # real mode forces phase to zero everywhere
        np.testing.assert_allclose(p.phi_nodes['B'], [0.0, 0.0, 0.0])

    def test_change_part_noop_when_unchanged(self):
        p = self._mode_plot()
        p.draw_msh()
        before = copy.deepcopy(p.phi_nodes)
        assert p.change_part(False) is None
        assert p.phi_nodes == before

    def test_compute_node_bounds_equal_sided_cube(self):
        geo = _simple_geometry({'1': (0.0, 0.0, 0.0), '2': (4.0, 0.0, 0.0),
                                '3': (0.0, 2.0, 0.0), '4': (0.0, 0.0, 1.0)})
        p = ModeShapePlot(geometry_data=geo)
        xmin, xmax, ymin, ymax, zmin, zmax = p._compute_node_bounds()
        sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin
        assert sx == pytest.approx(sy) == pytest.approx(sz)
        # side equals the largest coordinate span (x: 4)
        assert sx == pytest.approx(4.0)
        # box is centred on the node cloud
        assert 0.5 * (xmin + xmax) == pytest.approx(2.0)
        assert 0.5 * (ymin + ymax) == pytest.approx(1.0)
        assert 0.5 * (zmin + zmax) == pytest.approx(0.5)

    def test_compute_node_bounds_empty_geometry(self):
        p = ModeShapePlot(geometry_data=GeometryProcessor())
        assert p._compute_node_bounds() == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


# ── 4.5 Data-animation math ───────────────────────────────────────────────────

class TestDataAnimationMath:

    def test_compute_data_disp_nodes(self, geometry_data, prep_signals):
        geo = _simple_geometry({'A': (0.0, 0.0, 0.0), 'B': (1.0, 0.0, 0.0)})
        plot = ModeShapePlot(geometry_data=geo, prep_signals=prep_signals)
        # channel 0 -> node A along x, channel 1 -> node B along y
        plot.chan_dofs = [[0, 'A', 0.0, 0.0, 'c0'], [1, 'B', 90.0, 0.0, 'c1']]
        plot.amplitude = 2.0
        num = 3

        disp = plot._compute_data_disp_nodes(num)

        xa, ya, za = calc_xyz(0.0, 0.0)
        xb, yb, zb = calc_xyz(90.0 * np.pi / 180, 0.0)
        sig_a = prep_signals.signals_filtered[num, 0] * 2.0
        sig_b = prep_signals.signals_filtered[num, 1] * 2.0
        np.testing.assert_allclose(disp['A'], [sig_a * xa, sig_a * ya, sig_a * za])
        np.testing.assert_allclose(disp['B'], [sig_b * xb, sig_b * yb, sig_b * zb])

    def test_compute_data_disp_nodes_accumulates(self, prep_signals):
        # two channels on the same node accumulate into one node vector
        geo = _simple_geometry({'A': (0.0, 0.0, 0.0)})
        plot = ModeShapePlot(geometry_data=geo, prep_signals=prep_signals)
        plot.chan_dofs = [[0, 'A', 0.0, 0.0, 'c0'], [1, 'A', 0.0, 0.0, 'c1']]
        plot.amplitude = 1.0
        num = 5
        disp = plot._compute_data_disp_nodes(num)
        xa, ya, za = calc_xyz(0.0, 0.0)
        s = (prep_signals.signals_filtered[num, 0]
             + prep_signals.signals_filtered[num, 1])
        np.testing.assert_allclose(disp['A'], [s * xa, s * ya, s * za])

    def test_compute_data_disp_nodes_skips_unknown_node(self, prep_signals):
        geo = _simple_geometry({'A': (0.0, 0.0, 0.0)})
        plot = ModeShapePlot(geometry_data=geo, prep_signals=prep_signals)
        # channel 1 assigned to a node not in the geometry -> ignored
        plot.chan_dofs = [[0, 'A', 0.0, 0.0, 'c0'], [1, 'ZZ', 0.0, 0.0, 'c1']]
        plot.amplitude = 1.0
        disp = plot._compute_data_disp_nodes(0)
        assert set(disp) == {'A'}
