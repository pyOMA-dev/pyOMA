"""
Unit tests for pyOMA.core.PlotMSH.ModeShapePlot's trace-ellipse drawing
and the show_traces visibility flag.

Regression coverage for two bugs:
  * _animate_draw_traces() early-returned whenever trace_objects was
    still empty, which is always true on the first call, so trace
    ellipses never appeared.
  * Trace visibility was piggy-backing on show_cn_lines instead of its
    own flag, so it could not be toggled independently.
"""
import pytest

from pyOMA.core.PlotMSH import ModeShapePlot
from pyOMA.core.PreProcessingTools import GeometryProcessor


@pytest.fixture
def msh_plot(geometry_data, prep_signals):
    """Minimal ModeShapePlot (no modal_data/stabil_calc) with one
    chan_dof so _animate_draw_traces has a moving node to plot."""
    node = next(iter(geometry_data.nodes))
    prep_signals.chan_dofs = [(0, node, 0.0, 0.0)]
    plot = ModeShapePlot(geometry_data=geometry_data, prep_signals=prep_signals)
    plot.disp_nodes[node] = [1.0, 0.5, 0.2]
    plot.phi_nodes[node] = [0.0, 0.1, 0.2]
    return plot


class TestAnimateDrawTraces:

    def test_first_call_creates_traces(self, msh_plot):
        assert msh_plot.trace_objects == []
        msh_plot._animate_draw_traces()
        assert len(msh_plot.trace_objects) == 1

    def test_second_call_replaces_traces(self, msh_plot):
        msh_plot._animate_draw_traces()
        first = msh_plot.trace_objects[0]
        msh_plot._animate_draw_traces()
        assert len(msh_plot.trace_objects) == 1
        assert msh_plot.trace_objects[0] is not first


class TestRefreshLinesNewNode:
    """Regression: adding a node/line to geometry_data *after* the
    ModeShapePlot was constructed used to raise KeyError in refresh_lines,
    since disp_nodes/phi_nodes were snapshotted once at __init__ and never
    grew with the geometry (unlike refresh_nodes/refresh_cn_lines, which
    already used dict.get with a default)."""

    def test_reset_view_after_adding_node_and_line(self):
        geo = GeometryProcessor()
        geo.add_node('1', [0.0, 0.0, 0.0])
        geo.add_node('2', [1.0, 0.0, 0.0])
        plot = ModeShapePlot(geometry_data=geo)

        geo.add_node('3', [2.0, 0.0, 0.0])
        geo.add_line(['2', '3'])

        plot.reset_view()  # must not raise KeyError: '3'


class TestShowTracesFlag:

    def test_default_true_and_independent_of_cn_lines(self, msh_plot):
        assert msh_plot.show_traces is True
        msh_plot.refresh_cn_lines(False)
        assert msh_plot.show_traces is True

    def test_refresh_traces_toggles_visibility(self, msh_plot):
        msh_plot._animate_draw_traces()
        msh_plot.refresh_traces(False)
        assert msh_plot.show_traces is False
        assert all(not t.get_visible() for t in msh_plot.trace_objects)
        msh_plot.refresh_traces(True)
        assert msh_plot.show_traces is True
        assert all(t.get_visible() for t in msh_plot.trace_objects)
