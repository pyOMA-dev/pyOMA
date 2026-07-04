"""
Unit tests for the PlotMSHWeb options box in pyOMA.GUI.JupyterGUI.

Regression coverage: the Jupyter mode-shape viewer only exposed checkboxes
for lines/nodes/axis/connecting-lines/non-displaced-lines/parent-child/
chan-dof, with no way to toggle the trace ellipses independently.
"""
import types

import pytest

ipywidgets = pytest.importorskip('ipywidgets', reason='ipywidgets not installed')
pytest.importorskip('ipympl', reason='ipympl not installed')

from pyOMA.GUI.JupyterGUI import _msh_build_optbox


def _fake_msp(**overrides):
    """Minimal stand-in for a ModeShapePlot exposing only the show_* flags."""
    defaults = dict(
        show_axis=True, show_nodes=True, show_lines=True,
        show_cn_lines=True, show_nd_lines=True,
        show_parent_childs=False, show_chan_dofs=False,
        show_traces=True,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestMshBuildOptbox:

    def test_returns_eight_checkboxes(self):
        _optbox, checkboxes = _msh_build_optbox(_fake_msp())
        assert len(checkboxes) == 8

    def test_traces_checkbox_reflects_flag(self):
        _optbox, checkboxes = _msh_build_optbox(_fake_msp(show_traces=False))
        cb8 = checkboxes[-1]
        assert cb8.description == 'Show Traces'
        assert cb8.value is False

    def test_traces_checkbox_included_in_optbox_children(self):
        optbox, checkboxes = _msh_build_optbox(_fake_msp())
        assert checkboxes[-1] in optbox.children
