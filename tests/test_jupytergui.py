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

import pyOMA.GUI.JupyterGUI as jupytergui
from pyOMA.GUI.JupyterGUI import (
    _msh_build_optbox, _msh_build_savebox, _is_pyvista_backend, PlotMSHWeb)


class TestBackendDispatch:
    """PlotMSHWeb routes on the mode-shape backend of its argument."""

    def test_pyvista_backend_is_detected_by_its_plotter(self):
        pv_like = types.SimpleNamespace(plotter=object())
        mpl_like = types.SimpleNamespace(fig=object(), subplot=object())
        assert _is_pyvista_backend(pv_like) is True
        assert _is_pyvista_backend(mpl_like) is False

    def test_plotmshweb_routes_pyvista_to_the_full_panel(self, monkeypatch):
        # The pyvista branch builds its control panel via _plotmshweb_pyvista
        # (a trame client view that cannot run head-less); check the routing.
        sentinel = object()
        monkeypatch.setattr(jupytergui, '_plotmshweb_pyvista',
                            lambda msp: sentinel)
        pv_like = types.SimpleNamespace(plotter=object())
        assert PlotMSHWeb(pv_like) is sentinel


class TestSaveFramesBox:
    """The save-frames control calls export_animation_frames."""

    def test_button_exports_with_folder_and_format(self):
        rec = {}

        class Msp:
            mode_index = (1, 0)

            def export_animation_frames(self, directory, fmt='png'):
                rec.update(directory=directory, fmt=fmt)
                return ['a', 'b']

        box = _msh_build_savebox(Msp())
        button = next(w for w in box.children if hasattr(w, 'on_click'))
        button.click()
        assert rec == {'directory': 'animation_frames', 'fmt': 'png'}

    def test_button_refuses_when_no_mode_selected(self):
        class Msp:
            mode_index = None

            def export_animation_frames(self, *a, **k):
                raise AssertionError('must not export without a mode')

        box = _msh_build_savebox(Msp())
        next(w for w in box.children if hasattr(w, 'on_click')).click()
        assert 'mode' in box.children[-1].value.lower()


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
