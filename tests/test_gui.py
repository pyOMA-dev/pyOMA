"""
Qt GUI tests: StabilGUI and ModeShapeGUI.

All tests in this file require PyQt6 and are marked ``gui``.  They run
headless via ``QT_QPA_PLATFORM=offscreen`` (set in conftest.py).

To run only these tests::

    pytest -m gui

To skip them::

    pytest -m 'not gui'
"""
import pytest

# Skip the entire module when PyQt6 is absent (non-GUI installs).
pytest.importorskip('PyQt6', reason='PyQt6 not installed – pip install "pyOMA[gui]"')


# ── QApplication singleton ────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def qapp():
    """Headless Qt application, shared across all GUI tests."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


# ── Data fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def stabil_calc_gui(modal_data_ssi_cov):
    """StabilCluster with stabilisation masks – computed once per session."""
    from pyOMA.core.StabilDiagram import StabilCluster
    sc = StabilCluster(modal_data_ssi_cov)
    sc.calculate_stabilization_masks()
    return sc


@pytest.fixture
def stabil_plot_gui(stabil_calc_gui):
    """Fresh StabilPlot for each test (cheap; avoids shared canvas state)."""
    from pyOMA.core.StabilDiagram import StabilPlot
    return StabilPlot(stabil_calc_gui)


@pytest.fixture
def mode_shape_plot_gui(stabil_calc_gui, modal_data_ssi_cov, geometry_data,
                        prep_signals_with_corr):
    """Fresh ModeShapePlot for each test."""
    from pyOMA.core.PlotMSH import ModeShapePlot
    return ModeShapePlot(
        geometry_data=geometry_data,
        stabil_calc=stabil_calc_gui,
        modal_data=modal_data_ssi_cov,
        prep_signals=prep_signals_with_corr,
    )


# ── DataCursor ────────────────────────────────────────────────────────────────

class TestDataCursor:
    """Unit tests for DataCursor.add_callback – no QApplication needed."""

    @pytest.fixture
    def cursor(self, stabil_plot_gui):
        # StabilPlot.init_cursor() creates a DataCursor and wires it to the
        # figure's (Agg) canvas; the cursor object is returned.
        return stabil_plot_gui.init_cursor()

    def test_default_callbacks_present(self, cursor):
        assert 'show_current_info' in cursor.callbacks
        assert 'mode_selected' in cursor.callbacks
        assert 'mode_deselected' in cursor.callbacks

    def test_add_callback_registers_function(self, cursor):
        sentinel = []
        fn = sentinel.append  # store once – bound methods are not identical across accesses
        cursor.add_callback('show_current_info', fn)
        assert cursor.callbacks['show_current_info'] is fn

    def test_add_callback_rejects_unknown_key(self, cursor):
        with pytest.raises(ValueError):
            cursor.add_callback('nonexistent_event', lambda: None)

    def test_set_mask_stores_name(self, cursor, stabil_calc_gui):
        mask = stabil_calc_gui.get_stabilization_mask('mask_pre')
        cursor.set_mask(mask, 'mask_pre')
        assert cursor.name_mask == 'mask_pre'


# ── ComplexPlot ───────────────────────────────────────────────────────────────

@pytest.mark.gui
class TestComplexPlot:
    """ComplexPlot is a QMainWindow and requires a running QApplication."""

    @pytest.fixture
    def complex_plot(self, qapp):
        from pyOMA.GUI.StabilGUI import ComplexPlot
        cp = ComplexPlot()
        yield cp
        cp.close()

    def test_construction_succeeds(self, complex_plot):
        from pyOMA.GUI.StabilGUI import ComplexPlot
        assert isinstance(complex_plot, ComplexPlot)

    def test_plot_diagram_does_not_raise(self, complex_plot):
        """Regression: set_label_text() fix – plot_diagram() must not raise RuntimeError."""
        complex_plot.plot_diagram()

    def test_scatter_this_does_not_raise(self, complex_plot, stabil_calc_gui):
        """scatter_this() must draw a complex mode shape without error."""
        complex_plot.plot_diagram()
        msh = stabil_calc_gui.modal_data.mode_shapes[:, 0, 0]
        complex_plot.scatter_this(msh)


# ── StabilGUI ─────────────────────────────────────────────────────────────────

@pytest.mark.gui
class TestStabilGUI:
    """Integration tests for the full StabilGUI startup path."""

    @pytest.fixture
    def stabil_gui(self, qapp, stabil_plot_gui):
        from pyOMA.GUI.StabilGUI import StabilGUI, ComplexPlot
        from matplotlib.backend_bases import FigureCanvasBase
        cmpl_plot = ComplexPlot()
        gui = StabilGUI(stabil_plot_gui, cmpl_plot, msh_plot=None)
        yield gui
        gui.close()
        cmpl_plot.close()
        # Restore a non-Qt canvas so stabil_plot_gui remains usable after teardown.
        FigureCanvasBase(stabil_plot_gui.fig)

    def test_construction_succeeds(self, stabil_gui):
        from pyOMA.GUI.StabilGUI import StabilGUI
        assert isinstance(stabil_gui, StabilGUI)

    def test_window_is_visible(self, stabil_gui):
        assert stabil_gui.isVisible()

    def test_canvas_is_qt(self, stabil_gui):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        assert isinstance(stabil_gui.canvas, FigureCanvasQTAgg)

    def test_cursor_is_datacursor(self, stabil_gui):
        """Regression: init_cursor() must succeed (DataCursor.add_callback fix)."""
        from pyOMA.core.StabilDiagram import DataCursor
        assert isinstance(stabil_gui.cursor, DataCursor)

    def test_cursor_callback_wired(self, stabil_gui):
        """add_callback must have bound update_value_view to show_current_info."""
        # Use == not is: bound method wrappers are not identity-equal across accesses.
        assert (stabil_gui.cursor.callbacks['show_current_info']
                == stabil_gui.update_value_view)


# ── ModeShapeGUI ──────────────────────────────────────────────────────────────

@pytest.mark.gui
class TestModeShapeGUI:
    """Integration tests for ModeShapeGUI startup (FigureCanvasQTAgg fix)."""

    @pytest.fixture
    def msh_gui(self, qapp, mode_shape_plot_gui):
        from pyOMA.GUI.PlotMSHGUI import ModeShapeGUI
        from matplotlib.backend_bases import FigureCanvasBase
        gui = ModeShapeGUI(mode_shape_plot_gui)
        yield gui
        gui.close()
        # Restore non-Qt canvas on both the figure and the ModeShapePlot instance.
        base = FigureCanvasBase(mode_shape_plot_gui.fig)
        if hasattr(mode_shape_plot_gui, 'canvas'):
            mode_shape_plot_gui.canvas = base

    def test_construction_succeeds(self, msh_gui):
        from pyOMA.GUI.PlotMSHGUI import ModeShapeGUI
        assert isinstance(msh_gui, ModeShapeGUI)

    def test_window_is_visible(self, msh_gui):
        assert msh_gui.isVisible()

    def test_canvas_is_qt(self, msh_gui):
        """Regression: FigureCanvasQTAgg(fig) must replace the base canvas."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        assert isinstance(msh_gui.canvas, FigureCanvasQTAgg)

    def test_figure_canvas_matches_gui_canvas(self, msh_gui, mode_shape_plot_gui):
        """fig.canvas must be the same object as the canvas the GUI holds."""
        assert mode_shape_plot_gui.fig.canvas is msh_gui.canvas


# ── PlotMSHGUI Designer form (pytest-qt) ──────────────────────────────────────

pytest.importorskip('pytestqt', reason='pytest-qt not installed – pip install "pyOMA[dev]"')


@pytest.mark.gui
class TestPlotMSHGUIForm:
    """pytest-qt smoke test for the Designer-built ui/plot_msh.ui widget tree."""

    @pytest.fixture
    def msh_gui(self, qtbot, mode_shape_plot_gui):
        from pyOMA.GUI.PlotMSHGUI import ModeShapeGUI
        from matplotlib.backend_bases import FigureCanvasBase
        gui = ModeShapeGUI(mode_shape_plot_gui)
        qtbot.addWidget(gui)
        yield gui
        # qtbot.addWidget() already closes/deletes gui at teardown.
        base = FigureCanvasBase(mode_shape_plot_gui.fig)
        if hasattr(mode_shape_plot_gui, 'canvas'):
            mode_shape_plot_gui.canvas = base

    def test_key_widgets_have_expected_object_names(self, msh_gui):
        """Spot-check widgets from the B1 inventory carry their expected objectName."""
        expected = [
            'canvas', 'axis_checkbox', 'nodes_checkbox', 'line_checkbox',
            'ms_checkbox', 'chandof_checkbox', 'conn_lines_checkbox',
            'nd_lines_checkbox', 'traces_checkbox', 'mode_combo', 'amplitude_box',
            'real_checkbox', 'imag_checkbox', 'ani_button', 'reset_button',
            'tab_widget', 'info_box', 'x_limits_min_edit', 'zoom_plus_button',
        ]
        for name in expected:
            assert getattr(msh_gui, name).objectName() == name

    def test_reset_button_click_updates_axis_limit_field(self, qtbot, msh_gui):
        """A real click on reset_button must run the full wired path:
        widget -> released -> reset_view() -> ModeShapePlot -> back into
        the .ui-declared x_limits_min_edit field."""
        from PyQt6.QtCore import Qt
        qtbot.mouseClick(msh_gui.reset_button, Qt.MouseButton.LeftButton)
        text = msh_gui.x_limits_min_edit.text()
        assert text
        float(text)  # must be a valid, freshly-formatted number

    def test_lines_independent_of_parent_child_and_chan_dof_exclusivity(self, msh_gui):
        """Show Lines may be combined with either of the other two, while
        Show parent-childs Assignm. and Show Channel-DOF Assignm. remain
        mutually exclusive with each other."""
        def state():
            return (msh_gui.line_checkbox.isChecked(),
                    msh_gui.ms_checkbox.isChecked(),
                    msh_gui.chandof_checkbox.isChecked())

        assert state() == (True, False, False)

        msh_gui.ms_checkbox.click()
        assert state() == (True, True, False)

        msh_gui.chandof_checkbox.click()
        assert state() == (True, False, True)

        msh_gui.line_checkbox.click()
        assert state() == (False, False, True)

    def test_unchecking_hides_parent_childs_and_chan_dofs(self, msh_gui, mode_shape_plot_gui):
        """Regression: Qt.CheckState is a plain Enum in PyQt6 (always truthy),
        so ``if checkbox.checkState():`` in toggle_draw always took the
        "show" branch even when the box had just been unchecked."""
        def arrows_visible():
            return any(o.get_visible()
                       for patch in mode_shape_plot_gui.arrows_objects for o in patch)

        msh_gui.ms_checkbox.click()
        assert mode_shape_plot_gui.show_parent_childs is True
        assert arrows_visible() is True

        msh_gui.ms_checkbox.click()
        assert mode_shape_plot_gui.show_parent_childs is False
        assert arrows_visible() is False

        # This fixture's geometry has no channel-DOF assignments, so there
        # are no patches to check visibility on; the flag alone confirms
        # toggle_draw took the correct (i == 1) branch.
        msh_gui.chandof_checkbox.click()
        assert mode_shape_plot_gui.show_chan_dofs is True

        msh_gui.chandof_checkbox.click()
        assert mode_shape_plot_gui.show_chan_dofs is False


# ── StabilGUI Designer form (pytest-qt) ───────────────────────────────────────

@pytest.mark.gui
class TestStabilGUIForm:
    """pytest-qt smoke test for the Designer-built ui/stabil_gui.ui widget tree."""

    @pytest.fixture
    def stabil_gui(self, qtbot, stabil_plot_gui):
        from pyOMA.GUI.StabilGUI import StabilGUI, ComplexPlot
        from matplotlib.backend_bases import FigureCanvasBase
        cmpl_plot = ComplexPlot()
        qtbot.addWidget(cmpl_plot)
        gui = StabilGUI(stabil_plot_gui, cmpl_plot, msh_plot=None)
        qtbot.addWidget(gui)
        yield gui
        FigureCanvasBase(stabil_plot_gui.fig)

    def test_key_widgets_have_expected_object_names(self, stabil_gui):
        """Spot-check widgets from the widget inventory carry their expected objectName."""
        expected = [
            'canvas', 'mode_selector', 'plot_selector_c_checkbox',
            'plot_selector_msh_checkbox', 'mode_val_view_text',
            'current_value_view_text', 'df_edit', 'dd_edit', 'mac_edit',
            'mpc_edit', 'mpd_edit', 'MC_edit', 'apply_button',
            'save_figure_button', 'export_results_button', 'save_state_button',
            'ok_close_button', 'stable_pole_checkbox', 'all_poles_checkbox',
        ]
        for name in expected:
            assert getattr(stabil_gui, name).objectName() == name

    def test_current_value_view_is_visible(self, stabil_gui):
        """Regression: current_value_view was constructed and fed live hover
        info via update_value_view but never placed in any layout, so it was
        never actually visible to users."""
        assert stabil_gui.current_value_view_text.isVisible()

    def test_std_capability_hides_cov_rows(self, stabil_gui):
        """This fixture's StabilCluster has capabilities['std'] = False."""
        assert not stabil_gui.stabil_calc.capabilities['std']
        for widget in (stabil_gui.cov_freq_label, stabil_gui.stdf_edit,
                       stabil_gui.stdf_histo_button, stabil_gui.cov_damping_label,
                       stabil_gui.stdd_edit, stabil_gui.stdd_histo_button):
            assert not widget.isVisible()

    def test_f_d_msh_mc_auto_data_capabilities_show_rows(self, stabil_gui):
        """This fixture's StabilCluster has f/d/msh/MC/auto/data capabilities True."""
        for widget in (
                stabil_gui.freq_label, stabil_gui.df_edit, stabil_gui.df_histo_button,
                stabil_gui.damping_label, stabil_gui.dd_edit, stabil_gui.dd_histo_button,
                stabil_gui.damping_range_label, stabil_gui.d_min_edit,
                stabil_gui.d_max_edit, stabil_gui.dr_histo_button,
                stabil_gui.mac_label, stabil_gui.mac_edit, stabil_gui.mac_histo_button,
                stabil_gui.mpc_label, stabil_gui.mpc_edit, stabil_gui.mpc_histo_button,
                stabil_gui.mpd_label, stabil_gui.mpd_edit, stabil_gui.mpd_histo_button,
                stabil_gui.mc_label, stabil_gui.MC_edit, stabil_gui.show_mc_button,
                stabil_gui.clear_auto_button, stabil_gui.num_iter_edit,
                stabil_gui.classify_auto_button, stabil_gui.use_stabil_checkbox,
                stabil_gui.threshold_edit, stabil_gui.select_auto_button,
                stabil_gui.num_modes_edit, stabil_gui.show_psd_checkbox):
            assert widget.isVisible()

    def test_mtn_row_always_hidden(self, stabil_gui):
        """mtn is unimplemented in pyOMA/core (capabilities['mtn'] hardcoded
        off there) and mtn_histo_button has no method to call - stays hidden
        regardless of the capability flag."""
        assert not stabil_gui.stabil_calc.capabilities['mtn']
        for widget in (stabil_gui.mtn_label, stabil_gui.mtn_edit, stabil_gui.mtn_histo_button):
            assert not widget.isVisible()


# ── HistoPlot Designer form (pytest-qt) ───────────────────────────────────────

@pytest.mark.gui
class TestHistoPlotForm:
    """pytest-qt smoke test for the Designer-built ui/histo_plot.ui widget tree."""

    @pytest.fixture
    def histo_plot(self, qtbot):
        import numpy as np
        from pyOMA.GUI.StabilGUI import HistoPlot
        rng = np.random.default_rng(0)
        all_data = rng.normal(size=200)
        stabil_data = rng.normal(size=50)
        # select_ranges/select_callback given as single-element lists, matching
        # how every real create_histo_plot_* call site in StabilGUI uses this
        # class (e.g. create_histo_plot_f) - the bare no-args defaults
        # (select_ranges=[None]) hit an IndexError in
        # _should_setup_selector_lines and are never used in practice.
        hp = HistoPlot(all_data, stabil_data, title='Test Histogram',
                        select_ranges=[0.0], select_callback=[lambda x: None])
        qtbot.addWidget(hp)
        yield hp

    def test_key_widgets_have_expected_object_names(self, histo_plot):
        for name in ('canvas', 'lrange_box', 'urange_box'):
            assert getattr(histo_plot, name).objectName() == name

    def test_construction_populates_histogram(self, histo_plot):
        assert histo_plot.all_patches is not None
        assert histo_plot.stabil_patches is not None

    def test_close_hides_instead_of_destroying(self, histo_plot):
        """closeEvent is overridden to hide rather than destroy (visible flag)."""
        histo_plot.show()
        histo_plot.close()
        assert histo_plot.visible is False
