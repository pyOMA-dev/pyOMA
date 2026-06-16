"""
Qt GUI tests: StabilGUI and ModeShapeGUI.

All tests in this file require PyQt5 and are marked ``gui``.  They run
headless via ``QT_QPA_PLATFORM=offscreen`` (set in conftest.py).

To run only these tests::

    pytest -m gui

To skip them::

    pytest -m 'not gui'
"""
import pytest

# Skip the entire module when PyQt5 is absent (non-GUI installs).
pytest.importorskip('PyQt5', reason='PyQt5 not installed – pip install "pyOMA[gui]"')


# ── QApplication singleton ────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def qapp():
    """Headless Qt application, shared across all GUI tests."""
    from PyQt5.QtWidgets import QApplication
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
        with pytest.raises(AssertionError):
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
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
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
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        assert isinstance(msh_gui.canvas, FigureCanvasQTAgg)

    def test_figure_canvas_matches_gui_canvas(self, msh_gui, mode_shape_plot_gui):
        """fig.canvas must be the same object as the canvas the GUI holds."""
        assert mode_shape_plot_gui.fig.canvas is msh_gui.canvas
