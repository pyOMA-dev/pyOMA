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
            'canvas', 'mode_selector', 'msh_toggle_button',
            'mode_val_view_text',
            'current_value_view_text', 'df_edit', 'dd_edit', 'mac_edit',
            'mpc_edit', 'mpd_edit', 'MC_edit',
            'save_figure_button', 'export_results_button',
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

    def test_pane_toggle_buttons_hide_containers(self, stabil_gui):
        """left_pane_widget/right_pane_widget must exist and respond to the
        left/right toggle buttons (regression: these containers were briefly
        removed from stabil_gui.ui, which broke setVisible() toggling)."""
        assert stabil_gui.left_pane_widget.isVisible()
        stabil_gui.left_toggle_btn.setChecked(False)
        assert not stabil_gui.left_pane_widget.isVisible()

        assert stabil_gui.right_pane_widget.isVisible()
        stabil_gui.right_toggle_btn.setChecked(False)
        assert not stabil_gui.right_pane_widget.isVisible()

    def test_cmpl_plot_embedded_in_right_pane(self, stabil_gui):
        """The ComplexPlot window must be reparented into right_pane_widget,
        not left as a separate top-level window."""
        assert stabil_gui.cmpl_plot.parent() is stabil_gui.right_pane_widget

    def test_state_menu_actions_wired(self, stabil_gui):
        """actionSave_State/actionLoad_State must trigger save_state/load_state."""
        assert stabil_gui.actionSave_State.receivers(stabil_gui.actionSave_State.triggered) > 0
        assert stabil_gui.actionLoad_State.receivers(stabil_gui.actionLoad_State.triggered) > 0


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


# ── PreProcessSignalsGUI Designer form (pytest-qt) ────────────────────────────

@pytest.mark.gui
class TestPreProcessSignalsGUIForm:
    """pytest-qt smoke test for the Designer-built ui/preprocess_signals.ui widget tree."""

    @pytest.fixture
    def preprocess_gui(self, qtbot, prep_signals):
        from pyOMA.GUI.PreProcessSignalsGUI import PreProcessSignalsGUI
        gui = PreProcessSignalsGUI(prep_signals)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, preprocess_gui):
        expected = [
            'canvas_time', 'canvas_freq', 'channel_table', 'chk_auto_ref',
            'btn_delete_channels', 'btn_undo', 'combo_time_diagram', 'stack_time_params',
            'btn_correct_offset', 'btn_precondition',
            'btn_add_noise', 'chk_lowpass', 'chk_highpass', 'combo_ftype',
            'lbl_rp', 'lbl_rs', 'btn_apply_filter', 'spin_decimate_factor',
            'btn_decimate', 'lbl_signal_clarity_score', 'btn_compute_clarity_score',
            'save_figure_button', 'ok_close_button',
        ]
        for name in expected:
            assert getattr(preprocess_gui, name).objectName() == name

    def test_state_menu_actions_wired(self, preprocess_gui):
        assert preprocess_gui.actionSave_State.receivers(
            preprocess_gui.actionSave_State.triggered) > 0
        assert preprocess_gui.actionLoad_State.receivers(
            preprocess_gui.actionLoad_State.triggered) > 0

    def test_save_state_then_load_state_round_trip(
            self, preprocess_gui, prep_signals, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        fname = tmp_path / 'state.npz'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(fname), ''))
        preprocess_gui.save_state()
        assert fname.exists()

        setup_name_before = preprocess_gui.prep_signals.setup_name
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(fname), ''))
        preprocess_gui.load_state()
        assert preprocess_gui.prep_signals is not prep_signals
        assert preprocess_gui.prep_signals.setup_name == setup_name_before
        assert preprocess_gui.signal_plot.prep_signals is preprocess_gui.prep_signals

    def test_import_signals_npz(self, preprocess_gui, prep_signals, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        fname = tmp_path / "signals.npz"
        prep_signals.save_state(str(fname))
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(fname), ''))
        preprocess_gui.import_signals()
        assert preprocess_gui.prep_signals.sampling_rate == prep_signals.sampling_rate

    def test_import_signals_npy_prompts_for_sampling_rate(
            self, preprocess_gui, prep_signals, tmp_path, monkeypatch):
        import numpy as np
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        fname = tmp_path / "signals.npy"
        np.save(fname, prep_signals.signals)
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(fname), ''))
        monkeypatch.setattr(QInputDialog, 'getDouble', lambda *a, **k: (128.0, True))
        preprocess_gui.import_signals()
        assert preprocess_gui.prep_signals.sampling_rate == 128.0

    def test_import_signals_npy_cancelled_dialog_does_nothing(
            self, preprocess_gui, prep_signals, tmp_path, monkeypatch):
        import numpy as np
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        fname = tmp_path / "signals.npy"
        np.save(fname, prep_signals.signals)
        original = preprocess_gui.prep_signals
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(fname), ''))
        monkeypatch.setattr(QInputDialog, 'getDouble', lambda *a, **k: (0.0, False))
        preprocess_gui.import_signals()
        assert preprocess_gui.prep_signals is original

    def test_save_figure_writes_time_and_freq_files(self, preprocess_gui, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        base = tmp_path / 'plot.png'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(base), ''))
        preprocess_gui.save_figure()
        assert (tmp_path / 'plot_time.png').exists()
        assert (tmp_path / 'plot_freq.png').exists()

    def test_no_navigation_toolbar(self, preprocess_gui):
        """Regression: plot_layout must only contain the two canvases, no
        NavigationToolbar2QT."""
        assert not hasattr(preprocess_gui, 'toolbar_time')
        assert not hasattr(preprocess_gui, 'toolbar_freq')
        assert preprocess_gui.plot_layout.count() == 2

    def test_channel_table_has_generous_minimum_height(self, preprocess_gui):
        assert preprocess_gui.channel_table.minimumHeight() >= 200

    def test_undo_button_starts_disabled(self, preprocess_gui):
        """Single-step undo is a stub (PreProcessSignals.undo_available is
        always False for now) - the button must reflect that."""
        assert preprocess_gui.btn_undo.isEnabled() is False

    def test_close_deletes_window_so_blocking_event_loops_can_exit(self, qtbot, prep_signals):
        """Regression: start_preprocess_gui() blocks on
        `form.destroyed.connect(loop.quit)` - without a closeEvent that
        calls deleteLater(), closing the window only hides it and the
        script never continues. Uses its own instance (not the shared
        `preprocess_gui`/qtbot.addWidget fixture) since this test
        deliberately destroys the widget."""
        from pyOMA.GUI.PreProcessSignalsGUI import PreProcessSignalsGUI
        gui = PreProcessSignalsGUI(prep_signals)
        with qtbot.waitSignal(gui.destroyed, timeout=1000):
            gui.close()

    def test_channel_table_populated_and_fully_selected(self, preprocess_gui, prep_signals):
        n = prep_signals.num_analised_channels
        assert preprocess_gui.channel_table.rowCount() == n
        assert len(preprocess_gui.channel_table.selectionModel().selectedRows()) == n
        assert preprocess_gui._selected_channels() == list(range(n))

        preprocess_gui.channel_table.clearSelection()
        assert preprocess_gui._selected_channels() is None  # no selection -> None means "all"

    def test_channel_table_reflects_type_and_reference(self, preprocess_gui, prep_signals):
        # prep_signals fixture: ref_channels=[5], all channels default to acceleration
        assert preprocess_gui.channel_table.cellWidget(0, 1).currentText() == 'Acceleration'
        assert preprocess_gui.channel_table.cellWidget(0, 2).isChecked() is False
        assert preprocess_gui.channel_table.cellWidget(5, 2).isChecked() is True

    def test_changing_type_combo_updates_prep_signals(self, preprocess_gui, prep_signals, qtbot):
        combo = preprocess_gui.channel_table.cellWidget(0, 1)
        with qtbot.waitSignal(combo.currentTextChanged, timeout=1000):
            combo.setCurrentText('Velocity')
        qtbot.wait(10)  # let the deferred QTimer.singleShot table rebuild run
        assert 0 in prep_signals.velo_channels
        assert 0 not in prep_signals.accel_channels

    def test_toggling_reference_checkbox_updates_prep_signals(self, preprocess_gui, prep_signals):
        checkbox = preprocess_gui.channel_table.cellWidget(1, 2)
        checkbox.setChecked(True)
        assert 1 in prep_signals.ref_channels
        checkbox.setChecked(False)
        assert 1 not in prep_signals.ref_channels

    def test_editing_name_cell_renames_channel(self, preprocess_gui, prep_signals):
        item = preprocess_gui.channel_table.item(0, 0)
        item.setText('new_name')
        assert prep_signals.channel_headers[0] == 'new_name'

    def test_renaming_to_duplicate_name_warns_and_reverts(
            self, preprocess_gui, prep_signals, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: None)
        original_name = prep_signals.channel_headers[0]
        other_name = str(prep_signals.channel_headers[1])
        item = preprocess_gui.channel_table.item(0, 0)
        item.setText(other_name)
        assert prep_signals.channel_headers[0] == original_name
        assert preprocess_gui.channel_table.item(0, 0).text() == str(original_name)

    def test_renaming_to_empty_name_reverts_without_error(
            self, preprocess_gui, prep_signals):
        original_name = prep_signals.channel_headers[0]
        item = preprocess_gui.channel_table.item(0, 0)
        item.setText('')
        assert prep_signals.channel_headers[0] == original_name
        assert preprocess_gui.channel_table.item(0, 0).text() == str(original_name)

    def test_delete_selected_channels(self, preprocess_gui, prep_signals):
        n_before = prep_signals.num_analised_channels
        preprocess_gui.channel_table.selectRow(0)
        preprocess_gui._on_delete_channels()
        assert prep_signals.num_analised_channels == n_before - 1
        assert preprocess_gui.channel_table.rowCount() == n_before - 1

    def test_delete_with_no_selection_warns_and_does_not_delete(self, preprocess_gui, prep_signals, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: None)
        n_before = prep_signals.num_analised_channels
        preprocess_gui.channel_table.clearSelection()
        preprocess_gui._on_delete_channels()
        assert prep_signals.num_analised_channels == n_before

    def test_svd_scale_disables_channel_table(self, preprocess_gui):
        preprocess_gui.combo_psd_scale.setCurrentText('svd')
        assert not preprocess_gui.channel_table.isEnabled()
        assert not preprocess_gui.chk_auto_ref.isEnabled()
        preprocess_gui.combo_psd_scale.setCurrentText('db')
        assert preprocess_gui.channel_table.isEnabled()

    def test_freq_plot_falls_back_to_default_nlines_on_first_run(
            self, preprocess_gui, prep_signals):
        """Regression: with 'Auto n_lines' checked (the default) and no PSD
        ever computed on a fresh prep_signals, plot_psd() used to bubble up
        RuntimeError('Either n_lines or n_segments must be provided on
        first run.') from psd_welch, which _update_freq_plot renders as a
        "Could not plot" error message instead of an actual PSD."""
        assert preprocess_gui.chk_psd_auto_nlines.isChecked()
        assert preprocess_gui.combo_psd_method.currentText() == 'auto'
        ax = preprocess_gui.canvas_freq.figure.axes[0]
        assert ax.lines
        assert prep_signals.n_lines_wl == preprocess_gui.spin_psd_nlines.value()

    def test_switching_to_welch_with_auto_nlines_does_not_raise(
            self, preprocess_gui, prep_signals):
        """Same regression as above, triggered by explicitly selecting the
        'welch' method (rather than 'auto') after opening the GUI."""
        preprocess_gui.combo_psd_method.setCurrentText('welch')
        ax = preprocess_gui.canvas_freq.figure.axes[0]
        assert ax.lines
        assert prep_signals.n_lines_wl == preprocess_gui.spin_psd_nlines.value()

    def test_cycling_time_diagram_types_does_not_raise(self, preprocess_gui):
        for index in range(preprocess_gui.stack_time_params.count()):
            preprocess_gui.combo_time_diagram.setCurrentIndex(index)
            assert preprocess_gui.stack_time_params.currentIndex() == index

    def test_preprocessing_actions_refresh_plot_without_error(self, preprocess_gui, prep_signals):
        """Each pre-processing action mutates prep_signals in place and must
        be followed by a successful status/plot refresh."""
        preprocess_gui._on_correct_offset()
        preprocess_gui._on_precondition()

        preprocess_gui.spin_noise_amplitude.setValue(0.01)
        preprocess_gui._on_add_noise()

        preprocess_gui.chk_lowpass.setChecked(True)
        preprocess_gui.spin_lowpass.setValue(10)
        preprocess_gui._on_filter()

        sampling_rate_before = prep_signals.sampling_rate
        preprocess_gui._on_decimate()
        assert prep_signals.sampling_rate == sampling_rate_before / 2

    def test_clarity_score_shows_na_for_singular_correlation_matrix(self, preprocess_gui):
        """prep_signals fixture: channel 0 is the fixed end of the
        synthetic rod (zero response), which makes the correlation
        matrix singular. signal_clarity_score() raises LinAlgError for
        it - the widget must show 'n/a' rather than crash on init."""
        assert preprocess_gui.lbl_signal_clarity_score.text() == 'n/a'

    def test_clarity_score_updates_after_processing_and_button_click(
            self, preprocess_gui, prep_signals):
        preprocess_gui.channel_table.selectRow(0)
        preprocess_gui._on_delete_channels()  # drops the zero-variance channel
        score_text = preprocess_gui.lbl_signal_clarity_score.text()
        assert score_text != 'n/a'
        float(score_text)  # now a real, finite score

        preprocess_gui.lbl_signal_clarity_score.setText('stale')
        preprocess_gui.btn_compute_clarity_score.click()
        assert preprocess_gui.lbl_signal_clarity_score.text() == score_text

    def test_ftype_change_toggles_rprs_row(self, preprocess_gui):
        assert not preprocess_gui.spin_rp.isEnabled()
        assert not preprocess_gui.lbl_rp.isEnabled()
        preprocess_gui.combo_ftype.setCurrentText('cheby1')
        assert preprocess_gui.spin_rp.isEnabled()
        assert preprocess_gui.spin_rs.isEnabled()
        assert preprocess_gui.lbl_rp.isEnabled()

    def test_auto_checkbox_toggles_spinbox_enabled(self, preprocess_gui):
        assert not preprocess_gui.spin_lowpass.isEnabled()
        preprocess_gui.chk_lowpass.setChecked(True)
        assert preprocess_gui.spin_lowpass.isEnabled()

        assert not preprocess_gui.spin_corr_mlags.isEnabled()  # chk_corr_auto_mlags starts checked
        preprocess_gui.chk_corr_auto_mlags.setChecked(False)
        assert preprocess_gui.spin_corr_mlags.isEnabled()


# ── GeometryProcessorGUI Designer form (pytest-qt) ────────────────────────────

@pytest.fixture
def fresh_geometry_data(test_files_dir):
    """Function-scoped fresh GeometryProcessor - safe for mutation tests
    (unlike the shared session-scoped `geometry_data` fixture)."""
    from pyOMA.core.PreProcessingTools import GeometryProcessor
    return GeometryProcessor.load_geometry(
        nodes_file=test_files_dir / 'grid.txt',
        lines_file=test_files_dir / 'lines.txt',
        parent_childs_file=test_files_dir / 'parent_child_assignments.txt',
    )


@pytest.mark.gui
class TestGeometryProcessorGUIForm:
    """pytest-qt smoke test for the Designer-built ui/geometry_processor.ui widget tree."""

    @pytest.fixture
    def geo_gui(self, qtbot, fresh_geometry_data):
        from pyOMA.GUI.GeometryProcessorGUI import GeometryProcessorGUI
        gui = GeometryProcessorGUI(fresh_geometry_data)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, geo_gui):
        expected = [
            'canvas', 'node_table', 'line_table', 'pc_table',
            'btn_add_node', 'btn_delete_node', 'btn_add_line', 'btn_delete_line',
            'btn_add_pc', 'btn_delete_pc', 'viewport_button_x', 'viewport_button_iso',
            'reset_button', 'load_state_button', 'save_state_button',
            'save_figure_button', 'ok_close_button',
        ]
        for name in expected:
            assert getattr(geo_gui, name).objectName() == name

    def test_tables_populated_from_geometry_data(self, geo_gui, fresh_geometry_data):
        assert geo_gui.node_table.rowCount() == len(fresh_geometry_data.nodes)
        assert geo_gui.line_table.rowCount() == len(fresh_geometry_data.lines)
        assert geo_gui.pc_table.rowCount() == len(fresh_geometry_data.parent_childs)

    def test_canvas_is_qt(self, geo_gui):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        assert isinstance(geo_gui.canvas, FigureCanvasQTAgg)

    def test_add_node_updates_geometry_and_table(self, geo_gui, fresh_geometry_data, monkeypatch):
        from PyQt6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **k: ('brand_new', True))
        n_before = len(fresh_geometry_data.nodes)
        geo_gui._on_add_node()
        assert 'brand_new' in fresh_geometry_data.nodes
        assert geo_gui.node_table.rowCount() == n_before + 1

    def test_add_node_rejects_duplicate_name(self, geo_gui, fresh_geometry_data, monkeypatch):
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        existing = next(iter(fresh_geometry_data.nodes))
        monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **k: (existing, True))
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: None)
        n_before = len(fresh_geometry_data.nodes)
        geo_gui._on_add_node()
        assert len(fresh_geometry_data.nodes) == n_before

    def test_editing_node_coordinates_updates_geometry(self, geo_gui, fresh_geometry_data):
        name = geo_gui.node_table.item(0, 0).text()
        geo_gui.node_table.item(0, 1).setText('42.0')
        assert fresh_geometry_data.nodes[name][0] == 42.0

    def test_delete_node_cascades_connected_lines(self, geo_gui, fresh_geometry_data):
        name = geo_gui.node_table.item(0, 0).text()
        for row in range(geo_gui.node_table.rowCount()):
            if geo_gui.node_table.item(row, 0).text() == name:
                geo_gui.node_table.selectRow(row)
                break
        geo_gui._on_delete_node()
        assert name not in fresh_geometry_data.nodes
        assert all(name not in line for line in fresh_geometry_data.lines)

    def test_add_and_delete_line(self, geo_gui, fresh_geometry_data):
        n_before = len(fresh_geometry_data.lines)
        geo_gui._on_add_line()
        assert len(fresh_geometry_data.lines) == n_before + 1
        geo_gui.line_table.selectRow(geo_gui.line_table.rowCount() - 1)
        geo_gui._on_delete_line()
        assert len(fresh_geometry_data.lines) == n_before

    def test_add_and_delete_parent_child(self, geo_gui, fresh_geometry_data):
        n_before = len(fresh_geometry_data.parent_childs)
        geo_gui._on_add_pc()
        assert len(fresh_geometry_data.parent_childs) == n_before + 1
        geo_gui.pc_table.selectRow(geo_gui.pc_table.rowCount() - 1)
        geo_gui._on_delete_pc()
        assert len(fresh_geometry_data.parent_childs) == n_before

    def test_save_then_reload_round_trip(self, geo_gui, fresh_geometry_data, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        nodes_file = tmp_path / 'nodes.txt'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(nodes_file), ''))
        geo_gui._on_save_nodes()
        assert nodes_file.exists()

        removed_name = next(iter(fresh_geometry_data.nodes))
        fresh_geometry_data.take_node(removed_name)
        assert removed_name not in fresh_geometry_data.nodes

        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(nodes_file), ''))
        geo_gui._on_load_nodes()
        assert removed_name in fresh_geometry_data.nodes

    def test_load_merges_without_clearing_existing_data(
            self, geo_gui, fresh_geometry_data, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        extra_nodes_file = tmp_path / 'extra_nodes.txt'
        extra_nodes_file.write_text('node_name\tx\ty\tz\nZZZ\t9.0\t9.0\t9.0\n')
        n_before = len(fresh_geometry_data.nodes)

        monkeypatch.setattr(
            QFileDialog, 'getOpenFileName', lambda *a, **k: (str(extra_nodes_file), ''))
        geo_gui._on_load_nodes()
        assert len(fresh_geometry_data.nodes) == n_before + 1
        assert 'ZZZ' in fresh_geometry_data.nodes

    def test_save_state_then_load_state_round_trip(
            self, geo_gui, fresh_geometry_data, tmp_path, monkeypatch):
        """save_state/load_state bundle nodes+lines+parent_childs into one
        directory (nodes.txt/lines.txt/parent_childs.txt), reusing the
        existing save_geometry/load_geometry pair."""
        from PyQt6.QtWidgets import QFileDialog
        nodes_before = dict(fresh_geometry_data.nodes)
        lines_before = list(fresh_geometry_data.lines)
        pcs_before = list(fresh_geometry_data.parent_childs)

        monkeypatch.setattr(
            QFileDialog, 'getExistingDirectory', lambda *a, **k: str(tmp_path))
        geo_gui.save_state()
        assert (tmp_path / 'nodes.txt').exists()
        assert (tmp_path / 'lines.txt').exists()
        assert (tmp_path / 'parent_childs.txt').exists()

        fresh_geometry_data.nodes.clear()
        fresh_geometry_data.lines.clear()
        fresh_geometry_data.parent_childs.clear()

        geo_gui.load_state()
        assert fresh_geometry_data.nodes == nodes_before
        assert fresh_geometry_data.lines == lines_before
        assert fresh_geometry_data.parent_childs == pcs_before

    def test_load_state_warns_when_directory_has_no_nodes_file(
            self, geo_gui, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        monkeypatch.setattr(
            QFileDialog, 'getExistingDirectory', lambda *a, **k: str(tmp_path))
        warnings = []
        monkeypatch.setattr(
            QMessageBox, 'warning', lambda *a, **k: warnings.append(a))
        geo_gui.load_state()
        assert len(warnings) == 1


# ── PreProcessSignalsGUI DOF column (pytest-qt) ───────────────────────────────

@pytest.mark.gui
class TestPreProcessSignalsGUIDofColumn:
    """"Add DOF" button in channel_table, wired to ChanDofEditorGUI."""

    def test_dof_button_disabled_without_geometry_data(self, qtbot, prep_signals):
        from pyOMA.GUI.PreProcessSignalsGUI import PreProcessSignalsGUI
        gui = PreProcessSignalsGUI(prep_signals)
        qtbot.addWidget(gui)
        btn = gui.channel_table.cellWidget(0, 3)
        assert btn.isEnabled() is False

    def test_dof_button_enabled_with_geometry_data(self, qtbot, prep_signals, geometry_data):
        from pyOMA.GUI.PreProcessSignalsGUI import PreProcessSignalsGUI
        gui = PreProcessSignalsGUI(prep_signals, geometry_data)
        qtbot.addWidget(gui)
        assert gui.channel_table.columnCount() == 4
        btn = gui.channel_table.cellWidget(0, 3)
        assert btn.isEnabled() is True

    def test_clicking_dof_button_opens_editor_for_correct_channel(
            self, qtbot, prep_signals, geometry_data, monkeypatch):
        from PyQt6.QtWidgets import QDialog
        from pyOMA.GUI.PreProcessSignalsGUI import PreProcessSignalsGUI
        from pyOMA.GUI.ChanDofEditorGUI import ChanDofEditorGUI
        gui = PreProcessSignalsGUI(prep_signals, geometry_data)
        qtbot.addWidget(gui)

        captured = {}

        def fake_exec(self):
            captured['dialog'] = self
            return QDialog.DialogCode.Rejected
        monkeypatch.setattr(QDialog, 'exec', fake_exec)

        btn = gui.channel_table.cellWidget(2, 3)
        btn.click()
        assert isinstance(captured['dialog'], ChanDofEditorGUI)
        assert captured['dialog'].channel == 2


# ── ChanDofEditorGUI Designer form (pytest-qt) ────────────────────────────────

@pytest.mark.gui
class TestChanDofEditorGUIForm:
    """pytest-qt smoke test for the Designer-built ui/chan_dof_editor.ui widget tree."""

    @pytest.fixture
    def dof_editor(self, qtbot, prep_signals, geometry_data):
        from pyOMA.GUI.ChanDofEditorGUI import ChanDofEditorGUI
        dlg = ChanDofEditorGUI(prep_signals, geometry_data, channel=0)
        qtbot.addWidget(dlg)
        yield dlg

    def test_key_widgets_have_expected_object_names(self, dof_editor):
        expected = ['canvas', 'combo_node', 'spin_az', 'spin_elev',
                    'btn_delete', 'btn_ok', 'btn_cancel']
        for name in expected:
            assert getattr(dof_editor, name).objectName() == name

    def test_node_combo_populated_from_geometry(self, dof_editor, geometry_data):
        assert dof_editor.combo_node.count() == len(geometry_data.nodes)

    def test_delete_disabled_when_no_existing_assignment(self, dof_editor):
        assert dof_editor.btn_delete.isEnabled() is False

    def test_existing_assignment_is_preloaded_and_delete_enabled(
            self, qtbot, prep_signals, geometry_data):
        from pyOMA.GUI.ChanDofEditorGUI import ChanDofEditorGUI
        node = sorted(geometry_data.nodes.keys())[0]
        prep_signals.set_chan_dof(0, node, 45.0, 10.0)
        dlg = ChanDofEditorGUI(prep_signals, geometry_data, channel=0)
        qtbot.addWidget(dlg)
        assert dlg.combo_node.currentText() == node
        assert dlg.spin_az.value() == 45.0
        assert dlg.spin_elev.value() == 10.0
        assert dlg.btn_delete.isEnabled() is True

    def test_ok_commits_the_assignment(self, dof_editor, prep_signals, geometry_data):
        node = sorted(geometry_data.nodes.keys())[1]
        dof_editor.combo_node.setCurrentText(node)
        dof_editor.spin_az.setValue(90.0)
        dof_editor.spin_elev.setValue(-15.0)
        dof_editor._on_ok()
        assert prep_signals.get_chan_dof(0) == (node, 90.0, -15.0)

    def test_ok_without_a_node_warns_and_does_not_commit(
            self, dof_editor, prep_signals, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: None)
        dof_editor.combo_node.setCurrentIndex(-1)
        dof_editor._on_ok()
        assert prep_signals.get_chan_dof(0) is None

    def test_cancel_discards_changes(self, dof_editor, prep_signals):
        dof_editor.spin_az.setValue(123.0)
        dof_editor.reject()
        assert prep_signals.get_chan_dof(0) is None

    def test_delete_removes_existing_assignment(
            self, qtbot, prep_signals, geometry_data):
        from pyOMA.GUI.ChanDofEditorGUI import ChanDofEditorGUI
        node = sorted(geometry_data.nodes.keys())[0]
        prep_signals.set_chan_dof(0, node, 0.0, 0.0)
        dlg = ChanDofEditorGUI(prep_signals, geometry_data, channel=0)
        qtbot.addWidget(dlg)
        dlg._on_delete()
        assert prep_signals.get_chan_dof(0) is None

    def test_draft_arrow_is_highlighted(self, dof_editor):
        """Regression: the currently-edited assignment must render in a
        different color than the rest of the geometry preview."""
        import matplotlib.colors as mcolors
        from pyOMA.GUI.ChanDofEditorGUI import _HIGHLIGHT_COLOR
        node = dof_editor.combo_node.itemText(0)
        dof_editor.combo_node.setCurrentText(node)
        assert dof_editor.mode_shape_plot.channels_objects
        arrow = dof_editor.mode_shape_plot.channels_objects[-1]
        assert mcolors.same_color(arrow.get_edgecolor(), _HIGHLIGHT_COLOR)

    def test_other_channels_assignments_shown_undedited(
            self, qtbot, prep_signals, geometry_data):
        """The other channel's own assignment must still appear in the
        background context (not accidentally excluded)."""
        from pyOMA.GUI.ChanDofEditorGUI import ChanDofEditorGUI
        node = sorted(geometry_data.nodes.keys())[0]
        prep_signals.set_chan_dof(1, node, 0.0, 0.0)
        dlg = ChanDofEditorGUI(prep_signals, geometry_data, channel=0)
        qtbot.addWidget(dlg)
        assert any(cd[0] == 1 for cd in dlg.mode_shape_plot.chan_dofs)

    def test_parent_child_assignments_are_not_shown(self, dof_editor, geometry_data):
        """Parent-child assignments are unrelated to channel-DOF assignment
        and must not clutter this preview."""
        assert geometry_data.parent_childs  # fixture actually has some
        assert dof_editor.mode_shape_plot.show_parent_childs is False
        for arrow_pair in dof_editor.mode_shape_plot.arrows_objects:
            for arrow in arrow_pair:
                assert arrow.get_visible() is False


# ── PRCEWidget Designer form (pytest-qt) ──────────────────────────────────────

@pytest.mark.gui
class TestPRCEWidgetForm:
    """pytest-qt smoke test for the Designer-built ui/prce.ui widget tree."""

    @pytest.fixture
    def prce_gui(self, qtbot, prep_signals_real):
        from pyOMA.GUI.PRCEGUI import PRCEWidget
        gui = PRCEWidget(prep_signals_real)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, prce_gui):
        expected = [
            'spin_num_corr_samples', 'spin_max_model_order',
            'btn_build_corr_tensor', 'btn_compute_modal_params', 'lbl_status',
        ]
        for name in expected:
            assert getattr(prce_gui, name).objectName() == name

    def test_fresh_instance_created_from_prep_signals(self, prce_gui, prep_signals_real):
        from pyOMA.core.PRCE import PRCE
        assert isinstance(prce_gui.instance, PRCE)
        assert prce_gui.instance.prep_signals is prep_signals_real
        assert prce_gui.instance.state == [False, False]

    def test_compute_disabled_until_built(self, prce_gui):
        assert prce_gui.btn_compute_modal_params.isEnabled() is False

    def test_build_and_compute_sequence(self, prce_gui):
        prce_gui.spin_num_corr_samples.setValue(20)
        prce_gui._on_build_corr_tensor()
        assert prce_gui.instance.state[0] is True
        assert prce_gui.btn_compute_modal_params.isEnabled() is True

        prce_gui.spin_max_model_order.setValue(6)
        prce_gui._on_compute_modal_params()
        assert prce_gui.instance.state[1] is True
        assert prce_gui.instance.modal_frequencies is not None
        assert 'computed up to order 6' in prce_gui.lbl_status.text()

    def test_compute_before_build_shows_warning_and_leaves_state_unchanged(
            self, prce_gui, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        warned = []
        monkeypatch.setattr(
            QMessageBox, 'warning',
            lambda *a, **k: warned.append(a) or None)
        # Force the button into a clickable state to exercise the guard in
        # PRCE.compute_modal_params itself (RuntimeError -> QMessageBox.warning).
        prce_gui.btn_compute_modal_params.setEnabled(True)
        prce_gui._on_compute_modal_params()
        assert warned
        assert prce_gui.instance.state[1] is False

    def test_set_instance_adopts_existing_computed_object(
            self, qtbot, prep_signals_real):
        from pyOMA.GUI.PRCEGUI import PRCEWidget
        from pyOMA.core.PRCE import PRCE
        existing = PRCE(prep_signals_real)
        existing.build_corr_tensor(20)
        existing.compute_modal_params(6)

        gui = PRCEWidget(prep_signals_real, instance=existing)
        qtbot.addWidget(gui)
        assert gui.instance is existing
        assert gui.spin_num_corr_samples.value() == 20
        assert gui.spin_max_model_order.value() == 6
        assert gui.btn_compute_modal_params.isEnabled() is True
        assert 'computed up to order 6' in gui.lbl_status.text()


# ── SSIDataWidget Designer form (pytest-qt) ───────────────────────────────────

@pytest.mark.gui
class TestSSIDataWidgetForm:
    """pytest-qt smoke test for the Designer-built ui/ssi_data.ui widget tree.

    Uses ``prep_signals_real`` (not the synthetic ``prep_signals`` fixture):
    ``SSIDataMC``/``SSIDataCV``'s ``synthesize_signals`` solves a discrete
    Riccati/Kalman-gain equation from the estimated noise covariances, which
    is singular for the noise-free synthetic ambient rod simulation used
    elsewhere in this file - a pre-existing gap (no test in the repo
    exercises ``SSIDataMC``/``SSIDataCV`` end to end) rather than something
    to work around in the widget itself.
    """

    @pytest.fixture
    def ssidata_gui(self, qtbot, prep_signals_real):
        from pyOMA.GUI.SSIDataGUI import SSIDataWidget
        gui = SSIDataWidget(prep_signals_real)
        qtbot.addWidget(gui)
        gui.show()
        yield gui

    def test_key_widgets_have_expected_object_names(self, ssidata_gui):
        expected = [
            'combo_variant', 'spin_num_block_rows', 'chk_reduced_projection',
            'spin_num_blocks', 'edit_training_blocks', 'btn_build_block_hankel',
            'spin_max_model_order', 'spin_j', 'chk_synth_sig',
            'edit_validation_blocks', 'btn_compute_modal_params',
            'spin_estimate_order', 'spin_estimate_max_modes', 'combo_estimate_algo',
            'btn_estimate_state', 'lbl_status',
        ]
        for name in expected:
            assert getattr(ssidata_gui, name).objectName() == name

    def test_default_variant_is_plain_ssidata(self, ssidata_gui):
        from pyOMA.core.SSIData import SSIData
        assert type(ssidata_gui.instance) is SSIData
        assert ssidata_gui.combo_variant.currentText() == 'SSI-Data'
        assert ssidata_gui.plain_advanced_box.isVisible() is True
        assert ssidata_gui.cv_build_box.isVisible() is False
        assert ssidata_gui.mc_compute_box.isVisible() is False
        assert ssidata_gui.cv_compute_box.isVisible() is False

    def test_compute_and_estimate_disabled_until_built(self, ssidata_gui):
        assert ssidata_gui.btn_compute_modal_params.isEnabled() is False
        assert ssidata_gui.btn_estimate_state.isEnabled() is False

    def test_build_and_compute_plain_variant(self, ssidata_gui):
        ssidata_gui.spin_num_block_rows.setValue(20)
        ssidata_gui._on_build_block_hankel()
        assert ssidata_gui.instance.state[0] is True
        assert ssidata_gui.btn_compute_modal_params.isEnabled() is True

        ssidata_gui.spin_max_model_order.setValue(6)
        ssidata_gui._on_compute_modal_params()
        assert ssidata_gui.instance.state[2] is True
        assert ssidata_gui.instance.modal_frequencies is not None

        ssidata_gui.spin_estimate_order.setValue(4)
        ssidata_gui._on_estimate_state()
        assert 'Estimated state at order 4' in ssidata_gui.lbl_status.text()

    def test_switch_to_monte_carlo_shows_mc_options_and_builds(self, ssidata_gui):
        from pyOMA.core.SSIData import SSIDataMC
        ssidata_gui.combo_variant.setCurrentText('SSI-Data (Modal Contributions)')
        assert type(ssidata_gui.instance) is SSIDataMC
        assert ssidata_gui.mc_compute_box.isVisible() is True
        assert ssidata_gui.plain_advanced_box.isVisible() is False

        ssidata_gui.spin_num_block_rows.setValue(20)
        ssidata_gui._on_build_block_hankel()
        ssidata_gui.spin_max_model_order.setValue(6)
        ssidata_gui._on_compute_modal_params()
        assert ssidata_gui.instance.state[2] is True
        assert ssidata_gui.instance.modal_contributions is not None

    def test_switch_to_cross_validation_shows_cv_options_and_builds(self, ssidata_gui):
        from pyOMA.core.SSIData import SSIDataCV
        ssidata_gui.combo_variant.setCurrentText('SSI-Data (Cross-Validation)')
        assert type(ssidata_gui.instance) is SSIDataCV
        assert ssidata_gui.cv_build_box.isVisible() is True
        assert ssidata_gui.cv_compute_box.isVisible() is True

        ssidata_gui.spin_num_block_rows.setValue(20)
        ssidata_gui.spin_num_blocks.setValue(2)
        ssidata_gui._on_build_block_hankel()
        assert ssidata_gui.instance.state[0] is True

        ssidata_gui.spin_max_model_order.setValue(6)
        ssidata_gui._on_compute_modal_params()
        assert ssidata_gui.instance.state[2] is True

    def test_switching_variant_without_progress_does_not_prompt(
            self, ssidata_gui, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, 'question',
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not prompt")))
        ssidata_gui.combo_variant.setCurrentText('SSI-Data (Modal Contributions)')
        from pyOMA.core.SSIData import SSIDataMC
        assert type(ssidata_gui.instance) is SSIDataMC

    def test_switching_variant_with_progress_prompts_and_can_be_cancelled(
            self, ssidata_gui, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        from pyOMA.core.SSIData import SSIData
        ssidata_gui.spin_num_block_rows.setValue(20)
        ssidata_gui._on_build_block_hankel()
        built_instance = ssidata_gui.instance
        assert built_instance.state[0] is True

        monkeypatch.setattr(
            QMessageBox, 'question',
            lambda *a, **k: QMessageBox.StandardButton.No)
        ssidata_gui.combo_variant.setCurrentText('SSI-Data (Modal Contributions)')
        assert ssidata_gui.instance is built_instance
        assert type(ssidata_gui.instance) is SSIData
        assert ssidata_gui.combo_variant.currentText() == 'SSI-Data'

        monkeypatch.setattr(
            QMessageBox, 'question',
            lambda *a, **k: QMessageBox.StandardButton.Yes)
        ssidata_gui.combo_variant.setCurrentText('SSI-Data (Modal Contributions)')
        assert ssidata_gui.instance is not built_instance
        from pyOMA.core.SSIData import SSIDataMC
        assert type(ssidata_gui.instance) is SSIDataMC
        assert ssidata_gui.instance.state[0] is False

    def test_set_instance_adopts_existing_computed_object(self, qtbot, prep_signals_real):
        from pyOMA.GUI.SSIDataGUI import SSIDataWidget
        from pyOMA.core.SSIData import SSIDataMC
        existing = SSIDataMC(prep_signals_real)
        existing.build_block_hankel(20)
        existing.compute_modal_params(6)

        gui = SSIDataWidget(prep_signals_real, instance=existing)
        qtbot.addWidget(gui)
        assert gui.instance is existing
        assert gui.combo_variant.currentText() == 'SSI-Data (Modal Contributions)'
        assert gui.spin_num_block_rows.value() == 20
        assert gui.spin_max_model_order.value() == 6
        assert gui.btn_compute_modal_params.isEnabled() is True
        assert 'computed up to order 6' in gui.lbl_status.text()


# ── BRSSICovRefWidget Designer form (pytest-qt) ───────────────────────────────

@pytest.mark.gui
class TestBRSSICovRefWidgetForm:
    """pytest-qt smoke test for the Designer-built ui/ssi_cov_ref.ui widget tree."""

    @pytest.fixture
    def ssicov_gui(self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.SSICovRefGUI import BRSSICovRefWidget
        gui = BRSSICovRefWidget(prep_signals_with_corr)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, ssicov_gui):
        expected = [
            'spin_num_block_columns', 'spin_num_block_rows', 'spin_shift',
            'spin_num_blocks', 'edit_training_blocks',
            'btn_build_toeplitz_cov', 'spin_max_model_order', 'spin_max_modes',
            'combo_algo', 'chk_modal_contrib', 'edit_validation_blocks',
            'btn_compute_modal_params',
            'spin_estimate_order', 'spin_estimate_max_modes', 'combo_estimate_algo',
            'btn_estimate_state', 'lbl_status',
        ]
        for name in expected:
            assert getattr(ssicov_gui, name).objectName() == name

    def test_fresh_instance_created_from_prep_signals(
            self, ssicov_gui, prep_signals_with_corr):
        from pyOMA.core.SSICovRef import BRSSICovRef
        assert isinstance(ssicov_gui.instance, BRSSICovRef)
        assert ssicov_gui.instance.prep_signals is prep_signals_with_corr
        assert ssicov_gui.instance.state == [False, False, False]

    def test_compute_and_estimate_disabled_until_built(self, ssicov_gui):
        assert ssicov_gui.btn_compute_modal_params.isEnabled() is False
        assert ssicov_gui.btn_estimate_state.isEnabled() is False

    def test_build_and_compute_sequence(self, ssicov_gui):
        ssicov_gui.spin_num_block_columns.setValue(50)
        ssicov_gui._on_build_toeplitz_cov()
        assert ssicov_gui.instance.state[0] is True
        assert ssicov_gui.btn_compute_modal_params.isEnabled() is True

        ssicov_gui.spin_max_model_order.setValue(20)
        ssicov_gui._on_compute_modal_params()
        assert ssicov_gui.instance.state[2] is True
        assert ssicov_gui.instance.modal_frequencies is not None
        assert 'computed up to order 20' in ssicov_gui.lbl_status.text()

        ssicov_gui.spin_estimate_order.setValue(4)
        ssicov_gui._on_estimate_state()
        assert 'Estimated state at order 4' in ssicov_gui.lbl_status.text()

    def test_auto_special_value_passes_none_for_block_columns(self, ssicov_gui):
        # spin_num_block_columns defaults to 50 in the .ui; explicitly set to
        # the special "Auto" value (0) to exercise the None fallback in
        # BRSSICovRef.build_toeplitz_cov (uses half of prep_signals.m_lags).
        ssicov_gui.spin_num_block_columns.setValue(0)
        ssicov_gui._on_build_toeplitz_cov()
        assert ssicov_gui.instance.state[0] is True
        assert ssicov_gui.instance.num_block_columns is not None

    def test_build_failure_shows_warning_and_leaves_state_unchanged(
            self, ssicov_gui, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        warned = []
        monkeypatch.setattr(
            QMessageBox, 'warning', lambda *a, **k: warned.append(a) or None)
        # shift so large the requested lag range exceeds available correlations,
        # forcing build_toeplitz_cov's internal correlation() call to fail.
        ssicov_gui.spin_num_block_columns.setValue(50)
        ssicov_gui.spin_shift.setValue(10_000)
        ssicov_gui._on_build_toeplitz_cov()
        assert warned
        assert ssicov_gui.instance.state[0] is False

    def test_set_instance_adopts_existing_computed_object(
            self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.SSICovRefGUI import BRSSICovRefWidget
        from pyOMA.core.SSICovRef import BRSSICovRef
        existing = BRSSICovRef(prep_signals_with_corr)
        existing.build_toeplitz_cov(50)
        existing.compute_modal_params(20)

        gui = BRSSICovRefWidget(prep_signals_with_corr, instance=existing)
        qtbot.addWidget(gui)
        assert gui.instance is existing
        assert gui.spin_num_block_columns.value() == 50
        assert gui.spin_max_model_order.value() == 20
        assert gui.btn_compute_modal_params.isEnabled() is True
        assert 'computed up to order 20' in gui.lbl_status.text()

    def test_cross_validation_disabled_by_default(self, ssicov_gui):
        # spin_num_blocks defaults to 0 ("Disabled") since CV is optional here,
        # unlike SSIDataCV where it's a dedicated variant.
        assert ssicov_gui.spin_num_blocks.value() == 0
        ssicov_gui.spin_num_block_columns.setValue(50)
        ssicov_gui._on_build_toeplitz_cov()
        assert ssicov_gui.instance.num_blocks is None

    def test_build_and_compute_with_cross_validation(self, ssicov_gui):
        ssicov_gui.spin_num_block_columns.setValue(10)
        ssicov_gui.spin_num_blocks.setValue(4)
        ssicov_gui.edit_training_blocks.setText('0,1,2')
        ssicov_gui._on_build_toeplitz_cov()
        assert ssicov_gui.instance.state[0] is True
        assert ssicov_gui.instance.num_blocks == 4
        assert list(ssicov_gui.instance.training_blocks) == [0, 1, 2]

        ssicov_gui.spin_max_model_order.setValue(10)
        ssicov_gui.edit_validation_blocks.setText('3')
        ssicov_gui._on_compute_modal_params()
        assert ssicov_gui.instance.state[2] is True
        assert ssicov_gui.instance.modal_contributions is not None

    def test_set_instance_restores_cross_validation_fields(
            self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.SSICovRefGUI import BRSSICovRefWidget
        from pyOMA.core.SSICovRef import BRSSICovRef
        existing = BRSSICovRef(prep_signals_with_corr)
        existing.build_toeplitz_cov(10, num_blocks=4, training_blocks=[0, 1, 2])

        gui = BRSSICovRefWidget(prep_signals_with_corr, instance=existing)
        qtbot.addWidget(gui)
        assert gui.spin_num_blocks.value() == 4
        assert gui.edit_training_blocks.text() == '0,1,2'


# ── VarSSIRefWidget Designer form (pytest-qt) ─────────────────────────────────

@pytest.mark.gui
class TestVarSSIRefWidgetForm:
    """pytest-qt smoke test for the Designer-built ui/var_ssi_ref.ui widget tree."""

    @pytest.fixture
    def varssi_gui(self, qtbot, prep_signals):
        from pyOMA.GUI.VarSSIRefGUI import VarSSIRefWidget
        gui = VarSSIRefWidget(prep_signals)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, varssi_gui):
        expected = [
            'spin_num_block_columns', 'spin_num_block_rows', 'spin_num_blocks',
            'combo_subspace_method', 'btn_build_subspace_mat',
            'spin_state_max_model_order', 'combo_lsq_method',
            'btn_compute_state_matrices', 'combo_variance_algo',
            'chk_sensitivities_debug', 'btn_prepare_sensitivities',
            'spin_max_model_order', 'chk_compute_debug',
            'btn_compute_modal_params', 'lbl_status',
        ]
        for name in expected:
            assert getattr(varssi_gui, name).objectName() == name

    def test_steps_disabled_until_prerequisite_done(self, varssi_gui):
        assert varssi_gui.btn_compute_state_matrices.isEnabled() is False
        assert varssi_gui.btn_prepare_sensitivities.isEnabled() is False
        assert varssi_gui.btn_compute_modal_params.isEnabled() is False

    def test_full_build_sequence(self, varssi_gui):
        varssi_gui.spin_num_block_columns.setValue(10)
        varssi_gui.spin_num_blocks.setValue(2)
        varssi_gui._on_build_subspace_mat()
        assert varssi_gui.instance.state[0] is True
        assert varssi_gui.btn_compute_state_matrices.isEnabled() is True
        assert varssi_gui.btn_prepare_sensitivities.isEnabled() is False

        varssi_gui.spin_state_max_model_order.setValue(5)
        varssi_gui._on_compute_state_matrices()
        assert varssi_gui.instance.state[1] is True
        assert varssi_gui.btn_prepare_sensitivities.isEnabled() is True
        # state[1] alone doesn't mean sensitivities are ready yet.
        assert varssi_gui.btn_compute_modal_params.isEnabled() is False

        varssi_gui._on_prepare_sensitivities()
        assert varssi_gui.btn_compute_modal_params.isEnabled() is True

        varssi_gui._on_compute_modal_params()
        assert varssi_gui.instance.state[2] is True
        assert varssi_gui.instance.modal_frequencies is not None
        assert 'computed up to order 5' in varssi_gui.lbl_status.text()

    def test_rebuilding_after_compute_disables_downstream_buttons(self, varssi_gui):
        varssi_gui.spin_num_block_columns.setValue(10)
        varssi_gui.spin_num_blocks.setValue(2)
        varssi_gui._on_build_subspace_mat()
        varssi_gui.spin_state_max_model_order.setValue(5)
        varssi_gui._on_compute_state_matrices()
        varssi_gui._on_prepare_sensitivities()
        varssi_gui._on_compute_modal_params()
        assert varssi_gui.instance.state[2] is True

        varssi_gui._on_build_subspace_mat()
        assert varssi_gui.btn_prepare_sensitivities.isEnabled() is False
        assert varssi_gui.btn_compute_modal_params.isEnabled() is False

    def test_set_instance_adopts_existing_computed_object(self, qtbot, prep_signals):
        from pyOMA.GUI.VarSSIRefGUI import VarSSIRefWidget
        from pyOMA.core.VarSSIRef import VarSSIRef
        existing = VarSSIRef(prep_signals)
        existing.build_subspace_mat(num_block_columns=10, num_blocks=2)
        existing.compute_state_matrices(max_model_order=5)
        existing.prepare_sensitivities()
        existing.compute_modal_params()

        gui = VarSSIRefWidget(prep_signals, instance=existing)
        qtbot.addWidget(gui)
        assert gui.instance is existing
        assert gui.spin_num_block_columns.value() == 10
        assert gui.btn_compute_modal_params.isEnabled() is True
        assert 'computed up to order 5' in gui.lbl_status.text()


# ── PLSCFWidget Designer form (pytest-qt) ─────────────────────────────────────

@pytest.mark.gui
class TestPLSCFWidgetForm:
    """pytest-qt smoke test for the Designer-built ui/plscf.ui widget tree."""

    @pytest.fixture
    def plscf_gui(self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.PLSCFGUI import PLSCFWidget
        gui = PLSCFWidget(prep_signals_with_corr)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, plscf_gui):
        expected = [
            'spin_nperseg', 'spin_begin_frequency', 'spin_end_frequency',
            'spin_window_decay', 'spin_num_blocks', 'edit_training_blocks',
            'btn_build_half_spectra', 'lbl_num_omega',
            'spin_max_model_order', 'chk_complex_coefficients', 'combo_algo',
            'combo_modal_contrib', 'edit_validation_blocks',
            'btn_compute_modal_params', 'lbl_status',
        ]
        for name in expected:
            assert getattr(plscf_gui, name).objectName() == name

    def test_compute_disabled_until_built(self, plscf_gui):
        assert plscf_gui.btn_compute_modal_params.isEnabled() is False
        assert plscf_gui.lbl_num_omega.text() == '-'

    def test_build_and_compute_sequence(self, plscf_gui):
        plscf_gui.spin_nperseg.setValue(200)
        plscf_gui._on_build_half_spectra()
        assert plscf_gui.instance.state[0] is True
        assert plscf_gui.btn_compute_modal_params.isEnabled() is True
        assert plscf_gui.lbl_num_omega.text() == str(plscf_gui.instance.num_omega)

        plscf_gui.spin_max_model_order.setValue(5)
        plscf_gui._on_compute_modal_params()
        assert plscf_gui.instance.state[1] is True
        assert plscf_gui.instance.modal_frequencies is not None
        assert 'computed up to order 5' in plscf_gui.lbl_status.text()

    def test_auto_special_value_passes_none_for_frequency_bounds(self, plscf_gui):
        # Both spin_begin_frequency/spin_end_frequency default to the "Auto"
        # special value (-1) in the .ui - exercise the None fallback path in
        # PLSCF.build_half_spectra (0 .. Nyquist).
        assert plscf_gui.spin_begin_frequency.value() < 0
        assert plscf_gui.spin_end_frequency.value() < 0
        plscf_gui.spin_nperseg.setValue(200)
        plscf_gui._on_build_half_spectra()
        assert plscf_gui.instance.state[0] is True
        assert plscf_gui.instance.begin_frequency == 0.0

    def test_set_instance_adopts_existing_computed_object(
            self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.PLSCFGUI import PLSCFWidget
        from pyOMA.core.PLSCF import PLSCF
        existing = PLSCF(prep_signals_with_corr)
        existing.build_half_spectra(nperseg=200)
        existing.compute_modal_params(5)

        gui = PLSCFWidget(prep_signals_with_corr, instance=existing)
        qtbot.addWidget(gui)
        assert gui.instance is existing
        assert gui.spin_nperseg.value() == 200
        assert gui.spin_max_model_order.value() == 5
        assert gui.btn_compute_modal_params.isEnabled() is True
        assert 'computed up to order 5' in gui.lbl_status.text()

    def test_cross_validation_disabled_by_default(self, plscf_gui):
        assert plscf_gui.spin_num_blocks.value() == 0
        plscf_gui.spin_nperseg.setValue(80)
        plscf_gui._on_build_half_spectra()
        assert plscf_gui.instance.num_blocks is None

    def test_build_and_compute_with_cross_validation(self, plscf_gui):
        plscf_gui.spin_nperseg.setValue(80)
        plscf_gui.spin_num_blocks.setValue(4)
        plscf_gui.edit_training_blocks.setText('0,1,2')
        plscf_gui._on_build_half_spectra()
        assert plscf_gui.instance.state[0] is True
        assert plscf_gui.instance.num_blocks == 4
        assert list(plscf_gui.instance.training_blocks) == [0, 1, 2]

        plscf_gui.spin_max_model_order.setValue(5)
        plscf_gui.edit_validation_blocks.setText('3')
        plscf_gui._on_compute_modal_params()
        assert plscf_gui.instance.state[1] is True
        assert plscf_gui.instance.modal_contributions is not None

    def test_set_instance_restores_cross_validation_fields(
            self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.PLSCFGUI import PLSCFWidget
        from pyOMA.core.PLSCF import PLSCF
        existing = PLSCF(prep_signals_with_corr)
        existing.build_half_spectra(nperseg=80, num_blocks=4, training_blocks=[0, 1, 2])

        gui = PLSCFWidget(prep_signals_with_corr, instance=existing)
        qtbot.addWidget(gui)
        assert gui.spin_num_blocks.value() == 4
        assert gui.edit_training_blocks.text() == '0,1,2'


# ── ModalAnalysisGUI Designer form (pytest-qt) ────────────────────────────────

@pytest.mark.gui
class TestModalAnalysisGUIForm:
    """pytest-qt smoke test for the Designer-built ui/modal_analysis.ui widget tree."""

    @pytest.fixture
    def modal_gui(self, qtbot, prep_signals_real):
        # prep_signals_real (2 ref channels), not prep_signals_with_corr (1 ref
        # channel): the PRCE page requires num_ref_channels >= 2, and every
        # page is constructed eagerly against the same prep_signals here.
        from pyOMA.GUI.ModalAnalysisGUI import ModalAnalysisGUI
        gui = ModalAnalysisGUI(prep_signals_real)
        qtbot.addWidget(gui)
        yield gui

    def test_key_widgets_have_expected_object_names(self, modal_gui):
        expected = ['combo_method', 'stacked_widget', 'btn_save', 'btn_load', 'lbl_setup_name']
        for name in expected:
            assert getattr(modal_gui, name).objectName() == name

    def test_combo_has_one_entry_per_method_widget(self, modal_gui):
        assert modal_gui.combo_method.count() == 5
        labels = [modal_gui.combo_method.itemText(i) for i in range(5)]
        assert labels == ['SSI-Data', 'SSI-Cov-Ref', 'Var-SSI-Ref', 'pLSCF', 'PRCE']
        assert modal_gui.stacked_widget.count() == 5

    def test_default_page_is_ssidata_and_setup_name_shown(
            self, modal_gui, prep_signals_real):
        from pyOMA.GUI.SSIDataGUI import SSIDataWidget
        assert modal_gui.combo_method.currentIndex() == 0
        assert isinstance(modal_gui.stacked_widget.currentWidget(), SSIDataWidget)
        assert modal_gui.lbl_setup_name.text() == (prep_signals_real.setup_name or '-')

    def test_switching_combo_switches_stacked_page(self, modal_gui):
        from pyOMA.GUI.PLSCFGUI import PLSCFWidget
        modal_gui.combo_method.setCurrentIndex(3)
        assert modal_gui.stacked_widget.currentWidget() is modal_gui._pages[3]
        assert isinstance(modal_gui.stacked_widget.currentWidget(), PLSCFWidget)

    def test_modal_data_reflects_active_page_instance(self, modal_gui):
        modal_gui.combo_method.setCurrentIndex(4)  # PRCE
        assert modal_gui.modal_data is modal_gui._pages[4].instance

    def test_pages_are_independent_switching_does_not_reset_progress(self, modal_gui):
        prce_page = modal_gui._pages[4]
        prce_page.spin_num_corr_samples.setValue(20)
        prce_page._on_build_corr_tensor()
        assert prce_page.instance.state[0] is True

        modal_gui.combo_method.setCurrentIndex(0)
        modal_gui.combo_method.setCurrentIndex(4)
        assert prce_page.instance.state[0] is True

    def test_construction_with_existing_modal_data_selects_matching_page(
            self, qtbot, prep_signals_with_corr):
        from pyOMA.GUI.ModalAnalysisGUI import ModalAnalysisGUI
        from pyOMA.core.PLSCF import PLSCF
        existing = PLSCF(prep_signals_with_corr)
        existing.build_half_spectra(nperseg=200)
        existing.compute_modal_params(5)

        gui = ModalAnalysisGUI(prep_signals_with_corr, modal_data=existing)
        qtbot.addWidget(gui)
        assert gui.combo_method.currentIndex() == 3
        assert gui.modal_data is existing
        assert gui.stacked_widget.currentWidget().instance is existing

    def test_save_and_load_round_trip(self, modal_gui, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        modal_gui.combo_method.setCurrentIndex(4)  # PRCE, cheapest to compute
        page = modal_gui._pages[4]
        page.spin_num_corr_samples.setValue(20)
        page._on_build_corr_tensor()
        page.spin_max_model_order.setValue(6)
        page._on_compute_modal_params()

        original_instance = page.instance
        fname = tmp_path / 'prce_state.npz'
        monkeypatch.setattr(
            QFileDialog, 'getSaveFileName', lambda *a, **k: (str(fname), ''))
        modal_gui._on_save()
        assert fname.exists()

        monkeypatch.setattr(
            QFileDialog, 'getOpenFileName', lambda *a, **k: (str(fname), ''))
        modal_gui._on_load()
        # page.instance is reassigned in place by set_instance(), so
        # modal_data now *is* page.instance again - just a freshly-loaded
        # object, not the one that existed before _on_load() ran.
        assert modal_gui.modal_data is page.instance
        assert modal_gui.modal_data is not original_instance
        assert modal_gui.modal_data.state[1]  # numpy.bool_ after load_state(), not `is True`
        assert modal_gui.modal_data.modal_frequencies is not None

    def test_close_snapshots_modal_data_before_deletion(self, qtbot, prep_signals_real):
        # Deliberately destroys the widget (like
        # test_close_deletes_window_so_blocking_event_loops_can_exit in
        # TestPreProcessSignalsGUIForm) - uses its own instance rather than
        # the shared qtbot.addWidget-registered `modal_gui` fixture, since
        # pytest-qt's automatic teardown would otherwise try to close an
        # already-deleted widget.
        from pyOMA.GUI.ModalAnalysisGUI import ModalAnalysisGUI
        gui = ModalAnalysisGUI(prep_signals_real)
        gui.combo_method.setCurrentIndex(4)
        expected = gui._pages[4].instance
        with qtbot.waitSignal(gui.destroyed, timeout=1000):
            gui.close()
        assert gui._modal_data_at_close is expected

    def test_cancel_button_closes_without_validation(self, qtbot, prep_signals_real):
        """btn_save ("Cancel") just closes, even with nothing computed."""
        from pyOMA.GUI.ModalAnalysisGUI import ModalAnalysisGUI
        gui = ModalAnalysisGUI(prep_signals_real)
        assert gui.modal_data.modal_frequencies is None
        with qtbot.waitSignal(gui.destroyed, timeout=1000):
            gui.btn_save.click()

    def test_ok_close_button_warns_when_nothing_computed(self, modal_gui, monkeypatch):
        """btn_load ("OK and Close") refuses to close an unrun page."""
        warnings = []
        from PyQt6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))
        assert modal_gui.modal_data.modal_frequencies is None
        modal_gui.btn_load.click()
        assert len(warnings) == 1
        assert modal_gui.isVisible()

    def test_ok_close_button_closes_when_computed(self, qtbot, prep_signals_real):
        from pyOMA.GUI.ModalAnalysisGUI import ModalAnalysisGUI
        gui = ModalAnalysisGUI(prep_signals_real)
        gui.combo_method.setCurrentIndex(4)  # PRCE, cheapest to compute
        page = gui._pages[4]
        page.spin_num_corr_samples.setValue(20)
        page._on_build_corr_tensor()
        page.spin_max_model_order.setValue(6)
        page._on_compute_modal_params()
        assert gui.modal_data.modal_frequencies is not None

        with qtbot.waitSignal(gui.destroyed, timeout=1000):
            gui.btn_load.click()

    def test_config_menu_actions_wired(self, modal_gui):
        assert modal_gui.actionLoad_Config.receivers(modal_gui.actionLoad_Config.triggered) > 0
        assert modal_gui.actionSave_Config.receivers(modal_gui.actionSave_Config.triggered) > 0

    def test_save_config_then_load_config_round_trip(self, modal_gui, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog
        modal_gui.combo_method.setCurrentIndex(4)  # PRCE
        page = modal_gui._pages[4]
        page.spin_num_corr_samples.setValue(20)
        page._on_build_corr_tensor()
        page.spin_max_model_order.setValue(6)
        page._on_compute_modal_params()

        fname = tmp_path / 'prce_config.txt'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(fname), ''))
        modal_gui._on_save_config()
        assert fname.exists()

        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(fname), ''))
        modal_gui._on_load_config()
        assert modal_gui.modal_data.modal_frequencies.shape[0] == 6
