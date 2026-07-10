"""Qt GUI tests: MultiSetupGUI (PoSER/PoGER orchestrator).

All tests in this file require PyQt6 and are marked ``gui``. They run
headless via ``QT_QPA_PLATFORM=offscreen`` (set in conftest.py). Real modal
identification / pole-selection / stabilization-diagram interaction is
covered elsewhere (tests/test_multi_setup.py, tests/test_gui.py); here we
only check wiring, status/gating logic, and merge call sequencing (with the
numerical steps and sub-dialogs monkeypatched out).

To run only these tests::

    pytest -m gui
"""
import types

import pytest

pytest.importorskip('PyQt6', reason='PyQt6 not installed – pip install "pyOMA[gui]"')

from PyQt6.QtWidgets import QMessageBox, QFileDialog

from pyOMA.GUI import MultiSetupGUI as msg_module
from pyOMA.GUI.MultiSetupGUI import SetupTabWidget, MultiSetupGUI


# ── QApplication singleton ────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def qapp():
    """Headless Qt application, shared across all GUI tests."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_blocking_close_prompt(monkeypatch):
    """Default UnsavedChangesMixin's close-time save prompt to Continue
    (close without saving).

    qtbot.addWidget() closes widgets during teardown; without this, any
    test that leaves a GUI's _dirty flag True would block on a real modal
    QMessageBox in headless CI. UnsavedChangesMixin builds its own
    QMessageBox instance and calls .exec() on it (needed to relabel the
    "Discard" button to "Continue"), so QMessageBox.exec is mocked here,
    not .question() - the "Remove Setup" confirmation still goes through
    the real .question() and is mocked separately where exercised. Tests
    exercising the save prompt itself override QMessageBox.exec again
    within the test body."""
    monkeypatch.setattr(
        QMessageBox, 'exec',
        lambda self: QMessageBox.StandardButton.Discard)


def _fake_stabil_calc(select_modes):
    """Minimal stand-in for a StabilCalc, exposing only what the gating
    logic (``SetupTabWidget.is_ready_poser``) and status text read."""
    return types.SimpleNamespace(select_modes=select_modes)


def _real_prep_signals(test_files_dir):
    """A real PreProcessSignals, standing in for what start_preprocess_gui
    would return after a user loads a file inside PreProcessSignalsGUI -
    loading itself now happens there, not in SetupTabWidget."""
    from pyOMA.core.PreProcessingTools import PreProcessSignals
    meas_dir = test_files_dir / 'measurement_1'
    return PreProcessSignals.init_from_config(
        conf_file=str(meas_dir / 'setup_info.txt'),
        meas_file=str(meas_dir / 'measurement_1.npy'),
        chan_dofs_file=str(meas_dir / 'channel_dofs.txt'))


# ── SetupTabWidget ─────────────────────────────────────────────────────────────

@pytest.mark.gui
class TestSetupTabWidget:

    @pytest.fixture
    def tab(self, qapp, qtbot):
        widget = SetupTabWidget('PoSER')
        qtbot.addWidget(widget)
        return widget

    def test_construction_defaults(self, tab):
        assert tab.prep_signals is None
        assert tab.modal_data is None
        assert tab.stabil_calc is None
        assert tab.lbl_status.text() == "Not loaded."

    def test_poser_mode_shows_modal_id_and_poles_rows(self, tab):
        assert not tab.btn_modal_id.isHidden()
        assert not tab.btn_select_poles.isHidden()

    def test_poger_mode_hides_modal_id_and_poles_rows(self, tab):
        tab.set_mode('PoGER')
        assert tab.btn_modal_id.isHidden()
        assert tab.btn_select_poles.isHidden()

    def test_preprocess_button_starts_enabled(self, tab):
        assert tab.btn_preprocess.isEnabled()

    def test_preprocess_with_real_files_sets_prep_signals(
            self, tab, test_files_dir, monkeypatch):
        prep_signals = _real_prep_signals(test_files_dir)
        monkeypatch.setattr(msg_module, 'start_preprocess_gui', lambda *a, **k: prep_signals)

        tab._on_preprocess()

        assert tab.prep_signals is prep_signals
        assert tab.is_ready_poger
        assert "loaded" in tab.lbl_status.text()

    def test_preprocess_with_nothing_loaded_leaves_prep_signals_none(self, tab, monkeypatch):
        monkeypatch.setattr(msg_module, 'start_preprocess_gui', lambda *a, **k: None)
        tab._on_preprocess()
        assert tab.prep_signals is None

    def test_reopening_preprocess_with_same_object_keeps_downstream_state(
            self, tab, test_files_dir, monkeypatch):
        """Re-opening the pre-process window on the *same* prep_signals
        (further edits, not a fresh load) must not reset modal_data/
        stabil_calc - only a genuine replacement should."""
        prep_signals = _real_prep_signals(test_files_dir)
        monkeypatch.setattr(msg_module, 'start_preprocess_gui', lambda *a, **k: prep_signals)
        tab._on_preprocess()
        tab.modal_data = object()
        tab.stabil_calc = _fake_stabil_calc([0])

        tab._on_preprocess()

        assert tab.modal_data is not None
        assert tab.stabil_calc is not None

    def test_is_ready_poser_requires_selected_modes(self, tab):
        assert not tab.is_ready_poser
        tab.stabil_calc = _fake_stabil_calc([])
        assert not tab.is_ready_poser
        tab.stabil_calc = _fake_stabil_calc([0, 1])
        assert tab.is_ready_poser

    def test_status_reflects_lifecycle_stage(self, tab, test_files_dir, monkeypatch):
        prep_signals = _real_prep_signals(test_files_dir)
        monkeypatch.setattr(msg_module, 'start_preprocess_gui', lambda *a, **k: prep_signals)
        tab._on_preprocess()
        assert "not identified yet" in tab.lbl_status.text()

        tab.modal_data = object()
        tab._refresh_status()
        assert "poles not selected yet" in tab.lbl_status.text()

        tab.stabil_calc = _fake_stabil_calc([0, 1, 2])
        tab._refresh_status()
        assert "3 mode(s) selected" in tab.lbl_status.text()


# ── MultiSetupGUI ────────────────────────────────────────────────────────────

@pytest.mark.gui
class TestMultiSetupGUI:

    @pytest.fixture
    def gui(self, qapp, qtbot):
        form = MultiSetupGUI()
        qtbot.addWidget(form)
        return form

    def test_construction_without_geometry(self, gui):
        assert gui.geometry_data is None
        assert gui.mode == 'Single Setup'
        assert gui.lbl_geometry_status.text() == "Geometry: not loaded"

    def test_construction_with_geometry(self, qapp, qtbot, geometry_data):
        form = MultiSetupGUI(geometry_data)
        qtbot.addWidget(form)
        assert f"{len(geometry_data.nodes)} node(s)" in form.lbl_geometry_status.text()

    def test_add_and_remove_setup_changes_tab_count(self, gui, monkeypatch):
        assert gui.tabs_setups.count() == 0
        gui._on_add_setup()
        gui._on_add_setup()
        assert gui.tabs_setups.count() == 2

        monkeypatch.setattr(
            QMessageBox, 'question',
            lambda *a, **k: QMessageBox.StandardButton.Yes)
        gui._on_remove_setup()
        assert gui.tabs_setups.count() == 1

    def test_mode_switch_toggles_settings_page_and_tab_visibility(self, gui):
        gui._on_add_setup()
        tab = gui._tabs[0]
        assert gui.stack_settings.currentIndex() == 0
        assert not tab.btn_select_poles.isHidden()

        gui.combo_mode.setCurrentText('PoGER')

        assert gui.stack_settings.currentIndex() == 1
        assert tab.btn_select_poles.isHidden()

    def test_merge_button_disabled_with_fewer_than_two_setups(self, gui):
        assert not gui.btn_merge.isEnabled()
        gui._on_add_setup()
        assert not gui.btn_merge.isEnabled()

    def test_merge_button_enabled_only_once_all_poser_setups_ready(self, gui):
        gui.combo_mode.setCurrentText('PoSER')
        gui._on_add_setup()
        gui._on_add_setup()
        assert not gui.btn_merge.isEnabled()

        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._tabs[0]._refresh_status()
        gui._refresh_merge_status()
        assert not gui.btn_merge.isEnabled(), "only one of two setups ready"

        gui._tabs[1].stabil_calc = _fake_stabil_calc([0])
        gui._refresh_merge_status()
        assert gui.btn_merge.isEnabled()

    def test_selecting_poles_auto_enables_merge_button_without_mode_switch(self, gui):
        """Regression: btn_merge used to stay stale after selecting poles on
        the last setup - only switching Merge method away and back (which
        happens to call _refresh_merge_status()) would pick it up. Fixed by
        having SetupTabWidget emit status_changed from _refresh_status()
        (exactly what _on_select_poles() calls internally), which
        MultiSetupGUI connects to _refresh_merge_status() - so this test
        deliberately never calls gui._refresh_merge_status() itself."""
        gui.combo_mode.setCurrentText('PoSER')
        gui._on_add_setup()
        gui._on_add_setup()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._tabs[0]._refresh_status()
        assert not gui.btn_merge.isEnabled(), "only one of two setups ready"

        gui._tabs[1].stabil_calc = _fake_stabil_calc([0])
        gui._tabs[1]._refresh_status()
        assert gui.btn_merge.isEnabled()

    def test_merge_button_gating_in_poger_mode_only_needs_loaded_setups(
            self, gui, test_files_dir, monkeypatch):
        gui.combo_mode.setCurrentText('PoGER')
        gui._on_add_setup()
        gui._on_add_setup()
        assert not gui.btn_merge.isEnabled()

        prep_signals = _real_prep_signals(test_files_dir)
        monkeypatch.setattr(msg_module, 'start_preprocess_gui', lambda *a, **k: prep_signals)
        for tab in gui._tabs:
            tab._on_preprocess()
        gui._refresh_merge_status()
        assert gui.btn_merge.isEnabled()

    def test_merge_poser_calls_add_setup_per_tab_then_merge(self, gui, monkeypatch):
        calls = []
        monkeypatch.setattr(
            msg_module.MergePoSER, 'add_setup',
            lambda self, prep, modal, stabil: calls.append(('add_setup', prep, modal, stabil)))
        monkeypatch.setattr(
            msg_module.MergePoSER, 'merge', lambda self: calls.append(('merge',)))

        gui._on_add_setup()
        gui._on_add_setup()
        for i, tab in enumerate(gui._tabs):
            tab.prep_signals = f'prep{i}'
            tab.modal_data = f'modal{i}'
            tab.stabil_calc = f'stabil{i}'

        gui._merge_poser()

        assert calls == [
            ('add_setup', 'prep0', 'modal0', 'stabil0'),
            ('add_setup', 'prep1', 'modal1', 'stabil1'),
            ('merge',),
        ]
        assert isinstance(gui.merged_data, msg_module.MergePoSER)

    def test_merge_poger_calls_pipeline_in_order(self, gui, monkeypatch):
        calls = []
        monkeypatch.setattr(
            msg_module.PogerSSICovRef, 'add_setup',
            lambda self, prep: calls.append(('add_setup', prep)))
        monkeypatch.setattr(
            msg_module.PogerSSICovRef, 'pair_channels',
            lambda self: calls.append(('pair_channels',)))
        monkeypatch.setattr(
            msg_module.PogerSSICovRef, 'build_merged_subspace_matrix',
            lambda self, num_block_columns, **k: calls.append(
                ('build_merged_subspace_matrix', num_block_columns)))
        monkeypatch.setattr(
            msg_module.PogerSSICovRef, 'compute_modal_params',
            lambda self, max_model_order, **k: calls.append(
                ('compute_modal_params', max_model_order)))
        monkeypatch.setattr(
            msg_module, 'start_stabil_gui', lambda *a, **k: calls.append(('start_stabil_gui',)))

        class _FakeStabilCluster:
            def __init__(self, modal_data, prep_signals):
                pass

            def calculate_stabilization_masks(self, **k):
                pass

        monkeypatch.setattr(msg_module, 'StabilCluster', _FakeStabilCluster)
        monkeypatch.setattr(msg_module, 'StabilPlot', lambda stabil_calc: stabil_calc)

        gui.combo_mode.setCurrentText('PoGER')
        gui._on_add_setup()
        gui._on_add_setup()
        for i, tab in enumerate(gui._tabs):
            # m_lags must be set (not None) - _merge_poger() checks it before
            # add_setup() to fail with a clear message instead of a raw
            # TypeError deep inside build_merged_subspace_matrix.
            tab.prep_signals = types.SimpleNamespace(m_lags=100, setup_name=f'prep{i}')
        gui.spin_num_block_columns.setValue(80)
        gui.spin_max_model_order.setValue(40)

        gui._merge_poger()

        names = [c[1].setup_name if c[0] == 'add_setup' else c for c in calls]
        assert names == [
            'prep0',
            'prep1',
            ('pair_channels',),
            ('build_merged_subspace_matrix', 80),
            ('compute_modal_params', 40),
            ('start_stabil_gui',),
        ]
        assert isinstance(gui.merged_data, msg_module.PogerSSICovRef)

    def test_merge_poger_without_correlation_warns_instead_of_crashing(
            self, gui, test_files_dir, monkeypatch):
        """Regression: PogerSSICovRef.add_setup() silently leaves
        self.m_lags at None if a setup's prep_signals never had
        correlation computed, and build_merged_subspace_matrix() then
        crashes deep inside with a raw
        `TypeError: '<=' not supported between instances of 'int' and
        'NoneType'`. _merge_poger() must catch this before add_setup() and
        surface a clear, actionable message via the existing
        _on_merge()-level warning dialog instead."""
        gui.combo_mode.setCurrentText('PoGER')
        gui._on_add_setup()
        gui._on_add_setup()
        # _real_prep_signals() only calls init_from_config() - no
        # .correlation()/.corr_blackman_tukey() call, so m_lags is None.
        prep_signals = _real_prep_signals(test_files_dir)
        monkeypatch.setattr(msg_module, 'start_preprocess_gui', lambda *a, **k: prep_signals)
        for tab in gui._tabs:
            tab._on_preprocess()
        assert prep_signals.m_lags is None

        warnings = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a[-1]))

        gui._on_merge()

        assert len(warnings) == 1
        assert "correlation" in warnings[0].lower()
        assert gui.merged_data is None

    def test_close_event_snapshots_merged_data_before_deletion(self, qapp, qtbot):
        form = MultiSetupGUI()
        qtbot.addWidget(form)
        sentinel = object()
        form.merged_data = sentinel

        form.close()

        assert form._merged_data_at_close is sentinel


# ── MultiSetupGUI: Single Setup mode ──────────────────────────────────────────

@pytest.mark.gui
class TestMultiSetupGUISingleSetupMode:
    """Single Setup mode reuses SetupTabWidget's PoSER-style per-tab UI
    (load -> preprocess -> modal ID -> pole selection) but skips the merge
    step entirely - there is nothing to merge for one setup."""

    @pytest.fixture
    def gui(self, qapp, qtbot):
        form = MultiSetupGUI()
        qtbot.addWidget(form)
        return form

    def test_default_mode_is_single_setup(self, gui):
        assert gui.mode == 'Single Setup'

    def test_add_setup_disables_add_button_after_one_tab(self, gui):
        assert gui.btn_add_setup.isEnabled()
        gui._on_add_setup()
        assert gui.tabs_setups.count() == 1
        assert not gui.btn_add_setup.isEnabled()

    def test_switching_mode_re_enables_add_button(self, gui):
        gui._on_add_setup()
        assert not gui.btn_add_setup.isEnabled()
        gui.combo_mode.setCurrentText('PoSER')
        assert gui.btn_add_setup.isEnabled()

    def test_merge_button_labelled_continue(self, gui):
        assert gui.btn_merge.text() == 'Continue'
        gui.combo_mode.setCurrentText('PoSER')
        assert gui.btn_merge.text() == 'Merge Setups'

    def test_merge_status_gates_on_single_tab_pole_selection(self, gui):
        gui._on_add_setup()
        assert not gui.btn_merge.isEnabled()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0, 1])
        gui._refresh_merge_status()
        assert gui.btn_merge.isEnabled()

    def test_on_merge_does_not_raise_and_needs_no_second_tab(self, gui):
        gui._on_add_setup()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._refresh_merge_status()

        gui._on_merge()

        assert gui.merged_data is None
        assert gui._single_setup_tab is gui._tabs[0]
        assert gui.btn_show_mode_shapes.isEnabled()

    def test_on_show_mode_shapes_uses_single_tab_data(self, gui, geometry_data, monkeypatch):
        calls = []
        monkeypatch.setattr(
            msg_module, 'ModeShapePlot',
            lambda **kwargs: calls.append(('build', kwargs)) or 'plot')
        monkeypatch.setattr(
            msg_module, 'start_msh_gui', lambda plot: calls.append(('start', plot)))

        gui.geometry_data = geometry_data
        gui._on_add_setup()
        tab = gui._tabs[0]
        tab.prep_signals = 'prep0'
        tab.modal_data = 'modal0'
        tab.stabil_calc = 'stabil0'
        gui._on_merge()

        gui._on_show_mode_shapes()

        assert calls[0] == ('build', {
            'geometry_data': geometry_data,
            'stabil_calc': 'stabil0',
            'modal_data': 'modal0',
            'prep_signals': 'prep0',
        })
        assert calls[1] == ('start', 'plot')


# ── MultiSetupGUI: unsaved changes ────────────────────────────────────────────

@pytest.mark.gui
class TestMultiSetupGUIUnsavedChanges:
    """UnsavedChangesMixin wiring: a successful merge sets _dirty;
    closeEvent() prompts to save when it's set."""

    @pytest.fixture
    def gui(self, qapp, qtbot):
        form = MultiSetupGUI()
        qtbot.addWidget(form)
        return form

    def test_starts_clean(self, gui):
        assert gui._dirty is False

    def test_single_setup_merge_sets_dirty(self, gui):
        gui._on_add_setup()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._on_merge()
        assert gui._dirty is True

    def test_switching_mode_clears_dirty(self, gui):
        gui._on_add_setup()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._on_merge()
        gui.combo_mode.setCurrentText('PoSER')
        assert gui._dirty is False

    def test_close_with_dirty_cancel_keeps_window_open(self, qapp, qtbot, monkeypatch):
        gui = MultiSetupGUI()
        gui._on_add_setup()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._on_merge()
        monkeypatch.setattr(
            QMessageBox, 'exec', lambda self: QMessageBox.StandardButton.Cancel)
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        gui.closeEvent(event)
        assert event.isAccepted() is False

    def test_close_with_dirty_save_calls_do_save(self, qapp, qtbot, monkeypatch, tmp_path):
        gui = MultiSetupGUI()
        gui.combo_mode.setCurrentText('PoSER')
        gui._on_add_setup()
        gui._on_add_setup()
        gui._tabs[0].stabil_calc = _fake_stabil_calc([0])
        gui._tabs[1].stabil_calc = _fake_stabil_calc([0])
        for i, tab in enumerate(gui._tabs):
            tab.prep_signals = f'prep{i}'
            tab.modal_data = f'modal{i}'
        monkeypatch.setattr(
            msg_module.MergePoSER, 'add_setup', lambda self, prep, modal, stabil: None)
        monkeypatch.setattr(msg_module.MergePoSER, 'merge', lambda self: None)
        monkeypatch.setattr(
            msg_module.MergePoSER, 'save_state', lambda self, fname: tmp_path.joinpath(
                'saved.marker').write_text(fname))
        gui._on_merge()
        assert gui._dirty is True

        fname = tmp_path / 'merged.npz'
        monkeypatch.setattr(
            QMessageBox, 'exec', lambda self: QMessageBox.StandardButton.Save)
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(fname), ''))
        gui.close()
        assert (tmp_path / 'saved.marker').exists()
        assert gui._dirty is False
