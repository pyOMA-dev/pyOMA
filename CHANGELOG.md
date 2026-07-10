# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - Unreleased

First tagged release. Prior development happened without version tags;
this entry summarizes the state of the project at 1.0.0.

Published on PyPI as `pyoma-toolbox` (the import name remains `pyOMA`,
e.g. `pip install pyoma-toolbox` then `import pyOMA`); the originally
intended name `py-OMA` was already taken by an unrelated project.

### Added

- Full PyQt6 desktop GUI, built from Qt Designer `.ui` files with generated
  Python sources kept in sync via a pre-commit hook and a CI check
  (`scripts/build_ui.py --check`):
  - `PreProcessSignalsGUI` — signal loading, filtering, decimation, channel
    editing, and diagnostic plotting. Can now be started empty
    (`prep_signals=None`) and populated afterwards via `import_signals()`
    or `load_state()`.
  - `GeometryProcessorGUI` — interactive node/line/parent-child geometry
    editor with a live 3-D preview.
  - `ChanDofEditorGUI` — per-channel DOF assignment editor.
  - `ModalAnalysisGUI` — hosts one widget per single-setup OMA method
    (SSI-Data, SSI-Cov-Ref, Var-SSI-Ref, pLSCF, PRCE) behind a shared
    method-selector.
  - `StabilGUI` — interactive stabilization-diagram and pole-selection
    window, with `ComplexPlot`/`HistoPlot` companion views.
  - `PlotMSHGUI` (`ModeShapeGUI`) — interactive 3-D mode-shape animation.
  - `MultiSetupGUI` — orchestrates multi-setup PoSER/PoGER analysis across
    per-setup tabs, plus a "Single Setup" mode for single-setup analysis
    through the same interactive flow (skips the merge step).
  - `pyoma` console-script entry point (installed via
    `pip install "pyOMA[gui]"`) launches `MultiSetupGUI` directly.
- Save-before-close prompts (`UnsavedChangesMixin`): closing a window with
  unsaved edits now offers Save/Discard/Cancel instead of discarding
  silently. Wired into `PreProcessSignalsGUI`, `GeometryProcessorGUI`,
  `StabilGUI`, `ModalAnalysisGUI`, and `MultiSetupGUI`.
- Cross-validation support in `BRSSICovRef` and `PLSCF`, exposed in the GUI.
- `PreProcessSignals.signal_clarity_score()`, surfaced in
  `PreProcessSignalsGUI`.
- `PreProcessSignals.save_config()`/`save_chan_dofs()` — write-side
  counterparts of `init_from_config()`/`load_chan_dofs()`.
- Signal-import GUI action and example measurement-format converter scripts.
- `tests.yml` CI workflow: runs the full test suite (all markers, not just
  `-m gui`) on every push/PR across Python 3.9-3.12.
- `publish.yml` CI workflow: builds and publishes to PyPI via trusted
  publishing (OIDC) on a `v*` tag push.

### Fixed

- `PreProcessSignals._apply_delete_channels()`: deletion is now purely
  index-based. Previously it required a `chan_dofs` entry for every
  surviving channel, crashing with `chan_dofs=None` and silently dropping
  channels with a partial (but legitimate) chan-DOF assignment.
- `PlotMSH.refresh_lines()`: used to index `phi_nodes`/`disp_nodes`
  directly instead of `.get(key, [0, 0, 0])` like the rest of the class,
  raising `KeyError` when a node was added to a long-lived
  `ModeShapePlot` (e.g. via `GeometryProcessorGUI`) after construction.

### Changed

- `Development Status` classifier bumped from Alpha to
  `5 - Production/Stable`.
