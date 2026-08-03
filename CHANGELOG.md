# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2026-08-03

### Added

- **3-D mode-shape visualization with a pyvista/VTK backend**, selectable
  alongside the existing matplotlib backend:
  - `ModeShapePlotPVQt` — interactive desktop viewer embedded through a
    `QtInteractor` in `PlotMSHGUI`.
  - `ModeShapePlotPVJupyter` — notebook viewer rendering client-side via
    trame/vtk.js, with full control-panel parity (mode/amplitude, visibility
    toggles, view presets, animation).
  - Global backend switch: `resolve_mode_shape_backend()` picks pyvista when
    the `pyOMA[pyvista]` extra is installed, else matplotlib; override with the
    `PYOMA_MSH_BACKEND` environment variable or `pyOMA.core.MSH_BACKEND`.
  - A backend-neutral GUI contract, so `PlotMSHGUI`, `StabilGUI`, and the
    geometry / chan-DOF editors work unchanged against either backend.
  - Animation frame export (PNG/PDF) from both backends and both frontends.
- **Surface (face) geometry**: `GeometryProcessor` gains surfaces alongside
  nodes / lines / parent-childs, with a Surfaces tab and load/save in
  `GeometryProcessorGUI`. The pyvista backend colours surfaces by nodal modal
  displacement (viridis), updated every animation frame.
- **PreGER multi-setup SSI** (`MultiSetupSSI.py`): `PreGERSSI` point estimates
  and `VarPreGERSSI` uncertainty quantification (covariance- and data-driven),
  with build-time and post-hoc block weighting, wired into `MultiSetupGUI`.
- **`VarPLSCF`** — pLSCF (PolyMAX) variance estimator with build-time and
  post-hoc block weighting.
- **Weighted subspace estimation for `VarSSIRef`**: weighted UPC projections
  and externally-fed correlations with per-block weights.
- `init_from_config`, example configuration files, and GUI support for
  `PreGERSSI`, `VarPreGERSSI`, and `VarPLSCF`.
- Bounded multi-step undo for `PreProcessSignals` (cached spectral estimates
  are invalidated on undo).
- Error-bar visibility checkbox in `StabilGUI`.
- New pyOMA logo.
- Documentation: a polymorphic-UQ application page and block-weighting notes;
  refreshed install / project-structure / contributing sections.

### Changed

- Extracted a backend-agnostic `ModeShapeBase` from `ModeShapePlot`.
- `VarSSIRef` streams the projection Hankel matrices to cut peak memory.

### Fixed

- `VarSSIRef` variance computation: fast-path `B_j1`, eigenvector pairing, and
  the slow-path `sigma_R` rebuild (results produced before this fix need
  regeneration).
- Four pLSCF point-estimate defects (channel weighting, an LSFD typo, the
  integration frequency, and a synthesis conjugate).
- pyvista notebook mode-shape viewer: kernel crash on widget interaction,
  missing arrows/labels, and surfaces rendering as flat grey — resolved by
  client-side vtk.js rendering, `Text3D` label polydata, and opaque surfaces.
- Qt mode-shape animation driven against a real interactor and window.
- Read the Docs build for the PyQt6 GUI pages; the `docs` extra no longer pulls
  an unrelated PyPI package named `pyoma`.

## [1.0.0] - 2026-07-10

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
- Block weighting for the three uncertainty-quantifying methods, in two
  flavours that share one API and both reduce to the classical unweighted
  estimator at uniform weights:
  - *Build-time* weights (a `weights=` argument to the estimator's build
    step) make the point estimate the weighted mean of the blocks and
    propagate into the covariance factor, whose deviations are re-centred on
    that mean, scaled by `sqrt(w_k)`, and normalised by Kish's effective
    sample size `n_eff = 1 / sum(w**2)` in place of the block count (the
    exact scalar follows each class's own covariance normalisation). This is
    the only route that relocates an estimate contaminated by a bad block.
  - *Post-hoc* reweighting leaves the point estimates and every Jacobian at
    their original linearisation and recomputes only the `std_*` arrays, by
    right-multiplying the already-centred factors with a weighting matrix
    `W(w)`: `compute_modal_params_weighted()` re-runs the identification
    loop against the reweighted factor, while `apply_block_weights()`
    reweights per-mode factors cached by an opt-in
    `cache_variance_factors=True` run instead, so a weight sweep over one
    identification costs a matrix product per order. It is a delta-method
    (frozen-linearisation) covariance, first-order consistent for moderate
    weight changes, and requires an unweighted build.
  - Three covariance-normalisation conventions, agreeing at uniform weights:
    `'substitution'` (default — reproduces the covariance factor of a
    build-time weighted run, so a zero weight gives the covariance of a
    from-scratch run with that block deleted: a free jackknife),
    `'reliability'` and `'precision'`.
  - `VarSSIRef` — one weight per block of `build_subspace_mat()`, for both
    `subspace_method='covariance'` and `'projection'`; the latter also
    accepts an experimental pre-LQ weighted reading
    (`experimental_weighted_projection=True`, point estimates only).
  - `VarPLSCF` — one weight per training block of `build_half_spectra()`,
    generalising the `1 / N_avg` covariance scaling of equally weighted
    Welch averages to `1 / n_eff`.
  - `VarPreGERSSI` — weights per setup (one vector per setup, `None` for an
    unweighted one), since each setup's blocks are averaged into that
    setup's own subspace matrix; supported for both the covariance and the
    projection subspace source. The confidence-interval block count
    `num_blocks` follows the effective (Kish) count under weights.
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
