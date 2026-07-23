# CLAUDE_TASK: Mode-shape backend split — Phase 0 + 1

**Repo:** `pyOMA-dev/pyOMA` · **Base:** `master` · **Branch:** `feat/msh-backend-core`
**Target:** Opus, high effort, single session. No pyvista in this phase.

---

## 1. Objective

Prepare `PlotMSH.py` for a second rendering backend by (0) writing characterization
tests that pin down the *backend-agnostic* behaviour of `ModeShapePlot`, then (1)
extracting that behaviour into a new `ModeShapeBase` class that `ModeShapePlot`
inherits from.

**Nothing about the rendered output may change.** This is a pure refactor under a
test net that you write first.

---

## 2. Why this is needed

`pyOMA/core/PlotMSH.py` is 2550 lines in one class. Roughly 700–800 lines are pure
computation (merging-mode detection, mode lookup, chan_dof → node displacement
math, parent-child propagation); the remaining ~1700 are matplotlib artist
bookkeeping. A planned pyvista backend must reuse the former and replace the
latter.

`tests/test_plotmsh.py` currently has 5 tests, and **all five assert on matplotlib
internals** (`trace_objects`, `Artist.get_visible()`). There is effectively zero
backend-agnostic coverage today, so there is nothing to prove a second backend
agrees with the first.

---

## 3. Non-goals

Do **not** do any of the following in this task:

- Add pyvista, VTK, or pyvistaqt as a dependency, or write any pyvista code.
- Touch `pyOMA/GUI/*` — the `.fig` / `.subplot` leak is Phase 2.
- Change `GeometryProcessor` — surfaces are Phase 4.
- Change `ModeShapePlotConfig`'s fields. Note in the docstring which fields are
  matplotlib-specific (`beamcolor`, `beamstyle`, `nodecolor`, `nodemarker`,
  `nodesize`, `linewidth`), but leave the dataclass intact for backwards
  compatibility.
- Rename or remove any currently public method or attribute.

---

## 4. Step 0 — Characterization tests (do this first)

Create `tests/test_modeshape_core.py`. These tests must pass against the **current,
unrefactored** `ModeShapePlot`, and must still pass unchanged after Step 1. They
must not touch matplotlib artists.

Available fixtures in `tests/conftest.py`: `geometry_data` (loads
`tests/files/grid.txt`, `lines.txt`, `parent_child_assignments.txt`),
`prep_signals`, `prep_signals_with_corr`, `prep_signals_real`,
`modal_data_ssi_cov`, `modal_data_ssi_data`, `modal_data_plscf`, `test_files_dir`.

Cover at minimum:

**4.1 Merging-mode detection.** `_detect_merging_mode` / `_detect_and_apply_merging`
for all four paths: single-setup, PoGER/PreGER, PoSER (`merged_data`), and the
empty/`prep_signals`-only case. Assert on the resulting `mode_shapes`,
`chan_dofs`, `num_channels`, `select_modes` attributes, not on how they got there.
Use `tests/files/merged_poger` for the PoGER path.

**4.2 Mode lookup.** `_lookup_mode_index` and `change_mode` addressed by
`frequency=`, `index=`, and `mode_index=`. Include the out-of-range and
nearest-frequency behaviour, and `get_frequencies()` ordering.

**4.3 Displacement computation — the important one.** Build a small deterministic
`GeometryProcessor` inline (do *not* rely on `grid.txt` here) plus a hand-written
`chan_dofs` list and a hand-written complex `mode_shape` vector, then assert
`disp_nodes` and `phi_nodes` numerically with `np.testing.assert_allclose`.
Exercise each assignment branch separately:

- `_assign_single_sensor_disp` — one channel on a node
- `_assign_axis_aligned_disp` — 2–3 channels along x/y/z
- `_assign_multi_sensor_disp` — multiple non-aligned sensors
- `_assign_lstsq_sensor_disp` — overdetermined case
- `_compute_parent_child_displacements` — one parent, one child, non-unity
  amplification factors

Also pin `_disp_phase_mag` directly, and cover `real=True` vs `real=False`.

**4.4 State machine.** Default values of the seven `show_*` flags; that
`change_amplitude` scales `disp_nodes` linearly; that `change_part` flips `real`
and recomputes; that `_compute_node_bounds` returns an equal-sided cube.

**4.5 Data animation math.** `_compute_data_disp_nodes(num)` against a synthetic
`prep_signals.signals_filtered`, asserting the accumulated per-node vector.

Mark none of these `@pytest.mark.gui`. They must run in the plain `tests.yml` job.

---

## 5. Step 1 — Extract `ModeShapeBase`

Create `pyOMA/core/ModeShapeBase.py`. **It must contain no `import matplotlib`
statement of any kind** (verified by grep in the acceptance criteria).

### Moves to `ModeShapeBase`

Config & validation: `_resolve_config`, `_build_legacy_config`, `_check_type`,
`_validate_data_types`, `_check_bool`, `_check_numeric`, `_check_int`,
`_check_numeric_or_seq`, `_check_callable_or_none`, `_check_path_or_none`.

Merging: `_detect_merging_mode`, `_check_merging_requirements`,
`_detect_and_apply_merging`, `_apply_poser_attrs`, `_apply_poger_attrs`,
`_apply_single_attrs`, `_apply_empty_attrs`.

Mode selection: `change_mode`, `_lookup_mode_index`, `_get_stabil_params`,
`get_frequencies`, `change_amplitude`, `change_part`.

Displacement math: `_disp_phase_mag`, `_compute_chan_dof_displacements`,
`_assign_single_sensor_disp`, `_assign_axis_aligned_disp`,
`_assign_multi_sensor_disp`, `_assign_lstsq_sensor_disp`,
`_compute_parent_child_displacements`, `_compute_data_disp_nodes`.

Geometry-derived state: `_compute_node_bounds`, the `disp_nodes` / `phi_nodes`
dicts, the `show_*` flags portion of `_init_state`.

### Stays in `ModeShapePlot`

`_setup_figure`, the artist containers in `_init_state`, the matplotlib-specific
validators (`_check_color`, `_check_color_or_seq`, `_check_linestyle_or_seq`,
`_check_marker`), `reset_view`, `_setup_viewport_angles`, `change_viewport`,
all `add_*` / `take_*` geometry-editing methods, all `draw_*` / `refresh_*`,
`set_equal_aspect`, `save_plot`, `stop_ani`, `animate`, all `_animate_*` and
`_data_animate_*`, `filter_and_animate_data`, `_button_press`, `_on_move`,
`_button_release`.

### Template-method hooks

The moved methods currently call rendering code. Break that with no-op hooks
declared on the base and overridden in `ModeShapePlot`:

| Base hook | `ModeShapePlot` override calls |
|---|---|
| `_render_mode()` | the refresh block of `draw_msh` + `set_equal_aspect` + `fig.canvas.draw()` |
| `_render_amplitude()` | existing `change_amplitude` tail |
| `_render_part()` | existing `change_part` tail |

Split `draw_msh` into `compute_mode_displacements()` on the base (pure: rescales
the mode shape, fills `disp_nodes` / `phi_nodes`) and `draw_msh()` on
`ModeShapePlot` (calls the base method, then refreshes artists, restarts the
animation if `self.animated`).

Add an abstract-ish `widget` property on the base raising `NotImplementedError`,
so Phase 2 has somewhere to land. Do not wire it to anything yet.

---

## 6. Constraints

- Style must match the surrounding code: numpydoc docstrings, `logger` from the
  module-level `logging.getLogger(__name__)`, SPDX header on the new file.
- Preserve the existing `DeprecationWarning` behaviour for legacy style kwargs
  exactly, including `stacklevel`.
- Keep `from pyOMA.core.PlotMSH import ModeShapePlot` working; re-export
  `ModeShapeBase` from `pyOMA/core/__init__.py` alongside it.
- Python floor stays `>=3.9` for now (the bump to `>=3.10` lands with the pyvista
  extra in Phase 3).

---

## 7. Acceptance criteria

1. `pytest tests/` — all pre-existing tests pass, **unmodified**. If a pre-existing
   test needs editing to pass, the refactor is wrong; stop and report instead.
2. `pytest -m gui tests/` passes with `QT_QPA_PLATFORM=offscreen`.
3. `tests/test_modeshape_core.py` passes both before Step 1 (commit it first,
   separately) and after.
4. `grep -n "matplotlib" pyOMA/core/ModeShapeBase.py` returns nothing.
5. `ModeShapePlot.__mro__` contains `ModeShapeBase`.
6. Coverage of `pyOMA/core/ModeShapeBase.py` is above 80%.
7. Two commits minimum: one adding the characterization tests against the
   unrefactored class, one performing the extraction. This ordering is the
   evidence that behaviour was preserved.

---

## 8. Report back

- Final line count of `PlotMSH.py` vs `ModeShapeBase.py`.
- Any method you could not cleanly assign to one side, and why.
- Any behaviour the characterization tests revealed to be already broken —
  **document it, do not fix it** in this task.
