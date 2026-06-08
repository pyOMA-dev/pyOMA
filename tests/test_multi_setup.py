"""Tests for multi-setup analysis: PogerSSICovRef and MergePoSER.

The PoGer test processes two real measurement files end-to-end and is marked
@pytest.mark.slow.  The MergePoSER smoke test uses a minimal in-memory setup.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor
from pyOMA.core.SSICovRef import BRSSICovRef, PogerSSICovRef
from pyOMA.core.StabilDiagram import StabilCalc
from pyOMA.core.PostProcessingTools import MergePoSER

TEST_FILES = Path(__file__).parent / 'files'


# ── PogerSSICovRef ────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestPogerSSICovRef:
    """Full PoGer pipeline using the first two real measurement setups.

    Marked slow – exclude with: pytest -m "not slow"
    """

    @pytest.fixture(scope='class')
    def poger_result(self):
        modal = PogerSSICovRef()
        for i in (1, 2):
            meas_dir = TEST_FILES / f'measurement_{i}'
            ps = PreProcessSignals.init_from_config(
                conf_file=meas_dir / 'setup_info.txt',
                meas_file=meas_dir / f'measurement_{i}.npy',
                chan_dofs_file=meas_dir / 'channel_dofs.txt',
            )
            ps.corr_blackman_tukey(m_lags=200)
            modal.add_setup(ps)

        modal.pair_channels()
        # m_lags=200 → max num_block_columns+num_block_rows < 200, use 80
        modal.build_merged_subspace_matrix(num_block_columns=80)
        modal.compute_modal_params(max_model_order=40, max_modes=24)
        return modal

    def test_construction_succeeds(self, poger_result):
        assert isinstance(poger_result, PogerSSICovRef)

    def test_state_complete(self, poger_result):
        # PoGer state = [subspace_built, channels_paired, modal_params, setups_added, ...]
        # state[4] is an optional step not used in the basic workflow
        assert poger_result.state[0], "Subspace matrix not built"
        assert poger_result.state[1], "Channels not paired"
        assert poger_result.state[2], "Modal params not computed"

    def test_modal_frequencies_computed(self, poger_result):
        assert poger_result.modal_frequencies is not None

    def test_modal_frequencies_shape(self, poger_result):
        freqs = poger_result.modal_frequencies
        assert freqs.ndim == 2
        assert freqs.shape[0] == 40  # max_model_order

    def test_frequencies_below_nyquist(self, poger_result):
        freqs = poger_result.modal_frequencies
        valid = freqs[~np.isnan(freqs)]
        # Measurement sampling rate is 256 Hz → Nyquist = 128 Hz
        assert np.all(valid >= 0)
        assert np.all(valid < 128.0)

    def test_save_load_round_trip(self, poger_result):
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            poger_result.save_state(fname)
            loaded = PogerSSICovRef.load_state(fname)
            np.testing.assert_allclose(
                loaded.modal_frequencies, poger_result.modal_frequencies,
                rtol=1e-10, equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)


# ── MergePoSER ────────────────────────────────────────────────────────────────

class TestMergePoSER:
    """Smoke tests for MergePoSER using pre-computed single-setup results."""

    @pytest.fixture(scope='class')
    def single_setup_stabil(self, test_files_dir):
        """Load a pre-computed BRSSICovRef + StabilCalc from the test files."""
        meas_dir = test_files_dir / 'measurement_1'
        prep = PreProcessSignals.init_from_config(
            conf_file=meas_dir / 'setup_info.txt',
            meas_file=meas_dir / 'measurement_1.npy',
            chan_dofs_file=meas_dir / 'channel_dofs.txt',
        )
        modal = BRSSICovRef.load_state(meas_dir / 'modal_data.npz', prep)
        stabil = StabilCalc.load_state(meas_dir / 'stabil_data.npz', modal)
        return prep, modal, stabil

    def test_add_setup_requires_chan_dofs(self, single_setup_stabil):
        prep, modal, stabil = single_setup_stabil
        merger = MergePoSER()
        if not prep.chan_dofs:
            pytest.skip("chan_dofs not available in loaded state")
        merger.add_setup(prep, modal, stabil)
        assert merger.state[0]

    def test_construction(self):
        merger = MergePoSER()
        assert merger.mean_frequencies is None
        assert merger.mean_damping is None
        assert merger.merged_mode_shapes is None

    def test_save_load_empty_state(self):
        merger = MergePoSER()
        merger.state[0] = True   # pretend setups were added
        merger.mean_frequencies = np.array([1.0, 2.0, 3.0])
        merger.mean_damping = np.array([0.02, 0.03, 0.04])
        merger.std_frequencies = np.array([0.001, 0.001, 0.001])
        merger.merged_mode_shapes = np.eye(3, dtype=complex)
        merger.merged_chan_dofs = []
        merger.state[1] = True

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            merger.save_state(fname)
            loaded = MergePoSER.load_state(fname)
            np.testing.assert_allclose(
                loaded.mean_frequencies, merger.mean_frequencies)
        finally:
            Path(fname).unlink(missing_ok=True)
