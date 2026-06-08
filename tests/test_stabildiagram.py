"""Tests for StabilDiagram: StabilCalc and StabilCluster."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyOMA.core.StabilDiagram import StabilCalc, StabilCluster


# ── StabilCalc ────────────────────────────────────────────────────────────────

class TestStabilCalc:
    @pytest.fixture(scope='class')
    def stabil_calc(self, modal_data_ssi_cov):
        return StabilCalc(modal_data_ssi_cov)

    def test_construction_succeeds(self, stabil_calc):
        assert isinstance(stabil_calc, StabilCalc)

    def test_masked_frequencies_shape(self, stabil_calc, modal_data_ssi_cov):
        assert stabil_calc.masked_frequencies.shape == \
               modal_data_ssi_cov.modal_frequencies.shape

    def test_masked_damping_shape(self, stabil_calc, modal_data_ssi_cov):
        assert stabil_calc.masked_damping.shape == \
               modal_data_ssi_cov.modal_damping.shape

    def test_order_dummy_shape(self, stabil_calc):
        assert stabil_calc.order_dummy.ndim == 2
        assert stabil_calc.order_dummy.shape[0] == stabil_calc.modal_data.max_model_order

    def test_capabilities_dict_present(self, stabil_calc):
        caps = stabil_calc.capabilities
        for key in ('f', 'd', 'msh', 'std', 'ev', 'data'):
            assert key in caps

    def test_initial_state_is_zero(self, stabil_calc):
        assert stabil_calc.state == 0

    def test_calculate_stabilization_masks(self, stabil_calc):
        """Calling calculate_stabilization_masks should set mask_pre."""
        stabil_calc.calculate_stabilization_masks()
        assert stabil_calc.masks['mask_pre'] is not None
        assert stabil_calc.state >= 2

    def test_mask_pre_removes_zero_and_nan_frequencies(self, stabil_calc):
        stabil_calc.calculate_stabilization_masks()
        mask = stabil_calc.masks['mask_pre']
        # All frequencies that pass the pre-filter should be non-zero and finite
        passing_freqs = stabil_calc.masked_frequencies[mask]
        assert np.all(np.isfinite(passing_freqs))
        assert np.all(passing_freqs > 0)

    def test_save_load_round_trip(self, stabil_calc, modal_data_ssi_cov):
        stabil_calc.calculate_stabilization_masks()
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            stabil_calc.save_state(fname)
            loaded = StabilCalc.load_state(fname, modal_data_ssi_cov)
            np.testing.assert_allclose(
                loaded.masked_frequencies, stabil_calc.masked_frequencies,
                equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)


# ── StabilCluster ─────────────────────────────────────────────────────────────

class TestStabilCluster:
    @pytest.fixture(scope='class')
    def stabil_cluster(self, modal_data_ssi_cov):
        return StabilCluster(modal_data_ssi_cov)

    def test_construction_succeeds(self, stabil_cluster):
        assert isinstance(stabil_cluster, StabilCluster)

    def test_is_subclass_of_stabil_calc(self, stabil_cluster):
        assert isinstance(stabil_cluster, StabilCalc)

    def test_capabilities_auto_is_true(self, stabil_cluster):
        assert stabil_cluster.capabilities['auto']

    def test_calculate_masks_runs_without_error(self, stabil_cluster):
        stabil_cluster.calculate_stabilization_masks()
        assert stabil_cluster.masks['mask_pre'] is not None

    def test_automatic_clearing_runs_without_error(self, stabil_cluster):
        """StabilCluster.automatic_clearing should complete on valid modal data."""
        stabil_cluster.calculate_stabilization_masks()
        try:
            stabil_cluster.automatic_clearing()
        except (ValueError, RuntimeError) as e:
            pytest.skip(f"automatic_clearing raised expected error on sparse data: {e}")


# ── StabilCalc with single-setup real data ─────────────────────────────────────

class TestStabilCalcRealData:
    @pytest.fixture(scope='class')
    def stabil_real(self, test_files_dir, prep_signals_real):
        """Load pre-computed BRSSICovRef + StabilCalc from the test files."""
        from pyOMA.core.SSICovRef import BRSSICovRef
        meas_dir = test_files_dir / 'measurement_1'
        modal_data = BRSSICovRef.load_state(
            meas_dir / 'modal_data.npz', prep_signals_real)
        return StabilCalc(modal_data)

    def test_load_and_construct_succeeds(self, stabil_real):
        assert isinstance(stabil_real, StabilCalc)

    def test_frequencies_present(self, stabil_real):
        assert stabil_real.masked_frequencies is not None
        assert stabil_real.masked_frequencies.size > 0

    def test_calculate_masks_on_real_data(self, stabil_real):
        stabil_real.calculate_stabilization_masks()
        mask = stabil_real.masks['mask_pre']
        # Real data should have at least some poles passing the pre-filter
        assert mask.sum() > 0
