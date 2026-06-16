"""Integration tests for pyOMA modal analysis methods.

Tests run on a small synthetic dataset (8192 samples, 6 channels, 128 Hz)
with a seeded random-number generator so results are deterministic.

Slow tests (full VarSSIRef pipeline) are marked with @pytest.mark.slow
and can be excluded with: pytest -m "not slow"
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyOMA.core.PreProcessingTools import PreProcessSignals
from tests.conftest import M_LAGS, NUM_BLOCK_COLS, MAX_ORDER, SYN_FS

NYQUIST = SYN_FS / 2


# ── Shared assertion helpers ──────────────────────────────────────────────────

def _check_modal_arrays(modal_obj, max_order):
    """Verify that modal_frequencies and modal_damping have the expected shape."""
    freqs = modal_obj.modal_frequencies
    damp = modal_obj.modal_damping
    assert freqs is not None, "modal_frequencies is None"
    assert damp is not None, "modal_damping is None"
    assert freqs.ndim == 2, f"expected 2-D frequency array, got shape {freqs.shape}"
    assert freqs.shape[0] == max_order
    assert damp.shape == freqs.shape

def _check_physical_range(modal_obj):
    """Identified (non-zero, finite) frequencies must be positive and below Nyquist."""
    freqs = modal_obj.modal_frequencies
    damp = modal_obj.modal_damping
    # Unidentified poles are stored as 0.0 or NaN – exclude both
    identified = np.isfinite(freqs) & (freqs > 0)
    valid_f = freqs[identified]
    valid_d = damp[identified]
    if valid_f.size:
        assert np.all(valid_f < NYQUIST), f"Frequencies above Nyquist ({NYQUIST} Hz)"
    if valid_d.size:
        assert np.all(valid_d >= 0), "Negative damping ratios present"


# ── BRSSICovRef ───────────────────────────────────────────────────────────────

class TestBRSSICovRef:
    def test_build_and_compute_succeed(self, modal_data_ssi_cov):
        # state = [Toeplitz_built, (PoGer_pair_channels), ModalParams_computed]
        # state[1] is only set by the multi-setup PoGer pipeline
        assert modal_data_ssi_cov.state[0], "Toeplitz matrix not built"
        assert modal_data_ssi_cov.state[2], "Modal parameters not computed"

    def test_output_array_shapes(self, modal_data_ssi_cov):
        _check_modal_arrays(modal_data_ssi_cov, MAX_ORDER)

    def test_physical_frequency_range(self, modal_data_ssi_cov):
        _check_physical_range(modal_data_ssi_cov)

    def test_mode_shapes_shape(self, modal_data_ssi_cov):
        msh = modal_data_ssi_cov.mode_shapes
        n_ch = modal_data_ssi_cov.num_analised_channels
        assert msh.shape[0] == n_ch
        assert msh.shape[1] == MAX_ORDER

    def test_save_load_round_trip(self, modal_data_ssi_cov, prep_signals_with_corr):
        from pyOMA.core.SSICovRef import BRSSICovRef
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            modal_data_ssi_cov.save_state(fname)
            loaded = BRSSICovRef.load_state(fname, prep_signals_with_corr)
            np.testing.assert_allclose(
                loaded.modal_frequencies, modal_data_ssi_cov.modal_frequencies,
                rtol=1e-10, equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_load_from_saved_state(self, prep_signals_real, test_files_dir):
        """Smoke-test: load a pre-computed BRSSICovRef result from disk."""
        from pyOMA.core.SSICovRef import BRSSICovRef
        meas_dir = test_files_dir / 'measurement_1'
        obj = BRSSICovRef.load_state(meas_dir / 'modal_data.npz', prep_signals_real)
        assert obj is not None
        assert obj.modal_frequencies is not None

    def test_init_from_config(self, tmp_path, prep_signals_with_corr):
        from pyOMA.core.SSICovRef import BRSSICovRef
        cfg = tmp_path / 'ssi.txt'
        cfg.write_text('Number of Block-Columns:\n10\nMaximum Model Order:\n5\n')
        obj = BRSSICovRef.init_from_config(cfg, prep_signals_with_corr)
        assert obj.state[0], 'Toeplitz matrix not built'
        assert obj.state[2], 'Modal parameters not computed'
        assert obj.modal_frequencies.shape[0] == 5


# ── SSIData ───────────────────────────────────────────────────────────────────

class TestSSIData:
    def test_build_and_compute_succeed(self, modal_data_ssi_data):
        # state[0] = block Hankel built; state[2] = modal params computed
        # state[1] and state[3] are for optional cross-validation steps
        assert modal_data_ssi_data.state[0], "Block-Hankel matrix not built"
        assert modal_data_ssi_data.state[2], "Modal parameters not computed"

    def test_output_array_shapes(self, modal_data_ssi_data):
        _check_modal_arrays(modal_data_ssi_data, MAX_ORDER)

    def test_physical_frequency_range(self, modal_data_ssi_data):
        _check_physical_range(modal_data_ssi_data)

    def test_save_load_round_trip(self, modal_data_ssi_data, prep_signals_with_corr):
        from pyOMA.core.SSIData import SSIData
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            modal_data_ssi_data.save_state(fname)
            loaded = SSIData.load_state(fname, prep_signals_with_corr)
            np.testing.assert_allclose(
                loaded.modal_frequencies, modal_data_ssi_data.modal_frequencies,
                rtol=1e-10, equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_init_from_config(self, tmp_path, prep_signals_with_corr):
        from pyOMA.core.SSIData import SSIData
        cfg = tmp_path / 'ssi.txt'
        cfg.write_text('Number of Block-Columns:\n10\nMaximum Model Order:\n5\n')
        obj = SSIData.init_from_config(cfg, prep_signals_with_corr)
        assert obj.state[0], 'Block-Hankel matrix not built'
        assert obj.state[2], 'Modal parameters not computed'
        assert obj.modal_frequencies.shape[0] == 5


# ── PLSCF ─────────────────────────────────────────────────────────────────────

class TestPLSCF:
    def test_build_and_compute_succeed(self, modal_data_plscf):
        # PLSCF state = [HalfSpectra_built, ModalParams_computed]
        assert all(modal_data_plscf.state), \
            f"Not all stages completed: {modal_data_plscf.state}"

    def test_output_array_shapes(self, modal_data_plscf):
        _check_modal_arrays(modal_data_plscf, MAX_ORDER)

    def test_physical_frequency_range(self, modal_data_plscf):
        _check_physical_range(modal_data_plscf)

    def test_save_load_round_trip(self, modal_data_plscf, prep_signals_with_corr):
        from pyOMA.core.PLSCF import PLSCF
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            modal_data_plscf.save_state(fname)
            loaded = PLSCF.load_state(fname, prep_signals_with_corr)
            np.testing.assert_allclose(
                loaded.modal_frequencies, modal_data_plscf.modal_frequencies,
                rtol=1e-10, equal_nan=True)
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_init_from_config_real_data(self, prep_signals_real, test_files_dir):
        from pyOMA.core.PLSCF import PLSCF
        obj = PLSCF.init_from_config(
            test_files_dir / 'plscf_config.txt', prep_signals_real)
        assert all(obj.state)

    def test_init_from_config(self, tmp_path, prep_signals_with_corr):
        from pyOMA.core.PLSCF import PLSCF
        cfg = tmp_path / 'plscf.txt'
        cfg.write_text(
            'Begin Frequency:\n0\n'
            'End Frequency:\n64\n'
            'Samples per time segment:\n200\n'
            'Maximum Model Order:\n5\n'
        )
        obj = PLSCF.init_from_config(cfg, prep_signals_with_corr)
        assert all(obj.state)
        assert obj.modal_frequencies.shape[0] == 5


# ── PRCE ──────────────────────────────────────────────────────────────────────

class TestPRCE:
    def test_init_from_config(self, tmp_path, prep_signals_real):
        # PRCE requires num_ref_channels >= 2; prep_signals_real has ref [3, 4]
        from pyOMA.core.PRCE import PRCE
        cfg = tmp_path / 'prce.txt'
        cfg.write_text(
            'Number of Correlation Samples:\n50\n'
            f'Maximum Model Order:\n{MAX_ORDER}\n'
        )
        obj = PRCE.init_from_config(cfg, prep_signals_real)
        assert all(obj.state)
        assert obj.modal_frequencies.shape[0] == MAX_ORDER


# ── ModalBase utilities ───────────────────────────────────────────────────────

class TestModalBase:
    def test_remove_conjugates_filters_real_eigenvalues(self):
        from pyOMA.core.ModalBase import ModalBase
        # eigenvalues: one real (overdamped), one conjugate pair
        eigval = np.array([0.99 + 0j,          # real → removed
                           0.95 + 0.1j,        # first of conjugate pair
                           0.95 - 0.1j])       # second → removed (conjugate of [1])
        conj_inds = ModalBase.remove_conjugates(eigval, inds_only=True)
        # The real eigenvalue and one of the conjugate pair are removed.
        # Implementation keeps the first-encountered member of each conjugate pair.
        assert len(conj_inds) == 1
        assert eigval[conj_inds[0]].imag != 0   # the surviving pole is complex

    def test_remove_conjugates_filters_unstable_eigenvalues(self):
        from pyOMA.core.ModalBase import ModalBase
        eigval = np.array([1.5 + 0.1j,   # |λ| > 1 → unstable, removed
                           0.9 + 0.1j,
                           0.9 - 0.1j])
        conj_inds = ModalBase.remove_conjugates(eigval, inds_only=True)
        retained = eigval[conj_inds]
        assert np.all(np.abs(retained) <= 1.0)

    def test_rescale_mode_shape_max_component_is_one(self):
        from pyOMA.core.ModalBase import ModalBase
        msh = np.array([0.5 + 0.1j, 2.0 + 0.0j, -1.0 + 0.5j])
        scaled = ModalBase.rescale_mode_shape(msh)
        assert abs(abs(scaled[np.argmax(np.abs(scaled))]) - 1.0) < 1e-12

    def test_integrate_quantities_converts_accel_to_disp(self):
        from pyOMA.core.ModalBase import ModalBase
        omega = 2 * np.pi * 5.0  # 5 Hz mode
        msh = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        integrated = ModalBase.integrate_quantities(
            msh, accel_channels=[0, 1], velo_channels=[2], omega=omega)
        # accel → disp: multiply by -1/ω²
        expected_accel = -1 / omega**2
        np.testing.assert_allclose(integrated[:2], expected_accel, rtol=1e-10)


# ── VarSSIRef (slow – uses covariance estimation) ─────────────────────────────

@pytest.mark.slow
class TestVarSSIRef:
    """Full VarSSIRef pipeline.  Excluded from the default run with -m 'not slow'."""

    @pytest.fixture(scope='class')
    def varssi_result(self, prep_signals_with_corr, test_files_dir):
        from pyOMA.core.VarSSIRef import VarSSIRef
        obj = VarSSIRef(prep_signals_with_corr)
        obj.build_subspace_mat(
            num_block_columns=30,
            num_blocks=4,
            subspace_method='covariance',
        )
        obj.compute_state_matrices(max_model_order=15, lsq_method='pinv')
        obj.prepare_sensitivities(variance_algo='fast')
        obj.compute_modal_params()
        return obj

    def test_modal_frequencies_computed(self, varssi_result):
        assert varssi_result.modal_frequencies is not None

    def test_output_shapes(self, varssi_result):
        freqs = varssi_result.modal_frequencies
        assert freqs.ndim == 2

    def test_physical_frequency_range(self, varssi_result):
        _check_physical_range(varssi_result)

    def test_frequency_uncertainty_positive(self, varssi_result):
        std_f = varssi_result.std_frequencies
        if std_f is not None:
            valid = std_f[~np.isnan(std_f)]
            assert np.all(valid >= 0)

    def test_init_from_config(self, tmp_path, prep_signals_with_corr):
        from pyOMA.core.VarSSIRef import VarSSIRef
        cfg = tmp_path / 'varssi.txt'
        cfg.write_text(
            'Number of Block-Columns:\n10\n'
            'Maximum Model Order:\n5\n'
            'Number of Blocks:\n2\n'
            'Subspace Method (projection/covariance):\ncovariance\n'
            'LSQ Method for A (pinv/qr):\npinv\n'
            'Variance Algorithm (fast/slow):\nfast\n'
        )
        obj = VarSSIRef.init_from_config(cfg, prep_signals_with_corr)
        assert all(obj.state)
        assert obj.modal_frequencies.shape[0] == 5
