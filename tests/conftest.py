"""
Shared pytest fixtures for the pyOMA test suite.

Synthetic test data is generated from a simple rod model with known modal
parameters so that integration tests can check recovered frequencies.
"""
import matplotlib
matplotlib.use('Agg')  # headless backend – must precede any other mpl import

import numpy as np
import pytest
from pathlib import Path

from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor

# Wire up the .npy loader before any fixture calls init_from_config
PreProcessSignals.load_measurement_file = staticmethod(np.load)

TEST_FILES = Path(__file__).parent / 'files'

# ── Synthetic data parameters (small enough for fast CI) ────────────────────
SYN_N = 8192        # number of samples
SYN_NODES = 6       # total channels (0 = fixed end, always zero response)
SYN_INP = [5]       # excitation node (free end – non-zero mode shapes)
SYN_REF = [5]       # reference channel for SSI (must have non-zero variance)
SYN_FS = 128        # sampling rate [Hz]
SYN_FSCALE = 10     # force amplitude scale
SYN_MODES = 2       # number of synthesised modes
SYN_SEED = 42       # RNG seed for reproducibility

M_LAGS = 200        # correlation lag length for SSI-based methods
NUM_BLOCK_COLS = 50 # Toeplitz/Hankel block columns
MAX_ORDER = 20      # maximum model order for modal methods


def _synthetic_signals():
    """Return (signals, fs) for a seeded ambient vibration simulation.

    Node 0 is the fixed end of the rod (zero response). Excitation at
    SYN_INP (free end) ensures all other nodes have non-zero variance.
    """
    from tests.system_ambient_ifrf import ambient_ifrf
    _, sig = ambient_ifrf(
        SYN_N, SYN_NODES, SYN_INP, SYN_FS, SYN_FSCALE,
        seed=SYN_SEED, num_modes=SYN_MODES,
    )
    return sig, SYN_FS


# ── Geometry ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def test_files_dir():
    return TEST_FILES


@pytest.fixture(scope='session')
def geometry_data():
    return GeometryProcessor.load_geometry(
        nodes_file=TEST_FILES / 'grid.txt',
        lines_file=TEST_FILES / 'lines.txt',
        parent_childs_file=TEST_FILES / 'parent_child_assignments.txt',
    )


# ── Synthetic prep_signals ────────────────────────────────────────────────────

@pytest.fixture
def prep_signals():
    """Function-scoped fresh PreProcessSignals – safe for mutation tests."""
    sig, fs = _synthetic_signals()
    return PreProcessSignals(sig, fs, ref_channels=SYN_REF)


@pytest.fixture(scope='session')
def prep_signals_with_corr():
    """Session-scoped PreProcessSignals with Welch correlations pre-computed.
    Used as input to SSI-covariance methods."""
    sig, fs = _synthetic_signals()
    ps = PreProcessSignals(sig, fs, ref_channels=SYN_REF)
    ps.corr_welch(m_lags=M_LAGS)
    return ps


# ── Real measurement data ────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def prep_signals_real():
    """PreProcessSignals from the first real test measurement file."""
    meas_dir = TEST_FILES / 'measurement_1'
    return PreProcessSignals.init_from_config(
        conf_file=meas_dir / 'setup_info.txt',
        meas_file=meas_dir / 'measurement_1.npy',
        chan_dofs_file=meas_dir / 'channel_dofs.txt',
    )


# ── Pre-computed modal results (loaded from saved .npz) ──────────────────────

@pytest.fixture(scope='session')
def modal_data_ssi_cov(prep_signals_with_corr):
    """Session-scoped BRSSICovRef result computed from synthetic data."""
    from pyOMA.core.SSICovRef import BRSSICovRef
    obj = BRSSICovRef(prep_signals_with_corr)
    obj.build_toeplitz_cov(NUM_BLOCK_COLS)
    obj.compute_modal_params(MAX_ORDER)
    return obj


@pytest.fixture(scope='session')
def modal_data_ssi_data(prep_signals_with_corr):
    """Session-scoped SSIData result computed from synthetic data."""
    from pyOMA.core.SSIData import SSIData
    obj = SSIData(prep_signals_with_corr)
    obj.build_block_hankel(NUM_BLOCK_COLS)
    obj.compute_modal_params(MAX_ORDER)
    return obj


@pytest.fixture(scope='session')
def modal_data_plscf(prep_signals_with_corr):
    """Session-scoped PLSCF result computed from synthetic data."""
    from pyOMA.core.PLSCF import PLSCF
    obj = PLSCF(prep_signals_with_corr)
    obj.build_half_spectra(nperseg=M_LAGS)
    obj.compute_modal_params(MAX_ORDER)
    return obj
