"""
Convert a MATLAB .mat file to a pyOMA-loadable .npz file.

Edit the constants below, then run this script directly. The output .npz
uses the same key schema as PreProcessSignals.save_state() and can be
loaded directly via PreProcessSignals.load_state(), including from the
GUI's "Import Signals" action.
"""
import numpy as np
import scipy.io

INPUT_FILE = "measurement.mat"
OUTPUT_FILE = "measurement.npz"
SETUP_NAME = "Setup 1"

SIGNAL_VAR = "signals"        # name of the variable holding the (n_samples, n_channels) matrix
SAMPLING_RATE_VAR = None      # name of a scalar variable holding the sampling rate, or None
SAMPLING_RATE = 100.0         # used when SAMPLING_RATE_VAR is None

# ── Read the file ────────────────────────────────────────────────────────────
mat = scipy.io.loadmat(INPUT_FILE)
signals = mat[SIGNAL_VAR]
sampling_rate = float(mat[SAMPLING_RATE_VAR].squeeze()) if SAMPLING_RATE_VAR else SAMPLING_RATE

# ── Write a pyOMA-loadable .npz ──────────────────────────────────────────────
out_dict = {
    'self.signals': signals,
    'self.sampling_rate': sampling_rate,
    'self.ref_channels': None,
    'self.accel_channels': None,
    'self.velo_channels': None,
    'self.disp_channels': None,
    'self.setup_name': SETUP_NAME,
    'self.channel_headers': None,
    'self.start_time': None,
    'self.chan_dofs': [],
}
np.savez_compressed(OUTPUT_FILE, **out_dict)
print(f"Wrote {OUTPUT_FILE}")
