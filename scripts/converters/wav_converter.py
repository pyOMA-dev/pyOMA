"""
Convert a .wav file to a pyOMA-loadable .npz file.

Zero-extra-dependency example: scipy.io.wavfile.read() supplies both the
signal matrix and the sampling rate directly from the file.

Edit the constants below, then run this script directly. The output .npz
uses the same key schema as PreProcessSignals.save_state() and can be
loaded directly via PreProcessSignals.load_state(), including from the
GUI's "Import Signals" action.
"""
import numpy as np
import scipy.io.wavfile

INPUT_FILE = "measurement.wav"
OUTPUT_FILE = "measurement.npz"
SETUP_NAME = "Setup 1"

# ── Read the file ────────────────────────────────────────────────────────────
sampling_rate, signals = scipy.io.wavfile.read(INPUT_FILE)
signals = signals.astype(np.float64)
if signals.ndim == 1:
    signals = signals[:, np.newaxis]  # mono file -> single channel column

# ── Write a pyOMA-loadable .npz ──────────────────────────────────────────────
out_dict = {
    'self.signals': signals,
    'self.sampling_rate': float(sampling_rate),
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
