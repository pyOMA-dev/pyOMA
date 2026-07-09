"""
Convert a delimited ASCII/CSV file to a pyOMA-loadable .npz file.

Requires: pip install pandas (not a pyOMA dependency itself).

Edit the constants below, then run this script directly. The output .npz
uses the same key schema as PreProcessSignals.save_state() and can be
loaded directly via PreProcessSignals.load_state(), including from the
GUI's "Import Signals" action.
"""
import numpy as np
import pandas as pd

INPUT_FILE = "measurement.csv"
OUTPUT_FILE = "measurement.npz"
SETUP_NAME = "Setup 1"

DELIMITER = ","
HEADER_ROW = 0          # row index with column names, or None if the file has no header
FIRST_COLUMN_IS_TIME = True   # True: drop column 0 and derive sampling_rate from its spacing
SAMPLING_RATE = None    # only used when FIRST_COLUMN_IS_TIME is False - set this manually

# ── Read the file ────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, header=HEADER_ROW)

if FIRST_COLUMN_IS_TIME:
    time = df.iloc[:, 0].to_numpy()
    sampling_rate = 1.0 / np.mean(np.diff(time))
    signals = df.iloc[:, 1:].to_numpy()
    channel_headers = [str(c) for c in df.columns[1:]]
else:
    sampling_rate = SAMPLING_RATE
    signals = df.to_numpy()
    channel_headers = [str(c) for c in df.columns]

# ── Write a pyOMA-loadable .npz ──────────────────────────────────────────────
out_dict = {
    'self.signals': signals,
    'self.sampling_rate': sampling_rate,
    'self.ref_channels': None,
    'self.accel_channels': None,
    'self.velo_channels': None,
    'self.disp_channels': None,
    'self.setup_name': SETUP_NAME,
    'self.channel_headers': channel_headers,
    'self.start_time': None,
    'self.chan_dofs': [],
}
np.savez_compressed(OUTPUT_FILE, **out_dict)
print(f"Wrote {OUTPUT_FILE}")
