"""
Convert a National Instruments .tdms file to a pyOMA-loadable .npz file.

requires: pip install nptdms

TDMS is not read by any of pyOMA's own dependencies, so nptdms is imported
here rather than at package level - only running this specific script
requires it installed.

Edit the constants below, then run this script directly. The output .npz
uses the same key schema as PreProcessSignals.save_state() and can be
loaded directly via PreProcessSignals.load_state(), including from the
GUI's "Import Signals" action.
"""
import numpy as np
from nptdms import TdmsFile

INPUT_FILE = "measurement.tdms"
OUTPUT_FILE = "measurement.npz"
SETUP_NAME = "Setup 1"

GROUP_NAME = "Untitled"        # name of the TDMS group holding the channels
SAMPLING_RATE = 100.0          # fallback, used if the 'wf_increment' property is absent

# ── Read the file ────────────────────────────────────────────────────────────
tdms_file = TdmsFile.read(INPUT_FILE)
group = tdms_file[GROUP_NAME]
channels = group.channels()

signals = np.column_stack([channel[:] for channel in channels])
channel_headers = [channel.name for channel in channels]

wf_increment = channels[0].properties.get('wf_increment')
sampling_rate = 1.0 / wf_increment if wf_increment else SAMPLING_RATE

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
