import mne
from EegData import *


file_path = 'signals/record-FGT-Z-[2017.12.20-08.38.23].csv'
eeg_data = EegData(file_path)

triggering_electrode = 'F7'
electrodes_to_analyze = ['P7', 'O1', 'O2', 'P8']

# Initialize an info structure
info = mne.create_info(
    ch_names=electrodes_to_analyze,
    ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
    sfreq=128
)

custom_raw = mne.io.RawArray(eeg_data.get_eeg_data(electrodes_to_analyze), info)

custom_epochs = mne.EpochsArray()