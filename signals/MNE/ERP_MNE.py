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
    sfreq=128,
    montage='standard_1020'
)

raw = mne.io.RawArray(eeg_data.get_eeg_data(electrodes_to_analyze), info)
raw.pick_types(meg=False, eeg=True, eog=False)
print(raw.info)

raw.notch_filter([50, 60])
raw.filter(1., 15, n_jobs=1, fir_design='firwin')

raw.set_eeg_reference('average', projection=True)
raw.plot(n_channels=raw.info['nchan'], block=True, scalings='auto', duration=500)


