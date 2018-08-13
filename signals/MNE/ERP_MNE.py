import mne
from EegData import *
from EventsCreator import *
from ICAHelper import *

file_path = 'signals/record-FGT-Z-[2017.12.20-08.38.23].csv'
eeg_data = EegData(file_path)

triggering_channel = ['F7']
channels_to_analyze = ['P7', 'O1', 'O2', 'P8']
bad_channels = ['AF3', 'F3', 'FC5', 'T7', 'T8', 'FC6', 'F4', 'AF4']
all_channels_to_analyze = channels_to_analyze + bad_channels

# Initialize an info structure
info = mne.create_info(
    ch_names=all_channels_to_analyze,
    ch_types=["eeg"] * len(all_channels_to_analyze),
    sfreq=128,
    montage='standard_1020')

raw = mne.io.RawArray(eeg_data.get_by_column_names(all_channels_to_analyze), info)
raw.pick_types(meg=False, eeg=True, eog=False)

print(raw.info)

raw.notch_filter([50, 60])
raw.filter(1., 15, n_jobs=1, fir_design='firwin')

# Independent Component Analysis
corrected_raw = ICAHelper().fast_ica(raw)

raw_no_ref, _ = mne.set_eeg_reference(corrected_raw, [])

###############################################################################
# We next define Epochs and compute an ERP
tmin, tmax = -0.1, 0.5
events = EventsCreator().create(
    eeg_data.get_by_column_names(triggering_channel)[0],
    eeg_data.get_responses())

epochs_params = dict(events=events, tmin=tmin, tmax=tmax)

evoked_no_ref = mne.Epochs(raw_no_ref, **epochs_params).average()
del raw_no_ref  # save memory

title = 'EEG Original reference'
evoked_no_ref.plot(picks=range(0, len(channels_to_analyze)), titles=dict(eeg=title), time_unit='s')

###############################################################################
# **Average reference**: This is normally added by default, but can also
# be added explicitly.
corrected_raw.del_proj()
raw_car, _ = mne.set_eeg_reference(corrected_raw, 'average', projection=True)
evoked_car = mne.Epochs(raw_car, **epochs_params).average()


del raw_car  # save memory

title = 'EEG Average reference'
evoked_car.plot(picks=range(0, len(channels_to_analyze)), titles=dict(eeg=title), time_unit='s')
