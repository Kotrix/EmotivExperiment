import mne
from EegData import *
from EventsCreator import *

file_path = 'signals/record-FGT-Z-[2017.12.20-08.38.23].csv'
eeg_data = EegData(file_path)

triggering_electrode = ['F7']
electrodes_to_analyze = ['P7', 'O1', 'O2', 'P8']

# Initialize an info structure
info = mne.create_info(
    ch_names=electrodes_to_analyze,
    ch_types=['eeg', 'eeg', 'eeg', 'eeg'],
    sfreq=128,
    montage='standard_1020')

raw = mne.io.RawArray(eeg_data.get_by_column_names(electrodes_to_analyze), info)
raw.pick_types(meg=False, eeg=True, eog=False)

print(raw.info)
print(raw.info['chs'][0]['loc'])

raw.plot_sensors()

raw.notch_filter([50, 60])
raw.filter(1., 15, n_jobs=1, fir_design='firwin')

raw_no_ref, _ = mne.set_eeg_reference(raw, [])

###############################################################################
# We next define Epochs and compute an ERP
tmin, tmax = -0.1, 0.5
events = EventsCreator().create(
    eeg_data.get_by_column_names(triggering_electrode)[0],
    eeg_data.get_responses())

epochs_params = dict(events=events, tmin=tmin, tmax=tmax)

evoked_no_ref = mne.Epochs(raw_no_ref, **epochs_params).average()
del raw_no_ref  # save memory

title = 'EEG Original reference'
evoked_no_ref.plot(titles=dict(eeg=title), time_unit='s')
evoked_no_ref.plot_topomap(times=[0.1], size=3., title=title, time_unit='s')

###############################################################################
# **Average reference**: This is normally added by default, but can also
# be added explicitly.
raw.del_proj()
raw_car, _ = mne.set_eeg_reference(raw, 'average', projection=True)
evoked_car = mne.Epochs(raw_car, **epochs_params).average()
del raw_car  # save memory

title = 'EEG Average reference'
evoked_car.plot(titles=dict(eeg=title), time_unit='s')
evoked_car.plot_topomap(times=[0.1], size=3., title=title, time_unit='s')
