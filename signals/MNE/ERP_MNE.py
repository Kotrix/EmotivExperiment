from DataOrganizer import *
from ICAHelper import *

file_path = 'signals/record-FGT-Z-[2017.12.20-08.38.23].csv'
raw, events, channels_to_analyze_indexes = DataOrganizer(file_path).prepare_data()

# Independent Component Analysis
corrected_raw = ICAHelper().fast_ica(raw)

###############################################################################
# We next define Epochs and compute an ERP
tmin, tmax = -0.1, 0.5
epochs_params = dict(events=events, tmin=tmin, tmax=tmax)

raw_no_ref, _ = mne.set_eeg_reference(corrected_raw, [])
epochs_no_ref = mne.Epochs(raw_no_ref, **epochs_params)
epochs_no_ref.plot(picks=channels_to_analyze_indexes, block=True, scalings='auto')
evoked_no_ref = epochs_no_ref.average()
del raw_no_ref  # save memory

title = 'EEG Original reference'
evoked_no_ref.plot(picks=channels_to_analyze_indexes, titles=dict(eeg=title), time_unit='s')

corrected_raw.del_proj()
###############################################################################
# **Average reference**: This is normally added by default, but can also
# be added explicitly.
raw_car, _ = mne.set_eeg_reference(corrected_raw, 'average', projection=True)
epochs_car = mne.Epochs(raw_car, **epochs_params)
epochs_car.plot(picks=channels_to_analyze_indexes, block=True, scalings='auto')
evoked_car = epochs_car.average()
del raw_car  # save memory

title = 'EEG Average reference'
evoked_car.plot(picks=channels_to_analyze_indexes, titles=dict(eeg=title), time_unit='s')
