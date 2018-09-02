from DataOrganizer import *
from ICAHelper import *
from DataAnalyzer import *

file_path = 'signals/record-FGT-Z-[2017.12.20-08.38.23].csv'
raw, events, channels_to_analyze_indexes = DataOrganizer(file_path).prepare_data()
dataAnalyzer = DataAnalyzer()

tmin = -0.1
tmax = 0.5
epochs_params = dict(events=events, tmin=tmin, tmax=tmax)
corrected_raw, _ = mne.set_eeg_reference(raw, 'average', projection=True)

epochs = mne.Epochs(corrected_raw, **epochs_params, reject=dict(eeg=100), preload=True)
epochs.plot(picks=range(4), block=True, scalings='auto')
print(epochs)
corrected_epochs = ICAHelper().fast_ica(epochs)

dataAnalyzer.classic_averaging(corrected_epochs)
dataAnalyzer.robust_weighted_averaging_absolute(corrected_epochs)
dataAnalyzer.robust_weighted_averaging_quadratic(corrected_epochs)
# dataAnalyzer.GW6(corrected_epochs)
dataAnalyzer.principal_component_analysis(corrected_epochs)
