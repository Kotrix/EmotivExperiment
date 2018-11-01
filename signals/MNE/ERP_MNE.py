from DataOrganizer import *
from ICAHelper import *
from DataAnalyzer import *
import PlotHelper as plt


noise_type = None
isRWA = False
withICA = True
needCompare = True

file_path = 'signals/1/record-FGT-Z-[2017.12.20-08.38.23].csv'
raw, events, channels_to_analyze_indexes = DataOrganizer(file_path).prepare_data(noise_type=noise_type)
dataAnalyzer = DataAnalyzer()

tmin = -0.1
tmax = 0.5
epochs_params = dict(events=events, tmin=tmin, tmax=tmax)

# raw.plot(scalings='auto')


### CAR ###
if(isRWA):
    corrected_raw = raw
else:
    corrected_raw, _ = mne.set_eeg_reference(raw, 'average', projection=True)
    corrected_raw.apply_proj()


### FILTERING ###

if(not isRWA):
    corrected_raw.filter(1., 15., n_jobs=1, fir_design='firwin')


### EPOCHING ###

if(isRWA):
    epochs = mne.Epochs(corrected_raw, **epochs_params, preload=True)
else:
    epochs = mne.Epochs(corrected_raw, **epochs_params, reject=dict(eeg=100), preload=True)

# epochs.plot(picks=range(4), block=True, scalings='auto')


### ICA ###

if(not isRWA and withICA):
    fast_ica_epochs = ICAHelper().fast_ica(epochs.copy())
    infomax_epochs = ICAHelper().infomax(epochs.copy())


### METHOD ###

evokeds = list()

if(isRWA):
    for input_index in range(0, 10):
        evokeds.append(dataAnalyzer.robust_weighted_averaging_absolute(epochs.copy(), input_index))
        # evokeds.append(dataAnalyzer.robust_weighted_averaging_quadratic(epochs.copy(), input_index))
else:
    classic_averaging_evoked = dataAnalyzer.classic_averaging(epochs)
    classic_averaging_evoked.comment = 'Arithmetic Average'
    evokeds.append(classic_averaging_evoked)

    if(withICA):
        infomax_evoked = dataAnalyzer.classic_averaging(infomax_epochs)
        infomax_evoked.comment = 'ICA - InfoMax'
        evokeds.append(infomax_evoked)

        fast_ica_evoked = dataAnalyzer.classic_averaging(fast_ica_epochs)
        fast_ica_evoked.comment = 'ICA - FastICA'
        evokeds.append(fast_ica_evoked)

if(needCompare):
    plt.plot_compare_evokeds(evokeds, 'ERPs')

input()
