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

raw.notch_filter([50, 60])
raw.filter(1., 15, n_jobs=1, fir_design='firwin')

events = EventsCreator.create(
    eeg_data.get_by_column_names(triggering_electrode)[0],
    eeg_data.get_responses())

# raw.set_eeg_reference('average', projection=True)
# raw.plot(n_channels=raw.info['nchan'], block=True, scalings='auto', duration=500)

raw_no_ref, _ = mne.set_eeg_reference(raw, [])

###############################################################################
# We next define Epochs and compute an ERP for the left auditory condition.
reject = dict(eeg=180e-6)
event_id, tmin, tmax = {'left/auditory': 1}, -0.2, 0.5
# events = mne.read_events(event_fname)
epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     reject=reject)

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

###############################################################################
# **Custom reference**: Use the mean of channels EEG 001 and EEG 002 as
# a reference
raw_custom, _ = mne.set_eeg_reference(raw, ['EEG 001', 'EEG 002'])
evoked_custom = mne.Epochs(raw_custom, **epochs_params).average()
del raw_custom  # save memory

title = 'EEG Custom reference'
evoked_custom.plot(titles=dict(eeg=title), time_unit='s')
evoked_custom.plot_topomap(times=[0.1], size=3., title=title, time_unit='s')

###############################################################################
# Evoked arithmetics
# ------------------
#
# Trial subsets from Epochs can be selected using 'tags' separated by '/'.
# Evoked objects support basic arithmetic.
# First, we create an Epochs object containing 4 conditions.

event_id = {'left/auditory': 1, 'right/auditory': 2,
            'left/visual': 3, 'right/visual': 4}
epochs_params = dict(events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                     reject=reject)
epochs = mne.Epochs(raw, **epochs_params)

print(epochs)

###############################################################################
# Next, we create averages of stimulation-left vs stimulation-right trials.
# We can use basic arithmetic to, for example, construct and plot
# difference ERPs.

left, right = epochs["left"].average(), epochs["right"].average()

# create and plot difference ERP
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
mne.combine_evoked([left, -right], weights='equal').plot_joint(**joint_kwargs)

###############################################################################
# This is an equal-weighting difference. If you have imbalanced trial numbers,
# you could also consider either equalizing the number of events per
# condition (using
# :meth:`epochs.equalize_event_counts <mne.Epochs.equalize_event_counts>`).
# As an example, first, we create individual ERPs for each condition.

aud_l = epochs["auditory", "left"].average()
aud_r = epochs["auditory", "right"].average()
vis_l = epochs["visual", "left"].average()
vis_r = epochs["visual", "right"].average()

all_evokeds = [aud_l, aud_r, vis_l, vis_r]
print(all_evokeds)

###############################################################################
# This can be simplified with a Python list comprehension:
all_evokeds = [epochs[cond].average() for cond in sorted(event_id.keys())]
print(all_evokeds)

# Then, we construct and plot an unweighted average of left vs. right trials
# this way, too:
mne.combine_evoked(
    all_evokeds, weights=(0.25, -0.25, 0.25, -0.25)).plot_joint(**joint_kwargs)

###############################################################################
# Often, it makes sense to store Evoked objects in a dictionary or a list -
# either different conditions, or different subjects.

# If they are stored in a list, they can be easily averaged, for example,
# for a grand average across subjects (or conditions).
grand_average = mne.grand_average(all_evokeds)
mne.write_evokeds('/tmp/tmp-ave.fif', all_evokeds)

# If Evokeds objects are stored in a dictionary, they can be retrieved by name.
all_evokeds = dict((cond, epochs[cond].average()) for cond in event_id)
print(all_evokeds['left/auditory'])

# Besides for explicit access, this can be used for example to set titles.
for cond in all_evokeds:
    all_evokeds[cond].plot_joint(title=cond, **joint_kwargs)
