import Noise as noise
from mne.viz import plot_evoked_topo
import mne


def plot_joint_evokeds(evokeds: mne.Evoked, title):
    evokeds.apply_baseline()
    print(noise.snr_pp(evokeds.data))
    print(noise.std_for_100ms(evokeds.data))

    if True:
        return
    ts_args = dict(time_unit='ms')
    evokeds.plot_joint(title=title, ts_args=ts_args)
    # evokeds.plot_sensors(show_names=True)


def plot_evokeds(evokeds: mne.Evoked, title):
    evokeds.apply_baseline()
    print(noise.snr_pp(evokeds.data))
    print(noise.snr_mean(evokeds.data))
    evokeds.plot(titles=dict(eeg=title), time_unit='ms')


def plot_compare_evokeds(evokeds: list, title):
    colors = 'tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown',\
             'tab:pink', 'tab:gray', 'tab:olive', 'tab:orange', 'tab:cyan'

    for evoked in evokeds:
        evoked.apply_baseline()

    plot_evoked_topo(evokeds, color=colors[:len(evokeds)], title=title, background_color='w')