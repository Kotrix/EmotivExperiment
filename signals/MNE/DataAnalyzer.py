from PCAHelper import *
from RWA import *
from GW6 import GW6


class DataAnalyzer:
    def classic_averaging(self, epochs: mne.Epochs):
        evokeds = epochs.average()
        self._plot_evokeds(evokeds, 'EEG Original reference')

    def principal_component_analysis(self, epochs: mne.Epochs):
        evokeds = PCAHelper().pca(epochs)
        self._plot_evokeds(evokeds, 'PCA')

    def robust_weighted_averaging_absolute(self, epochs: mne.Epochs):
        evokeds = mne.EvokedArray(
            RobustWeightedAveraging().absolute(epochs.get_data()),
            epochs.info,
            tmin=-0.1)

        self._plot_evokeds(evokeds, 'EEG Robust weighted averaging - absolute')

    def robust_weighted_averaging_quadratic(self, epochs: mne.Epochs):
        evokeds = mne.EvokedArray(
            RobustWeightedAveraging().quadratic(epochs.get_data()),
            epochs.info,
            tmin=-0.1)

        self._plot_evokeds(evokeds, 'EEG Robust weighted averaging - quadratic')

    def GW6(self, epochs: mne.Epochs):
        evokeds = mne.EvokedArray(
            GW6(epochs.get_data().transpose(1, 2, 0))[1],
            epochs.info,
            tmin=-0.1)

        self._plot_evokeds(evokeds, 'GW6')


    def _plot_evokeds(self, evokeds: mne.Evoked, title):
        evokeds.plot_joint(title=title)
