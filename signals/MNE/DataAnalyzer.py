from PCAHelper import *
from RWA import *
from GW6 import GW6
import PlotHelper as plt


class DataAnalyzer:
    def classic_averaging(self, epochs: mne.Epochs):
        evoked = epochs.average()
        plt.plot_joint_evokeds(evoked, 'EEG Original reference')
        return evoked

    def principal_component_analysis(self, epochs: mne.Epochs):
        evoked = PCAHelper().pca(epochs)
        plt.plot_joint_evokeds(evoked, 'PCA')
        return evoked

    def robust_weighted_averaging_absolute(self, epochs: mne.Epochs, input_signal_index):
        evoked = self._to_evoked(RobustWeightedAveraging().absolute(
            epochs.get_data(),
            input_signal_index),
            epochs.info,
            'RWA - absolute ' + str(input_signal_index))

        plt.plot_joint_evokeds(evoked, 'EEG Robust weighted averaging - absolute')
        return evoked

    def robust_weighted_averaging_quadratic(self, epochs: mne.Epochs, input_signal_index):
        evoked = self._to_evoked(RobustWeightedAveraging().quadratic(
            epochs.get_data(),
            input_signal_index),
            epochs.info,
            'RWA - quadratic' + str(input_signal_index))

        plt.plot_joint_evokeds(evoked, 'EEG Robust weighted averaging - quadratic')
        return evoked

    def GW6(self, epochs: mne.Epochs):
        evoked = self._to_evoked(
            GW6(epochs.get_data().transpose(1, 2, 0))[1],
            epochs.info,
            'GW6')

        plt.plot_joint_evokeds(evoked, 'GW6')
        return evoked

    def _to_evoked(self, data: np.ndarray, info, comment):
        return mne.EvokedArray(
            data,
            info,
            tmin=-0.1,
            comment=comment)
