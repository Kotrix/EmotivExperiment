import mne
import numpy as np
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA


class PCAHelper:
    def pca(self, epochs: mne.Epochs):
        epochs_data = epochs.get_data()
        _, number_of_channels, _ = np.shape(epochs_data)

        pca = UnsupervisedSpatialFilter(PCA(number_of_channels), average=True)
        pca_data = pca.fit_transform(epochs_data)
        return mne.EvokedArray(np.mean(pca_data, axis=0),
                               mne.create_info(number_of_channels,
                                               epochs.info['sfreq'],
                                               ch_types='eeg'),
                               tmin=-0.1)
