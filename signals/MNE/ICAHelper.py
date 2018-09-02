from mne.preprocessing import ICA
import mne


class ICAHelper:
    def fast_ica(self, epochs: mne.Epochs):
        return self._ica_algorithm(epochs=epochs, method='fastica')

    def extended_infomax(self, epochs: mne.Epochs):
        return self._ica_algorithm(epochs=epochs, method='extended-infomax')

    def infomax(self, epochs: mne.Epochs):
        return self._ica_algorithm(epochs=epochs, method='infomax')

    def _ica_algorithm(self, epochs: mne.Epochs, method):
        random_state = 0
        ica = ICA(n_components=0.95, method=method, random_state=random_state).fit(epochs)
        # ica.plot_components(inst=epochs)
        epochs = ica.apply(epochs, exclude=[2])
        return epochs
