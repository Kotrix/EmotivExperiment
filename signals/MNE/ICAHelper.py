from mne.preprocessing import ICA
import mne


class ICAHelper:
    def fast_ica(self, raw: mne.io.Raw):
        return self._ica_algorithm(raw=raw, method='fastica')

    def extended_infomax(self, raw):
        return self._ica_algorithm(raw=raw, method='extended-infomax')

    def _ica_algorithm(self, raw: mne.io.Raw, method):
        corrected_raw = raw.copy()
        random_state = 0

        ica = ICA(method=method, random_state=random_state)
        print(ica)
        ica.fit(corrected_raw)
        print(ica)
        ica.plot_components(inst=corrected_raw)
        ica.apply(corrected_raw)
        return corrected_raw
