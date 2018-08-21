import mne
from EegData import *
from EventsCreator import *


class DataOrganizer:
    def __init__(self, file_path):
        self._triggering_channel = ['F7']
        self._channels_to_analyze = ['P7', 'O1', 'O2', 'P8']
        self._bad_channels = ['AF3', 'F3', 'FC5', 'T7', 'T8', 'FC6', 'F4', 'AF4']
        self._all_channels_to_analyze = self._channels_to_analyze + self._bad_channels
        self._eeg_data = EegData(file_path)

    def prepare_data(self):
        return self._create_raw(), self._create_events(), self._get_channels_to_analyze_indexes()

    def _create_raw(self):
        # Initialize an info structure
        info = mne.create_info(
            ch_names=self._all_channels_to_analyze,
            ch_types="eeg",
            sfreq=128,
            montage='standard_1020')

        raw = mne.io.RawArray(self._eeg_data.get_by_column_names(self._all_channels_to_analyze), info, first_samp=self._eeg_data.experiment_start_index)
        raw.pick_types(meg=False, eeg=True, eog=False)

        print(raw.info)
        self._filter_raw(raw)
        return raw

    def _filter_raw(self, raw: mne.io.Raw):
        raw.filter(1., 15, n_jobs=1, fir_design='firwin')

    def _create_events(self):
        return EventsCreator().create(
            self._eeg_data.get_by_column_names(self._triggering_channel)[0],
            self._eeg_data.get_stimuli(),
            self._eeg_data.experiment_start_index)

    def _get_channels_to_analyze_indexes(self):
        return range(len(self._channels_to_analyze))
