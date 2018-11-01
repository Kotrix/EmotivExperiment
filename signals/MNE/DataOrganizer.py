import mne
from EegData import *
from EventsCreator import *
import Noise as noise
from enum import Enum


class NoiseType(Enum):
    GAUSSIAN = 1
    SALT_AND_PEPPER = 2
    SPECKLE = 3


class DataOrganizer:
    def __init__(self, file_path):
        self._triggering_channel = ['F7']
        self._channels_to_analyze = ['P7', 'O1', 'O2', 'P8']
        self._bad_channels = ['AF3', 'F3', 'FC5', 'T7', 'T8', 'FC6', 'F4', 'AF4']
        self._all_channels_to_analyze = self._channels_to_analyze + self._bad_channels
        self._eeg_data = EegData(file_path)

    def prepare_data(self, noise_type=None):
        return self._create_raw(noise_type), self._create_events(), self._get_channels_to_analyze_indexes()

    def _create_raw(self, noise_type=None):
        # Initialize an info structure
        info = mne.create_info(
            ch_names=self._all_channels_to_analyze,
            ch_types="eeg",
            sfreq=128,
            montage='standard_1020')

        raw = mne.io.RawArray(
            self._eeg_data.get_by_column_names(self._all_channels_to_analyze),
            info,
            first_samp=self._eeg_data.experiment_start_index)

        raw.pick_types(meg=False, eeg=True, eog=False)

        self._add_noise(raw, noise_type)

        print(raw.info)
        return raw

    def _add_noise(self, raw: mne.io.Raw, noise_type=None):
        if noise_type is None:
            return

        if noise_type == NoiseType.GAUSSIAN:
            raw._data = noise.add_gaussian(raw.get_data())
        elif noise_type == NoiseType.SALT_AND_PEPPER:
            raw._data = noise.add_salt_and_pepper(raw.get_data())
        elif noise_type == NoiseType.SPECKLE:
            raw._data = noise.add_speckle(raw.get_data())


    def _filter_raw(self, raw: mne.io.Raw):
        raw.filter(1., 15., n_jobs=1, fir_design='firwin')

    def _create_events(self):
        return EventsCreator().create(
            self._eeg_data.get_by_column_names(self._triggering_channel)[0],
            self._eeg_data.get_stimuli(),
            self._eeg_data.experiment_start_index)

    def _get_channels_to_analyze_indexes(self):
        return range(len(self._channels_to_analyze))
