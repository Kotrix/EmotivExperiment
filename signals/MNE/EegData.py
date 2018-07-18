import numpy as np


class EegData:
    _eeg_data = None

    def __init__(self, file_path):
        self.read_eeg_data_from_csv(file_path)

    def read_eeg_data_from_csv(self, file_path):
        print('read file: ' + file_path)
        self._eeg_data = np.genfromtxt(file_path, delimiter=',', names=True)

    def get_eeg_data(self, column_names: list):
        return np.array(self._eeg_data[column_names].tolist())

