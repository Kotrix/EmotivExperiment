import numpy as np


class EegData:
    def __init__(self, file_path):
        self._eeg_data = self.read_from_csv(file_path)
        self._find_experiment_start_index()
        self._create_responses()

    def read_from_csv(self, file_path):
        print('read file: ' + file_path)
        return np.genfromtxt(
            file_path,
            delimiter=',',
            names=True,
            dtype=[
                ('Time:128Hz', 'f8'),
                ('Epoch', 'i8'),
                ('AF3', 'f8'),
                ('F7', 'f8'),
                ('F3', 'f8'),
                ('FC5', 'f8'),
                ('T7', 'f8'),
                ('P7', 'f8'),
                ('O1', 'f8'),
                ('O2', 'f8'),
                ('P8', 'f8'),
                ('T8', 'f8'),
                ('FC6', 'f8'),
                ('F4', 'f8'),
                ('F8', 'f8'),
                ('AF4', 'f8'),
                ('Gyro-X', 'f8'),
                ('Gyro-Y', 'f8'),
                ('Event Id', 'U20'),
                ('Event Date', 'U20'),
                ('Event Duration', 'U30')])

    def get_by_column_names(self, column_names: list):
        return np.array(self._eeg_data[column_names].tolist()).T

    def _find_experiment_start_index(self):
        experiment_start_id = '32769'

        event_ids = self.get_by_column_names(['Event_Id'])[0]

        index = 0
        for event_id in event_ids:
            if event_id.split(':')[0] == experiment_start_id:
                break
            else:
                index += 1

        self.experiment_start_index = index
        self._eeg_data = self._eeg_data[self.experiment_start_index:]

    def _create_responses(self):
        self._responses = list()

        event_ids = self.get_by_column_names(['Event_Id'])[0]
        event_dates = self.get_by_column_names(['Event_Date'])[0]

        # Read timestamps, signal values and stimuli
        for event_id, event_date in zip(event_ids, event_dates):

            if event_id != '':
                stimuli_id = int(event_id.split(':')[0])
                event_timestamp = float(event_date.split(':')[0])
                if stimuli_id == 770:
                    self._responses.append((True, event_timestamp))
                elif stimuli_id == 769:
                    self._responses.append((False, event_timestamp))

    def get_responses(self):
        return self._responses
