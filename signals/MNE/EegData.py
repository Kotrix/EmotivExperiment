import numpy as np


class EegData:
    def __init__(self, file_path):
        self._eeg_data = self.read_from_csv(file_path)
        self._create_responses_and_order()

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

    def _create_responses_and_order(self):
        self._order = list()
        self._responses = list()

        face_min_id = 33025
        face_max_id = 33048
        experiment_start_id = '32769'
        experiment_started = False

        event_ids = self.get_by_column_names(['Event_Id'])[0]
        event_dates = self.get_by_column_names(['Event_Date'])[0]

        # Read timestamps, signal values and stimuli
        index = -1
        for event_id in event_ids:
            index += 1

            # Start reading from the ExperimentStart event
            if experiment_started or event_id.split(':')[0] == experiment_start_id:
                experiment_started = True
            else:
                continue

            if event_id != '':
                stimuli_id = int(event_id.split(':')[0])
                event_timestamp = float(event_dates[index].split(':')[0])
                if face_min_id <= stimuli_id <= face_max_id:
                    self._order.append((stimuli_id, event_timestamp))
                elif stimuli_id == 770:
                    self._responses.append((True, event_timestamp))
                elif stimuli_id == 769:
                    self._responses.append((False, event_timestamp))

    def get_responses(self):
        return self._responses

    def get_order(self):
        return self._order
