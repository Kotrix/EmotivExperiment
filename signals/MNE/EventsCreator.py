import numpy as np
from Time import *


class EventsCreator:

    def create(self, raw_trigger_signal: np.ndarray, responses: list, offset: int):
        events_list = list()  # np.zeros((len(responses), 3))

        # Define threshold for trigger signal
        max_trigger_peak_width = Time.to_sample(2)  # in seconds

        trigger_signal = np.gradient(raw_trigger_signal)

        trigger_threshold = self._count_treshold(raw_trigger_signal, len(responses))

        # Find next stimuli start and save related epoch for every electrode
        i = 0
        trigger_iter = 0
        while i < len(trigger_signal):
            if raw_trigger_signal[i] < trigger_threshold:
                try:
                    was_response_correct = responses[trigger_iter][1]
                except:
                    i += 1
                    continue

                if was_response_correct:
                    # Find stimuli index
                    margin = max_trigger_peak_width // 2
                    search_area_start = max(0, i - margin)
                    search_area_end = min(i + margin + 1, len(trigger_signal))
                    stimuli_index = int(
                        search_area_start + np.argmin(trigger_signal[search_area_start:search_area_end])) - 4
                    if stimuli_index < margin:
                        i += max_trigger_peak_width
                        trigger_iter += 1
                        continue

                    events_list.append([offset + stimuli_index, 0, responses[trigger_iter][0]])

                i += max_trigger_peak_width
                trigger_iter += 1
            else:
                i += 1

        return self._sorted_and_distinct_events(events_list)

    def _sorted_and_distinct_events(self, events_list: list):
        return self._unique_events(self._sort_events(events_list))

    def _unique_events(self, sorted_events: np.ndarray):
        u, ind = np.unique(sorted_events, return_index=True, axis=0)
        return u[np.argsort(ind)]

    def _sort_events(self, events_list: set):
        events_list.sort(key=lambda x: x[0])
        return np.asarray(events_list)

    def _count_treshold(self, raw_trigger_signal: np.ndarray, responses_length: int):
        sorted_raw_trigger_signal = np.sort(raw_trigger_signal)
        begin_raw_trigger_signal_median = np.median(sorted_raw_trigger_signal[:responses_length])

        return (2 * begin_raw_trigger_signal_median + np.median(raw_trigger_signal)) / 3
