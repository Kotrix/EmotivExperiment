import numpy as np
from MNE_utils import *


class EventsCreator:

    @staticmethod
    def create(raw_trigger_signal: np.ndarray, responses: list):
        events_list = np.zeros((len(responses), 3))

        # Define threshold for trigger signal
        max_trigger_peak_width = time2sample(2)  # in seconds

        trigger_signal = np.gradient(raw_trigger_signal)

        trigger_threshold = EventsCreator._count_treshold(raw_trigger_signal, len(responses))

        # Find next stimuli start and save related epoch for every electrode
        i = 0
        trigger_iter = 0
        iter = 0
        while i < len(trigger_signal):
            if raw_trigger_signal[i] < trigger_threshold:
                try:
                    was_response_correct = responses[trigger_iter][0]
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

                events_list[iter, :] = [stimuli_index, 0, 1]

                i += max_trigger_peak_width
                trigger_iter += 1
                iter += 1
            else:
                i += 1

        return np.unique(events_list.astype(int), axis=0)

    @staticmethod
    def _count_treshold(raw_trigger_signal: np.ndarray, responses_length: int):
        sorted_raw_trigger_signal = np.sort(raw_trigger_signal)
        begin_raw_trigger_signal_median = np.median(sorted_raw_trigger_signal[:responses_length])

        return (2 * begin_raw_trigger_signal_median + np.median(raw_trigger_signal)) / 3
