import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import csv
import os
import numpy as np

def prepare_dirs(*args):
    """
    Create directories if not exist.
    :param args: directory names or lists of directory names.
    :return: None
    """
    for d in args:
        if type(d) == list:
            prepare_dirs(*d)
            continue
        if not os.path.isdir(d):
            os.makedirs(d)

figures_dir = 'figures'
prepare_dirs(figures_dir)
prepare_dirs(os.path.join(figures_dir, 'all'))
fs = 128 # EPOC sampling freq, Hz

def time2sample(time):
    """
    :param time: in seconds
    :return: integer number of the nearest sample
    """
    return int(round(time * fs))

def sample2time(sample):
    """
    :param sample: number of samples
    :return: time of given sample
    """
    return sample / fs

def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std

def scale(signal, low, high):
    min = np.min(signal)
    max = np.max(signal)
    rng = max - min
    return high - (((high - low) * (max - signal)) / rng)

def basename(filename):
    return os.path.basename(filename)[:-4]

face_min_id = 33025
face_max_id = 33048
experiment_start_id = '32769'

def read_openvibe_csv(filename, electrodes):
    """
    return dictionary with list of timestamps, signals dictionary, stimuli dictionary for every electrode
    """

    # Prepare directory for results
    new_dir = os.path.join(figures_dir, basename(filename))
    prepare_dirs(new_dir)
    prepare_dirs(new_dir + 'chunks')
    prepare_dirs(new_dir + 'erp')

    # Init containers
    timestamps = list()
    order = list()
    responses = list()
    signals = OrderedDict()
    for e in electrodes:
        signals[e] = list()
    event_col = 16  # default event column without gyroscope

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        # Find electrodes columns numbers
        header = next(reader)
        col_numbers = OrderedDict()
        for e in electrodes:
            for i, val in enumerate(header):
                if e == val:
                    col_numbers[val] = i
                elif val == 'Event Id':
                    event_col = i
        
        # Read timestamps, signal values and stimuli
        experiment_started = False
        correct_answers = 0
        answer_time = 0
        for row in reader:

            # Start reading from the ExperimentStart event
            if experiment_started or row[event_col].split(':')[0] == experiment_start_id:
                experiment_started = True
            else:
                continue

            timestamps.append(float(row[0]))
            for e, col in col_numbers.items():
                signals[e].append(float(row[col]))
            if row[event_col] != '':
                stimuli_id = int(row[event_col].split(':')[0])
                event_timestamp = float(row[event_col+1].split(':')[0])
                if face_min_id <= stimuli_id <= face_max_id:
                    order.append((stimuli_id, event_timestamp))
                elif stimuli_id == 770:
                    responses.append((True, event_timestamp))
                    correct_answers += 1
                    answer_time += event_timestamp - order[-1][1]
                elif stimuli_id == 769:
                    responses.append((False, event_timestamp))
                    answer_time += event_timestamp - order[-1][1]

    print(filename, 'score:', 100 * correct_answers / len(responses), 'avg. time:', 1000 * answer_time / len(responses))
    
    return {'timestamps': timestamps, 'signals': signals, 'order': order, 'responses': responses}


def read_openvibe_csv_database(database_regex, electrodes):
    """
    return dictionary [filename matching database_regex] -> [EEG data in dictionary]
    """
    files = glob.glob(database_regex)
    
    # Init output dictionary
    signals = OrderedDict()
    
    for filename in files:
        signals[filename] = read_openvibe_csv(filename, electrodes)
    
    print(len(signals), 'files loaded')
    
    return signals

def plot_database(database, file=None):
    """
    plot whole database (dictionary of [filename matching glob_regex] -> [timestamps, dictionary of signals for electrodes])
    file: name or number of the specific file to plot
    """
    name = None
    if file is not None:
        try:
            filenum = int(file) - 1
            if 0 <= filenum < len(database):
                i = 0
                for key in database.keys():
                    if i == filenum:
                        name = key
                        break
                    i += 1
        except:
            pass


    for filename, record in database.items():

        if file is not None and name != filename:
            continue
        
        timestamps = record['timestamps']

        for electrode, signal in record['signals'].items():
            plt.figure()
            plt.title(electrode + ' - ' + filename)
            plt.plot(timestamps, signal, 'g-', linewidth=1)
            for id, time in record['order']:
                plt.axvline(time, linestyle='dashed', label=str(id))
            plt.xlabel('Time [s]')
            plt.ylabel('uV')
            #plt.grid()
            plt.show()
