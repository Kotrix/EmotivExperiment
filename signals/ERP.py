from data_utils import *
import scipy
############# CONFIG ###########################
database_regex = 'csv/1/record-F*.csv'

triggering_electrode = 'F7'
electrodes_to_analyze = ['P7', 'P8']

common_avg_ref = True
ref_electrodes = ['AF3','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','AF4'] if common_avg_ref else []

all_electrodes = np.unique([triggering_electrode] + electrodes_to_analyze + ref_electrodes)

# Filtering options
filter_on = True
low_cutoff = 0.2
high_cutoff = 24 # desired cutoff frequency of the filter, Hz (0 to disable filtering)

# Define pre and post stimuli period of chunk (in seconds)
pre_stimuli = time2sample(0.2)
post_stimuli = time2sample(1.0)

# Define threshold for trigger signal
max_trigger_peak_width = time2sample(3) # in seconds
slope_width = 9 # in number of samples, controls shift of the stimuli start

# Valid signal value limits
chunk_max_peak_to_peak = 70
peak_filtering = False
min_peak_score = 2


######### READ SIGNALS FROM DISK ############################

database = read_openvibe_csv_database(database_regex, all_electrodes)

if common_avg_ref:
    ######## COMMON AVERAGE REFERENCE ###########################
    for filename, record in database.items():
        # Find average signal over all ref_electrodes
        mean = np.zeros(len(record['signals'][triggering_electrode]))
        for electrode, signal in record['signals'].items():
            if electrode in ref_electrodes:
                mean += signal
        mean /= len(ref_electrodes)

        # Change reference of all electrodes
        for electrode, signal in record['signals'].items():
            if electrode in ref_electrodes:
                database[filename]['signals'][electrode] -= mean


#plot_database(database, 1)


########### SIGNALS FILTERING ###############
from filtering import *

# Modify database by filtering signals
for filename, record in database.items():
    for electrode, signal in record['signals'].items():
        if electrode == triggering_electrode or filter_on:
            # filtered_signal = butter_lowpass_filter(signal, high_cutoff, fs, 6)
            filtered_signal = fft_bandpass_filter(signal, low_cutoff, high_cutoff, fs)

            database[filename]['signals'][electrode] = filtered_signal
#plot_database(database, 1)


############ CUT THE END OF SIGNAL TO REMOVE FILTERING ARTIFACTS ##################
# Times (in seconds) to cut signals in the beginning and end
left_cut = time2sample(0)
right_cut = time2sample(3)

for filename, record in database.items():
    database[filename]['timestamps'] = record['timestamps'][left_cut:-right_cut]
    for electrode, signal in record['signals'].items():
        database[filename]['signals'][electrode] = signal[left_cut:-right_cut]
#plot_database(database, 1)


########### DISCARD SIGNALS WITHOUT TRIGGER FOUND ###########
trigger_peak_to_peak_min = 500

# Filter by peak-to-peak value
corrupted_files = list()
for filename, record in database.items():
    triggering_signal = record['signals'][triggering_electrode]
    if np.max(triggering_signal) - np.min(triggering_signal) < trigger_peak_to_peak_min:
        print("Cannot find triggering signal in:", filename)
        corrupted_files.append(filename)

if len(corrupted_files) == 0:
    print("No corrupted signals")
else:
    for filename in corrupted_files:
        database.pop(filename, None)

######### EXTRACT CHUNKS OF SIGNAL AFTER STIMULI ##################

def is_face_emotional(face_id):
    neutral_labels = [2, 6, 9, 11, 15, 17, 19, 24]
    for i in neutral_labels:
        if face_id == 33024 + i:
            return False
    return True


def forward_diff(signal, order):
    new_signal = np.zeros_like(signal)
    for i in range(len(signal) - order):
        new_signal[i] = np.sum(np.diff(signal[i:(i + order)]))

    return new_signal


for filename, record in database.items():

    # Init container for ERP chunks
    extracted_chunks_emo = OrderedDict()
    extracted_chunks_neutral = OrderedDict()
    for e in electrodes_to_analyze:
        extracted_chunks_emo[e] = list()
        extracted_chunks_neutral[e] = list()

    # Compute forward difference of triggering electrode signal and find its minima
    raw_trigger_signal = np.array(record['signals'][triggering_electrode])
    trigger_signal = forward_diff(raw_trigger_signal, slope_width)
    trigger_threshold = (np.mean(np.sort(trigger_signal)[:len(record['responses'])]) + np.mean(np.sort(trigger_signal))) / 2

    # #Compare raw triggering signal and its difference
    # plt.figure()
    # plt.plot(range(len(trigger_signal)), raw_trigger_signal, 'b-')
    # plt.plot(range(len(trigger_signal)), trigger_signal, 'g-', linewidth=1)
    # plt.axhline(trigger_threshold, color='k', linestyle='dashed')
    # plt.xlabel('Time [s]')
    # plt.ylabel('uV')
    # plt.grid()
    # plt.show()

    # Find next stimuli start and save related chunk for every electrode
    i = 0
    trigger_iter = 0
    true_timestamps = list()
    openvibe_timestamps = list()
    while i < len(trigger_signal):
        if trigger_signal[i] < trigger_threshold:

            try:
                was_response_correct = record['responses'][trigger_iter][0]
            except:
                i += 1
                continue

            if was_response_correct:
                # Find stimuli index
                search_area_start = max(0, i - max_trigger_peak_width // 2)
                search_area_end = min(i + max_trigger_peak_width // 2, len(trigger_signal))
                stimuli_index = int(search_area_start + np.argmin(trigger_signal[search_area_start:search_area_end]))

                # #Plot single triggers
                # margin = 10
                # plt.figure()
                # plt.plot(range(len(trigger_signal[stimuli_index-margin:stimuli_index+margin])), raw_trigger_signal[stimuli_index-margin:stimuli_index+margin], 'b-')
                # plt.plot(range(len(trigger_signal[stimuli_index-margin:stimuli_index+margin])), trigger_signal[stimuli_index-margin:stimuli_index+margin], 'g-', linewidth=1)
                # plt.axvline(margin, color='k', linestyle='dashed')
                # plt.xlabel('Time [s]')
                # plt.ylabel('uV')
                # plt.grid()
                # plt.show()

                try:
                    true_timestamps.append(record['timestamps'][stimuli_index])
                    openvibe_timestamps.append(record['order'][trigger_iter][1])
                except:
                    pass

                # Save chunk
                for electrode, signal in record['signals'].items():
                    if electrode in electrodes_to_analyze:
                        if stimuli_index - pre_stimuli < 0 or stimuli_index + post_stimuli > len(signal):
                            continue
                        chunk = signal[stimuli_index - pre_stimuli:stimuli_index + post_stimuli]
                        chunk_max = np.max(chunk)
                        chunk_min = np.min(chunk)
                        chunk_peak_to_peak = chunk_max - chunk_min

                        if peak_filtering:
                            def inv_ric(points, a):
                                return -scipy.signal.ricker(points, a)

                            widths = 0.5 * np.arange(1, 10)
                            cwtmatr = scipy.signal.cwt(chunk, inv_ric, widths)
                            peak_score = np.mean(cwtmatr[:, pre_stimuli + time2sample(0.17)])

                            # plt.figure()
                            # plt.title(peak_score)
                            # plt.imshow(cwtmatr, extent=[-0.2, 1, chunk.min(), chunk.max()], cmap='PRGn', aspect='auto',
                            #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
                            # plt.show()
                        else:
                            peak_score = min_peak_score + 1

                        if trigger_iter < 10:
                            #Plot triggers
                            plt.figure()
                            plt.title(electrode + (', peak score: ' + str(peak_score)) if peak_filtering else '')
                            plt.plot(((np.array(range(len(chunk)))) - pre_stimuli)/fs, chunk, 'g-', linewidth=1)
                            plt.axvline(0, color='k', linestyle='dashed')
                            plt.axvline(0.17, color='b', linestyle='dashed')
                            plt.xlabel('Time [s]')
                            plt.ylabel('uV')
                            plt.grid()
                            plt.savefig("figures\\" + basename(filename) + "\\chunks\\"+electrode+'_'+str(trigger_iter)+ ('_common' if common_avg_ref else '_org') +'.png')
                            plt.close()

                        if chunk_peak_to_peak < chunk_max_peak_to_peak and peak_score > min_peak_score:
                            try:
                                face_id = record['order'][trigger_iter][0]
                                if len(record['order']) > 0 and is_face_emotional(face_id):
                                    extracted_chunks_emo[electrode].append(chunk)
                                else:
                                    extracted_chunks_neutral[electrode].append(chunk)
                            except:
                                pass

            i += max_trigger_peak_width
            trigger_iter += 1
        else:
            i += 1

    # Calculate difference between true timestamps and openvibe timestamps
    diff = np.subtract(true_timestamps[:len(openvibe_timestamps)], openvibe_timestamps)
    m = np.mean(diff)
    v = np.std(diff)
    print(filename, m, v)


    ######### AVERAGE CHUNKS ##################
    invert_y_axis = False

    # N170 area 140-185ms
    n170_begin = pre_stimuli + time2sample(0.14)
    n170_end = pre_stimuli + time2sample(0.185)

    for electrode in electrodes_to_analyze:

        chunks_emo = extracted_chunks_emo[electrode]
        chunks_neutral = extracted_chunks_neutral[electrode]

        if len(chunks_emo) == 0 or len(chunks_neutral) == 0:
            continue

        # Grand-average over all chunks
        averaged_emo = np.mean(chunks_emo, axis=0)
        averaged_neutral = np.mean(chunks_neutral, axis=0)

        # change voltage scale as difference from baseline
        averaged_emo -= np.mean(averaged_emo[:pre_stimuli+1])
        averaged_neutral -= np.mean(averaged_neutral[:pre_stimuli+1])

        plt.figure()
        plt.title(electrode + ' - ' + str(len(chunks_emo)) + '/' + str(len(chunks_neutral)) + ' chunks average')
        plt.plot(np.multiply(np.arange(len(averaged_emo)) - pre_stimuli, 1000 / fs), averaged_emo, color='r',
                 linestyle='dashed')
        plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), averaged_neutral, color='g')
        plt.axvline(0, color='k', linestyle='dashed')
        plt.axhline(0, color='k', linestyle='dashed')
        plt.xlabel('Time [ms]')
        plt.ylabel('uV')
        if invert_y_axis:
            plt.gca().invert_yaxis()
        figure_file = "figures\\" + basename(filename) + "\\erp\\" + electrode + ('_common_' if common_avg_ref else '_org_') + str(int(filter_on*high_cutoff)) + 'Hz'
        plt.savefig(figure_file + '.png')
        #plt.show()

        print(electrode)
        print(len(chunks_emo), '\t', len(chunks_neutral))
        print(averaged_emo[n170_begin:n170_end].min(), '\t', averaged_neutral[n170_begin:n170_end].min())

        with open(figure_file + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|')
            writer.writerow(['emo', 'neutral'])
            writer.writerow([len(chunks_emo), len(chunks_neutral)])
            writer.writerow([averaged_emo[n170_begin:n170_end].min(), averaged_neutral[n170_begin:n170_end].min()])

    print('\n')