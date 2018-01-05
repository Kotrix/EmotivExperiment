############# CONFIG ###########################
database_regex = 'csv/record-FGT*.csv'
electrodes = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','AF4']
#electrodes = ['F7','P7','P8']

triggering_electrode = 'F7'
fs = 128 # EPOC sampling freq, Hz

# Filtering options
low_cutoff = 0.2
high_cutoff = 24 # desired cutoff frequency of the filter, Hz (0 to disable filtering)

# Define pre and post stimuli period of chunk (in seconds)
pre_stimuli = 0.1
post_stimuli = 0.4

# Define threshold for trigger signal
trigger_threshold = -600 # trigger peak has to be below this value
max_trigger_peak_width = 3 # in seconds
slope_width = 4 # in number of samples, controls shift of the stimuli start

# Valid signal value limits
chunk_lower_limit = -50
chunk_upper_limit = 50
chunk_max_peak_to_peak = 70

# Please save this config as txt in the database folder after adjusting

# Transform periods in seconds to number of samples
pre_stimuli = int(pre_stimuli * fs)
post_stimuli = int(post_stimuli * fs)
max_trigger_peak_width = int(max_trigger_peak_width * fs)


######### READ SIGNALS FROM DISK ############################

from data_utils import *
database = read_openvibe_csv_database(database_regex, electrodes)

######## COMMON AVERAGE REFERENCE ###########################
#Modify database by filtering signals
for filename, record in database.items():
    mean = np.zeros(len(record['signals'][triggering_electrode]))
    for electrode, signal in record['signals'].items():
        if electrode != triggering_electrode:
            mean += signal
    mean /= len(electrodes) - 1
    for electrode, signal in record['signals'].items():
        if electrode != triggering_electrode:
            database[filename]['signals'][electrode] -= mean


# plot_database(database, 1)


########### SIGNALS FILTERING ###############
if high_cutoff > 0:
    from filtering import *

    # Modify database by filtering signals
    for filename, record in database.items():
        for electrode, signal in record['signals'].items():
            if electrode != triggering_electrode:
                #filtered_signal = butter_lowpass_filter(signal, high_cutoff, fs, 6)
                filtered_signal = fft_bandpass_filter(signal, low_cutoff, high_cutoff, fs)

                database[filename]['signals'][electrode] = filtered_signal
#plot_database(database, 1)


############ CUT THE END OF SIGNAL TO REMOVE FILTERING ARTIFACT ##################
# Times (in seconds) to cut signals in the beginning and end
left_cut = 0
right_cut = 3
#convert to sample number
left_cut = int(left_cut*fs)
right_cut = max(1, int(right_cut*fs))

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

# Init container for ERP chunks
extracted_chunks_emo = OrderedDict()
extracted_chunks_neutral = OrderedDict()
for e in electrodes:
    if e != triggering_electrode:
        extracted_chunks_emo[e] = list()
        extracted_chunks_neutral[e] = list()


def is_face_emotional(face_id):
    neutral_labels = [2, 6, 9, 11, 15, 17, 19, 24]
    for i in neutral_labels:
        if face_id == 33024 + i:
            return False
    return True


def forward_diff(signal, order):
    new_signal = np.zeros_like(signal)
    for i in range(len(signal) - order):
        new_signal[i] = np.sum(np.diff(signal[i:i + order]))

    return new_signal


for filename, record in database.items():

    # Compute forward difference of triggering electrode signal and find its minima
    raw_trigger_signal = np.array(record['signals'][triggering_electrode])
    trigger_signal = forward_diff(raw_trigger_signal, slope_width)

    # plt.figure()
    # plt.plot(range(len(trigger_signal)),
    #          raw_trigger_signal, 'b-')
    # plt.plot(range(len(trigger_signal)),
    #          trigger_signal, 'g-', linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('uV')
    # plt.grid()
    # plt.show()


    # Find next stimuli start and save related chunk for every electrode
    i = 0
    trigger_iter = 0
    ts = list()
    os = list()
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

                #Plot triggers
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
                    ts.append(record['timestamps'][stimuli_index])
                    os.append(record['order'][trigger_iter][1])
                except:
                    pass

                # Save chunk
                for electrode, signal in record['signals'].items():
                    if electrode in extracted_chunks_emo:
                        if stimuli_index - pre_stimuli < 0 or stimuli_index + post_stimuli > len(signal):
                            continue
                        chunk = signal[stimuli_index - pre_stimuli:stimuli_index + post_stimuli]
                        chunk_max = np.max(chunk)
                        chunk_min = np.min(chunk)
                        chunk_peak_to_peak = chunk_max - chunk_min

                        if trigger_iter < 0:
                            #Plot triggers
                            plt.figure()
                            plt.title(electrode)
                            plt.plot(np.array(range(len(chunk)))/fs, chunk, 'g-', linewidth=1)
                            plt.axvline(0.1, color='k', linestyle='dashed')
                            plt.axvline(0.27, color='b', linestyle='dashed')
                            plt.xlabel('Time [s]')
                            plt.ylabel('uV')
                            plt.grid()
                            plt.savefig("figures\\one_chunk\\"+electrode+'_'+str(trigger_iter)+'.png')
                            plt.close('all')

                        if chunk_min > chunk_lower_limit and chunk_max < chunk_upper_limit and chunk_peak_to_peak < chunk_max_peak_to_peak:
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

    diff = np.subtract(ts[:len(os)], os)
    m = np.mean(diff)
    v = np.std(diff)
    print(filename, m, v)


######### AVERAGE CHUNKS ##################
invert_y_axis = False

# N170 area 140-185ms
n170_begin = pre_stimuli + int(0.14 * fs)
n170_end = pre_stimuli + int(0.185 * fs)

for electrode in extracted_chunks_emo:

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
    plt.draw()
    plt.savefig("figures\\n170\\Robert\\" + electrode + '_' + 'Robert_FET_correct_Common_' + str(int(high_cutoff)) + '.png')
    plt.show()

    print(electrode)
    print(len(chunks_emo), '\t', len(chunks_neutral))
    print(averaged_emo[n170_begin:n170_end].min(), '\t', averaged_neutral[n170_begin:n170_end].min())
    print('\n')