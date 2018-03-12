from data_utils import *
import scipy

############# CONFIG ###########################
person_id = -1
database_regex = 'csv/'+ str(person_id if person_id > 0 else '*') + '/record-F*.csv'
suffix='ALL'

triggering_electrode = 'F7'
electrodes_to_analyze = ['P7', 'P8', 'O1','O2']

common_avg_ref = True
ref_electrodes = ['AF3','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','AF4'] if common_avg_ref else []

all_electrodes = np.unique([triggering_electrode] + electrodes_to_analyze + ref_electrodes)

# Filtering options
filter_on = True
low_cutoff = 0.5
high_cutoff = 24 # desired cutoff frequency of the filter, Hz (0 to disable filtering)

# Define pre and post stimuli period of epoch (in seconds)
pre_stimuli = time2sample(0.1)
post_stimuli = time2sample(0.5)

# Define threshold for trigger signal
max_trigger_peak_width = time2sample(3) # in seconds
slope_width = 7 # in number of samples, controls shift of the stimuli start

# Valid signal value limits
epoch_max_peak_to_peak = 70
peak_filtering = True
min_peak_score = 0

csv_logs = ['latencies', 'n170_amplitudes', 'epn_amplitudes', 'resps']
csv_writers = dict()
for name in csv_logs:
    csvfile = open(name + ('_common_' if common_avg_ref else '_org_') + suffix + '.csv', 'w', newline='')
    csv_writers[name] = csv.writer(csvfile, delimiter=',', quotechar='|')
    csv_writers[name].writerow(['emotional', 'neutral'])


######### READ SIGNALS FROM DISK ############################

database = read_openvibe_csv_database(database_regex, all_electrodes)
#plot_database(database, 1)

if common_avg_ref:
    ######## COMMON AVERAGE REFERENCE ###########################
    common_database = database.copy()
    for filename, record in database.items():
        for e in electrodes_to_analyze:
            # Find average signal over all ref_electrodes
            mean = np.zeros(len(record['signals'][triggering_electrode]))
            count = 0
            for electrode, signal in record['signals'].items():
                if electrode in ref_electrodes:
                    mean += signal
                    count += 1
            mean /= count

            common_database[filename]['signals'][e] -= mean

    database = common_database


#plot_database(database, 1)



########### SIGNALS FILTERING ###############
from filtering import *

# Modify database by filtering signals
for filename, record in database.items():
    for electrode, signal in record['signals'].items():
        if electrode == triggering_electrode or filter_on:
            filtered_signal = butter_bandpass_filter(signal, [low_cutoff, high_cutoff], fs)

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

######### EXTRACT EPOCHS OF SIGNAL AFTER STIMULI ##################

# face IDs based on eperimental setup in OpenViBE
def is_face_angry(face_id):
    labels = {1, 4, 7, 10, 13, 16, 20, 22}
    for i in labels:
        if face_id == 33024 + i:
            return True
    return False

def is_face_happy(face_id):
    labels = {3, 5, 8, 12, 14, 18, 21, 23}
    for i in labels:
        if face_id == 33024 + i:
            return True
    return False

def is_face_emotional(face_id):
    if is_face_angry(face_id) or is_face_happy(face_id):
        return True
    return False

# my own method, there could be something more universal for finding peak
def forward_diff(signal, order):
    new_signal = np.zeros_like(signal)
    for n in range(len(signal) - order):
        new_signal[n] = np.sum(np.diff(signal[n:(n + order)]))

    return new_signal

# Init container for ERP epochs
extracted_epochs_emo = OrderedDict()
extracted_epochs_neutral = OrderedDict()
wrong_range = OrderedDict()
wrong_peak = OrderedDict()
for e in electrodes_to_analyze:
    extracted_epochs_emo[e] = list()
    extracted_epochs_neutral[e] = list()
    wrong_range[e] = 0
    wrong_peak[e] = 0

# Just accumulators for statistics
wrong_response_emo = 0
wrong_response_neutral = 0
all_responses = 0
answer_time_emo = 0
answer_time_emos = []
resp_emo = 0
answer_time_neutral = 0
resp_neutral = 0
answer_time_neutrals = []

for filename, record in database.items():

    # Compute forward difference of triggering electrode signal and find its minima
    raw_trigger_signal = np.array(record['signals'][triggering_electrode])
    trigger_signal = forward_diff(raw_trigger_signal, slope_width)
    trigger_threshold = (np.mean(np.sort(trigger_signal)[:len(record['responses'])]) + np.mean(np.sort(trigger_signal))) / 2

    # Compare raw triggering signal and its derivation
    # plt.figure()
    # plt.plot(range(len(trigger_signal)), raw_trigger_signal, 'b-')
    # plt.plot(range(len(trigger_signal)), trigger_signal, 'g-', linewidth=1)
    # plt.axhline(trigger_threshold, color='k', linestyle='dashed')
    # plt.xlabel('Time [s]')
    # plt.ylabel('uV')
    # plt.grid()
    # plt.show()

    # Find next stimuli start and save related epoch for every electrode
    i = 0
    trigger_iter = 0
    true_timestamps = list()
    openvibe_timestamps = list()
    while i < len(trigger_signal):
        if trigger_signal[i] < trigger_threshold:

            if all_responses >= 5120:
                break
            all_responses += 1

            try:
                was_response_correct = record['responses'][trigger_iter][0]
            except:
                all_responses -= 1
                i += 1
                continue

            face_id = record['order'][trigger_iter][0]

            event_timestamp = record['responses'][trigger_iter][1]
            resp_time = record['order'][trigger_iter][1]
            answer_time = event_timestamp - resp_time

            if is_face_emotional(face_id):
                answer_time_emos.append(answer_time)
                answer_time_emo += answer_time
                resp_emo += 1
            else:
                answer_time_neutrals.append(answer_time)
                answer_time_neutral += answer_time
                resp_neutral += 1

            if was_response_correct:
                # Find stimuli index
                margin = 100
                search_area_start = max(0, i - max_trigger_peak_width // 2)
                search_area_end = min(i + max_trigger_peak_width // 2, len(trigger_signal))
                stimuli_index = int(search_area_start + np.argmin(trigger_signal[search_area_start:search_area_end]))
                if stimuli_index < margin:
                    i += max_trigger_peak_width
                    trigger_iter += 1
                    all_responses -= 1
                    continue

                # Plot trigger synchronization timestamps
                # plt.figure()
                # plt.plot(range(len(trigger_signal[stimuli_index-margin:stimuli_index+margin])), raw_trigger_signal[stimuli_index-margin:stimuli_index+margin], 'g-', linewidth=3)
                # plt.plot(range(len(trigger_signal[stimuli_index-margin:stimuli_index+margin])), trigger_signal[stimuli_index-margin:stimuli_index+margin], 'b-', linewidth=3)
                # plt.axvline(margin, color='k', linestyle='dashed')
                # plt.xlabel('Time [s]')
                # plt.ylabel('uV')
                # #plt.grid()
                # plt.show()

                try:
                    true_timestamps.append(record['timestamps'][stimuli_index])
                    openvibe_timestamps.append(record['order'][trigger_iter][1])
                except:
                    pass

                # Save epoch
                for electrode, signal in record['signals'].items():
                    if stimuli_index - pre_stimuli < 0 or stimuli_index + post_stimuli > len(signal):
                        all_responses -= 1
                        break
                    if electrode in electrodes_to_analyze:
                        epoch = signal[stimuli_index - pre_stimuli:stimuli_index + post_stimuli]
                        epoch_max = np.max(epoch)
                        epoch_min = np.min(epoch)
                        epoch_peak_to_peak = epoch_max - epoch_min

                        if peak_filtering:
                            def inv_ric(points, a):
                                return -scipy.signal.ricker(points, a)

                            widths = 0.5 * np.arange(1, 11)
                            cwtmatr = scipy.signal.cwt(epoch, inv_ric, widths)
                            peak_score = np.mean(cwtmatr[:, pre_stimuli + time2sample(0.17)])

                            # plt.figure()
                            # plt.title('Peak score: ' + str(peak_score))
                            # plt.imshow(cwtmatr, extent=[-100, 500, epoch.min(), epoch.max()], cmap='coolwarm', aspect='auto',
                            #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
                            # plt.plot(
                            #     np.multiply(np.arange(len(epoch)) - pre_stimuli, 1000 / fs),
                            #     epoch, linewidth=2)
                            # plt.axvline(165, color='k', linestyle='dashed')
                            # plt.xlim([-100, 500])
                            # plt.show()
                        else:
                            peak_score = min_peak_score + 1

                        if trigger_iter < 0:
                            # Draw and save single epochs
                            plt.figure()
                            plt.title(electrode + (', peak score: ' + str(peak_score)) if peak_filtering else '')
                            plt.plot(((np.array(range(len(epoch)))) - pre_stimuli) / fs, epoch, 'g-', linewidth=1)
                            plt.axvline(0, color='k', linestyle='dashed')
                            plt.axvline(0.17, color='b', linestyle='dashed')
                            plt.xlabel('Time [s]')
                            plt.ylabel('uV')
                            plt.grid()
                            plt.savefig(os.path.join(figures_dir, basename(filename), 'epochs', electrode+'_'+str(trigger_iter)+ ('_common' if common_avg_ref else '_org') +'.png'))
                            plt.close()

                        # Collect correct epochs
                        if epoch_peak_to_peak <= epoch_max_peak_to_peak and peak_score >= min_peak_score:
                            try:
                                if len(record['order']) > 0 and is_face_emotional(face_id):
                                    extracted_epochs_emo[electrode].append(epoch)
                                else:
                                    extracted_epochs_neutral[electrode].append(epoch)
                            except:
                                pass

                        if epoch_peak_to_peak > epoch_max_peak_to_peak:
                            wrong_range[electrode] += 1
                        elif peak_score < min_peak_score:
                            wrong_peak[electrode] += 1
            else:
                face_id = record['order'][trigger_iter][0]
                if is_face_emotional(face_id):
                    wrong_response_emo += 1
                else:
                    wrong_response_neutral +=1

            i += max_trigger_peak_width
            trigger_iter += 1
        else:
            i += 1

    # Calculate difference between true timestamps and openvibe timestamps - jitter and drift analysis
    # diff = np.subtract(true_timestamps[:len(openvibe_timestamps)], openvibe_timestamps)
    # m = np.mean(diff)
    # v = np.std(diff)
    print(filename)

# Print stats
print(wrong_response_emo, wrong_response_neutral, all_responses, 100 * wrong_response_emo / all_responses, 100 * wrong_response_neutral / all_responses, 100 * (wrong_response_emo+wrong_response_neutral) / all_responses)
print(wrong_range)
print(wrong_peak)
print(1000*answer_time_emo/resp_emo, 1000*answer_time_neutral/resp_neutral, resp_emo, resp_neutral)

######### AVERAGE EPOCHS ##################
invert_y_axis = False

# N170 area 140-185ms
n170_begin = pre_stimuli + time2sample(0.10)
n170_end = pre_stimuli + time2sample(0.20) + 1

epn_begin = pre_stimuli + time2sample(0.25)
epn_end = pre_stimuli + time2sample(0.35) + 1

lats_emo = []
lats_neutral = []

n170_amps_emo = []
n170_amps_neutral = []

epn_amps_emo = []
epn_amps_neutral = []


for electrode in electrodes_to_analyze:

    #TODO: use dictionary
    epochs_emo = extracted_epochs_emo[electrode]
    epochs_neutral = extracted_epochs_neutral[electrode]

    num_emo = len(epochs_emo)
    num_neutral = len(epochs_neutral)
    print(num_emo, num_neutral)

    if num_emo == 0 or num_neutral == 0:
        continue

    # Grand-average over all epochs
    averaged_emo = np.mean(epochs_emo, axis=0)
    averaged_neutral = np.mean(epochs_neutral, axis=0)

    # change voltage scale as difference from baseline
    averaged_emo -= np.mean(averaged_emo[:pre_stimuli + 1])
    averaged_neutral -= np.mean(averaged_neutral[:pre_stimuli + 1])

    # Fill csv logs for statistical analysis
    # ANOVAs calculated using: http://vassarstats.net/anova1u.html
    for i in range(min(num_emo, num_neutral)):
        amp_emo = (np.argmin(epochs_emo[i][n170_begin:n170_end]) + n170_begin)
        amp_neutral = (np.argmin(epochs_neutral[i][n170_begin:n170_end]) + n170_begin)
        lats_emo.append(amp_emo)
        lats_neutral.append(amp_neutral)
        csv_writers['latencies'].writerow([amp_emo, amp_neutral])

    csv_writers['n170_amplitudes'].writerow(['c']*20)
    csv_writers['n170_amplitudes'].writerow([electrode])
    for i in range(min(num_emo, num_neutral)):
        amp_emo = np.min(epochs_emo[i][n170_begin:n170_end])
        amp_neutral = np.min(epochs_neutral[i][n170_begin:n170_end])
        n170_amps_emo.append(amp_emo)
        n170_amps_neutral.append(amp_neutral)
        csv_writers['n170_amplitudes'].writerow([amp_emo, amp_neutral])

    csv_writers['epn_amplitudes'].writerow(['c'] * 20)
    csv_writers['epn_amplitudes'].writerow([electrode])
    for i in range(min(num_emo, num_neutral)):
        amp_emo = np.mean(epochs_emo[i][epn_begin:epn_end])
        amp_neutral = np.mean(epochs_neutral[i][epn_begin:epn_end])
        epn_amps_emo.append(amp_emo)
        epn_amps_neutral.append(amp_neutral)
        csv_writers['epn_amplitudes'].writerow([amp_emo, amp_neutral])

    # Draw and save ERP plots
    plt.figure()
    plt.title(electrode)
    neutral_plot, = plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), averaged_neutral, color='#505050', linewidth=2)
    emotion_plot, = plt.plot(np.multiply(np.arange(len(averaged_emo)) - pre_stimuli, 1000 / fs), averaged_emo,
                             color='r', linewidth=2, linestyle='dashed')
    plt.axvline(0, color='k', linestyle='dashed')
    plt.axhline(0, color='k', linestyle='dashed')
    plt.xlabel('Time [ms]')
    plt.ylabel('uV')
    #plt.ylim([-8, 10.5])
    #plt.yticks(np.arange(-8.0, 10.1, 2))
    plt.legend(handles=[emotion_plot,neutral_plot],
               labels=['Emotional', 'Neutral'], fontsize=12)
    plt.axvspan(0.25, 0.35, facecolor='#F0F0F0', edgecolor='#F0F0F0', alpha=0.5)
    if invert_y_axis:
        plt.gca().invert_yaxis()

    figure_file_all = os.path.join(figures_dir, 'all', electrode + ('_common_' if common_avg_ref else '_org_') + str(int(filter_on * low_cutoff)) + '_' + str(int(filter_on * high_cutoff)) + 'Hz_' + suffix)
    plt.savefig(figure_file_all + '.png')
    #plt.savefig(figure_file_all + '.eps')
    #plt.show()
