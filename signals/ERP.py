from data_utils import *
import RWA as rwa
import GW6 as gw6
import scipy

############# CONFIG ###########################
# person_id = 1
for person_id in range(1,11):
    database_regex = 'csv/'+ str(person_id if person_id > 0 else '*') + '/record-F*.csv'
    suffix = 'CLASSIC'

    triggering_electrode = 'F7'
    electrodes_to_analyze = ['P7','O1','O2','P8']

    common_avg_ref = True
    ref_electrodes = ['AF3','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','AF4'] if common_avg_ref else []

    all_electrodes = np.unique([triggering_electrode] + electrodes_to_analyze + ref_electrodes)

    # Filtering options
    filter_on = True
    low_cutoff = 0.5
    high_cutoff = 15 # desired cutoff frequency of the filter, Hz (0 to disable filtering)
    trigger_high_cutoff = 15

    # Define pre and post stimuli period of epoch (in seconds)
    pre_stimuli = time2sample(0.1)
    post_stimuli = time2sample(0.5)
    # pre_stimuli = time2sample(1)
    # post_stimuli = time2sample(1)

    # Define threshold for trigger signal
    max_trigger_peak_width = time2sample(2) # in seconds

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

    epoch_logs = ['OK', 'No_peak', 'Max_peak']
    epoch_writers = dict()
    for name in epoch_logs:
        for e in electrodes_to_analyze:
            csvfile = open(os.path.join('epochs', name, e + '.csv'), 'w', newline='')
            epoch_writers[name + e] = csv.writer(csvfile, delimiter=',', quotechar='|')

    ######### READ SIGNALS FROM DISK ############################

    database = read_openvibe_csv_database(database_regex, all_electrodes)
    #plot_database(database, 1)

    if common_avg_ref:
        ######## COMMON AVERAGE REFERENCE ###########################
        common_database = database.copy()
        for filename, record in database.items():

            # Find average signal over all ref_electrodes
            mean = np.zeros(len(record['signals'][triggering_electrode]))
            count = 0
            for electrode, signal in record['signals'].items():
                if electrode in ref_electrodes:
                    mean += signal
                    count += 1
            mean /= count

            for e in electrodes_to_analyze:
                common_database[filename]['signals'][e] -= mean

        database = common_database


    #plot_database(database, 1)



    ########### SIGNALS FILTERING ###############
    from filtering import *

    org_database = database.copy()

    # Modify database by filtering signals
    for filename, record in database.items():
        for electrode, signal in record['signals'].items():
            if electrode == triggering_electrode:
                temp_high = trigger_high_cutoff
            else:
                temp_high = high_cutoff

            if filter_on or electrode == triggering_electrode:
                filtered_signal = butter_bandpass_filter(signal, [low_cutoff, temp_high], fs)
                database[filename]['signals'][electrode] = filtered_signal
    #plot_database(database, 1)


    ############ CUT THE END OF SIGNAL TO REMOVE FILTERING ARTIFACTS ##################
    # Times (in seconds) to cut signals in the beginning and end
    left_cut = time2sample(1)
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

    averaged_data = OrderedDict()

    for filename, record in database.items():
        averaged_data[filename] = list()

        data = np.ndarray((4, len(record['signals']['P7'])))
        i = 0
        for electrode in electrodes_to_analyze:
            data[i,:] = record['signals'][electrode]
            i += 1

        # RWA absolute
        # averaged_data[filename] = rwa.robust_weighted_averaging_absolute(data)
        averaged_data[filename] = rwa.robust_weighted_averaging_quadratic(data)
        # averaged_data[filename] = np.mean((data), axis=0)



    # Init container for ERP epochs
    extracted_epochs_emo = list()
    extracted_epochs_neutral = list()
    wrong_range = OrderedDict()
    wrong_peak = OrderedDict()
    for e in electrodes_to_analyze:
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
        trigger_signal = np.gradient(raw_trigger_signal)
        trigger_threshold = (2*np.median(np.sort(raw_trigger_signal)[:len(record['responses'])]) +
            np.median(raw_trigger_signal)) / 3

        # Compare raw triggering signal and its derivation
        # plt.figure()
        # plt.plot(np.array(range(len(trigger_signal))) / fs, raw_trigger_signal, 'b-')
        # #plt.plot(np.array(range(len(trigger_signal))) / fs, trigger_signal, 'g-', linewidth=1)
        # plt.axhline(trigger_threshold, color='k', linestyle='dashed')
        # plt.xlabel('Latency (ms)', fontsize=12)
        # plt.ylabel('Amplitude ($\mu$V)', fontsize=12)
        # plt.grid()
        # plt.show()

        # Find next stimuli start and save related epoch for every electrode
        i = 0
        trigger_iter = 0
        true_timestamps = list()
        openvibe_timestamps = list()
        while i < len(trigger_signal):
            if np.all(raw_trigger_signal[i:i+5] < trigger_threshold):

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
                    margin = max_trigger_peak_width // 2
                    search_area_start = max(0, i - margin)
                    search_area_end = min(i + margin + 1, len(trigger_signal))
                    stimuli_index = int(search_area_start + np.argmin(trigger_signal[search_area_start:search_area_end])) - 4
                    if stimuli_index < margin:
                        i += max_trigger_peak_width
                        trigger_iter += 1
                        all_responses -= 1
                        continue

                    try:
                        true_timestamps.append(record['timestamps'][stimuli_index])
                        openvibe_timestamps.append(record['order'][trigger_iter][1])
                    except:
                        pass

                    # Save epoch
                    for filename in averaged_data:
                        if stimuli_index - pre_stimuli < 0 or stimuli_index + post_stimuli > len(averaged_data[filename]):
                            all_responses -= 1
                            break

                        epoch = averaged_data[filename][stimuli_index - pre_stimuli:stimuli_index + post_stimuli + 1]
                        epoch_max = np.max(epoch)
                        epoch_min = np.min(epoch)
                        epoch_peak_to_peak = epoch_max - epoch_min

                        if epoch_peak_to_peak > epoch_max_peak_to_peak:
                            wrong_range[electrode] += 1
                            continue

                        if peak_filtering:
                            def inv_ric(points, a):
                                return -scipy.signal.ricker(points, a)


                            widths = 0.5 * np.arange(2, 11)
                            cwtmatr = scipy.signal.cwt(epoch, inv_ric, widths)
                            peak_point = pre_stimuli + time2sample(0.16)
                            peak_score = np.mean(cwtmatr[:, peak_point - 1:peak_point + 4])

                            # if peak_score < min_peak_score:
                            #
                            #     plt.figure()
                            #     plt.title('Condition: ' + str(peak_score), fontsize=14)
                            #     plt.imshow(cwtmatr, extent=[-100, 500, epoch.min(), epoch.max()], cmap='gray', aspect='auto',
                            #                vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), alpha=0.7)
                            #     plt.xticks(fontsize=14)
                            #     plt.yticks(fontsize=14)
                            #     plt.xlabel('Latency (ms)', fontsize=12)
                            #     plt.ylabel('Amplitude ($\mu$V)', fontsize=12)
                            #     plt.ylim([epoch.min(), epoch.max()])
                            #     #plt.twinx()
                            #     plt.plot(
                            #         np.multiply(np.arange(len(epoch)) - pre_stimuli, 1000 / fs),
                            #         epoch, 'k-', linewidth=3, linestyle='dashed')
                            #     plt.axvline(1000*sample2time(peak_point-1-pre_stimuli), color='k', linestyle='dashed', linewidth=2)
                            #     plt.axvline(1000*sample2time(peak_point+3-pre_stimuli), color='k', linestyle='dashed', linewidth=2)
                            #     #plt.xticks(fontsize=14)
                            #     #plt.yticks(epoch.min() + (np.arange(9)) * (epoch.max() - epoch.min()) / 9 + (epoch.max() - epoch.min()) / 18, widths, fontsize=14)
                            #     #plt.ylabel('Wavelet width', fontsize=12)
                            #     plt.xlim([-100, 500])
                            #     plt.ylim([epoch.min(), epoch.max()])
                            #     plt.tight_layout()
                            #     plt.show()
                        else:
                            peak_score = min_peak_score + 1

                        if peak_score < min_peak_score:
                            wrong_peak[electrode] += 1
                            continue

                        # Collect correct epochs
                        try:
                            if len(record['order']) > 0 and is_face_emotional(face_id):
                                extracted_epochs_emo.append(epoch)
                            else:
                                extracted_epochs_neutral.append(epoch)
                        except:
                            pass
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

    epn_begin = pre_stimuli + time2sample(0.24)
    epn_end = pre_stimuli + time2sample(0.34) + 1

    lats_emo = []
    lats_neutral = []

    n170_amps_emo = []
    n170_amps_neutral = []

    epn_amps_emo = []
    epn_amps_neutral = []

    # # GW6
    # averaged_emo, averaged_emo_by_channel = gw6.GW6(extracted_epochs_to_ndarray(extracted_epochs_emo))
    # averaged_neutral, averaged_neutral_by_channel = gw6.GW6(extracted_epochs_to_ndarray(extracted_epochs_neutral))
    #
    # # for channel in range(len(electrodes_to_analyze)):
    # # Draw and save ERP plots
    # plt.figure(figsize=(8.5, 5.5))
    # plt.title('GW6', fontsize=12)
    # plt.axvspan(240, 340, facecolor='#E0E0E0', edgecolor='#E0E0E0', alpha=0.5)
    # neutral_plot, = plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), averaged_neutral,
    #                          color='#505050', linewidth=2)
    # emotion_plot, = plt.plot(np.multiply(np.arange(len(averaged_emo)) - pre_stimuli, 1000 / fs), averaged_emo,
    #                          color='k', linewidth=2, linestyle='dashed')
    # plt.axvline(0, color='k', linewidth=1)
    # plt.text(270, -15, 'EPN', fontsize=16)
    # plt.axhline(0, color='k', linewidth=1)
    # plt.xlabel('Latency (ms)', fontsize=12)
    # plt.ylabel('Amplitude ($\mu$V)', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # # plt.xlim([-100, 500])
    # # plt.ylim([-16.5, 10])
    # # plt.yticks(np.arange(-15.0, 10.1, 5))
    # plt.tight_layout()
    # plt.legend(handles=[neutral_plot, emotion_plot],
    #            labels=['Neutral', 'Emotional'], fontsize=12)
    # if invert_y_axis:
    #     plt.gca().invert_yaxis()
    #
    # figure_file_all = os.path.join(figures_dir, 'all',str(person_id) + '_' + 'all' + ('_common_' if common_avg_ref else '_org_') + str(int(filter_on * low_cutoff)) + '_' + str(int(filter_on * high_cutoff)) + 'Hz_' +  suffix)
    # plt.savefig(figure_file_all + '.png')
    # plt.close()


    #TODO: use dictionary
    epochs_emo = extracted_epochs_emo
    epochs_neutral = extracted_epochs_neutral

    num_emo = len(epochs_emo)
    num_neutral = len(epochs_neutral)
    print(num_emo, num_neutral)

    if num_emo == 0 or num_neutral == 0:
        continue

    # # Grand-average over all epochs
    # Classic method
    averaged_emo = np.mean(epochs_emo, axis=0)
    averaged_neutral = np.mean(epochs_neutral, axis=0)

    # # RWA absolute
    # averaged_emo = rwa.robust_weighted_averaging_absolute(epochs_emo)
    # averaged_neutral = rwa.robust_weighted_averaging_absolute(epochs_neutral)
    #
    # # RWA quadratic
    # averaged_emo = rwa.robust_weighted_averaging_quadratic(epochs_emo)
    # averaged_neutral = rwa.robust_weighted_averaging_quadratic(epochs_neutral)

    # # change voltage scale as difference from baseline
    # averaged_emo -= np.mean(averaged_emo[:pre_stimuli + 1])
    # averaged_neutral -= np.mean(averaged_neutral[:pre_stimuli + 1])

    # plt.figure(figsize=(6.5, 5.5))
    # plt.title(electrode, fontsize=12)
    # acc_mean = []
    # for i, epoch in enumerate(np.array(extracted_epochs_emo[electrode])[[1,3,10,11,13,17,20,24]]):
    #     acc_mean.append(epoch)
    #     #plt.plot((epoch - np.mean(epoch))/np.std(epoch), linewidth=1, color='k', alpha=0.7)
    #     plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), epoch, linewidth=1, color='k', linestyle='dashed')
    #
    # # for i, epoch in enumerate(np.array(extracted_epochs_neutral[electrode])[[1,3,10,11,13,17,20,24]]):
    # #     acc_mean.append(epoch)
    # #     #plt.plot((epoch - np.mean(epoch))/np.std(epoch), linewidth=1, color='k', alpha=0.7)
    # #     plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), epoch, linewidth=1, color='k')
    #
    # acc_mean = np.mean(acc_mean, axis=0)
    # plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), acc_mean, linewidth=3, color='k')
    #
    # plt.xlabel('Latency (ms)', fontsize=12)
    # plt.ylabel('Amplitude ($\mu$V)', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlim([-100, 500])
    # plt.ylim([-21, 12])
    # plt.yticks(np.arange(-20.0, 15.1, 5))
    # plt.tight_layout()
    # plt.show()


    # Draw and save ERP plots
    plt.figure(figsize=(8.5, 5.5))
    plt.title(electrode, fontsize=12)
    plt.axvspan(240, 340, facecolor='#E0E0E0', edgecolor='#E0E0E0', alpha=0.5)
    neutral_plot, = plt.plot(np.multiply(np.arange(len(averaged_neutral)) - pre_stimuli, 1000 / fs), averaged_neutral, color='#505050', linewidth=2)
    emotion_plot, = plt.plot(np.multiply(np.arange(len(averaged_emo)) - pre_stimuli, 1000 / fs), averaged_emo,
                             color='k', linewidth=2, linestyle='dashed')
    plt.axvline(0, color='k', linewidth=1)
    plt.text(270, -15, 'EPN', fontsize=16)
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Amplitude ($\mu$V)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim([-100, 500])
    # plt.ylim([-16.5, 10])
    plt.yticks(np.arange(-15.0, 10.1, 5))
    plt.tight_layout()
    plt.legend(handles=[neutral_plot, emotion_plot],
               labels=['Neutral', 'Emotional'], fontsize=12)
    if invert_y_axis:
        plt.gca().invert_yaxis()

    figure_file_all = os.path.join(figures_dir, 'all', str(person_id) + '_' + electrode + ('_common_' if common_avg_ref else '_org_') +
                                   str(int(filter_on * low_cutoff)) + '_' + str(int(filter_on * high_cutoff)) + 'Hz_' + suffix)
    plt.savefig(figure_file_all + '.png')
    #plt.show()
    plt.close()
