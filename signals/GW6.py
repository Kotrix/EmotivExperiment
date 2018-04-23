import numpy as np

def GW6(data: np.ndarray):
    return calculate_EEG_correlation(calculate_pearson_correlation_combinations(data), data.shape[0])

def calculate_pearson_correlation_combinations(data: np.ndarray):
    # definition of the value of data
    number_of_channels, epochs_length, number_of_stimuli = np.shape(data)  # 14, 384 (duration 3s, frequency 128Hz), 100
    number_of_channels_combination = int((number_of_channels ** 2 - number_of_channels) / 2)  # in our case Nt=91 is the number of combination with 14 channels

    windows_length = 34  # 34 is a L window of about 270 ms, at 128 samples/s, could be changed
    half_windows_length = int(windows_length / 2)  # half window L

    data = np.pad(data, [(0,), (windows_length,), (0,)], mode='constant')  # addition tails
    data_length = epochs_length + 2 * windows_length  # 384 samples + 2 tails of 34 samples

    pearson_correlation_array = np.zeros((number_of_channels_combination, data_length))  # put zero each element of array R(I, X)

    I = 0
    for Ax in range(number_of_channels - 1):
        for Bx in range(Ax + 1, number_of_channels):
            A1 = np.zeros((data_length, number_of_stimuli))
            for U in range(half_windows_length, data_length - half_windows_length):
                x1 = U - half_windows_length
                x2 = U + half_windows_length

                vec1 = data[Ax, x1:x2, :]
                vec2 = data[Bx, x1:x2, :]

                # subroutine of  Pearson's  Correlation
                A1[U, :] = pearson_correlation(vec1, vec2)

            pearson_correlation_array[I, :] += np.sum(A1, axis=1)
            I = I + 1  # counter of the progressive combinations of two channels

    pearson_correlation_array /= number_of_stimuli
    return pearson_correlation_array[:, windows_length:data_length - windows_length]
    # now the array pearson_correlation_array is the output of this stage of elaboration

# Pearson’s correlation subroutine
def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray):
    vec1_avg = np.mean(vec1, axis=0)
    vec2_avg = np.mean(vec2, axis=0)

    f1 = np.average((vec1 - vec1_avg) * (vec2 - vec2_avg), axis=0)
    f2 = np.average((vec1 - vec1_avg) ** 2, axis=0)
    f3 = np.average((vec2 - vec2_avg) ** 2, axis=0)

    if f2.all() == 0 or f3.all() == 0:
        return np.zeros_like(f1)
    else:
        return 100 * f1 / np.sqrt(f2 * f3)  # the r of Pearson is multiplied by 100

def calculate_EEG_correlation(pearson_correlation_array: np.ndarray, number_of_channels: int):
    number_of_channels_combination, epochs_length = np.shape(pearson_correlation_array)
    B1 = 128
    B2 = B1 + 128
    # the stimulus zone is between X = B1 and X = B2
    # for the calculation of a balanced baseline, we take the pre-stimulus zone
    # (from X = 1 to B1), the second is the post-stimulus zone (from X = B2 to N1)

    # baseline for each combination
    Bs = np.sum(pearson_correlation_array[:, 0:B1], axis=1) +\
         np.sum(pearson_correlation_array[:, B2:epochs_length], axis=1)
    Bs /= (epochs_length + B1 - B2)

    pearson_correlation_array_absolute_value = (abs(pearson_correlation_array.transpose() - Bs)).transpose()
    Sync1 = np.mean(pearson_correlation_array_absolute_value, axis=0)
    # Now the array Sync1(X) is the average (global average) of Correlation for all the channels combinations and for all of the stimuli.

    # Calculation of the array Sync2 for each EEG channel
    Sync2 = np.zeros((number_of_channels, epochs_length))

    I = 0
    for Ax in range(number_of_channels - 1):
        for Bx in range(Ax + 1, number_of_channels):
            Sync2[Ax, :] += pearson_correlation_array_absolute_value[I, :]
            Sync2[Bx, :] += pearson_correlation_array_absolute_value[I, :]

            I = I + 1  # counter of all the combinations of the channels
    # Now the array Sync2 is the Correlation for each channel.

    Sync2 /= (number_of_channels - 1)
    # Each channel is the average of (NC-1) data.

    return Sync1, Sync2
