import numpy as np
import mne


class RobustWeightedAveraging:
    def quadratic(self, data: np.ndarray):
        # [V,W] = AVE1(data,V,Vt)
        # Weighted averaging - quadratic loss function.
        # data <- data set cycles in rows,
        # V <- averaged signal (row vector),
        # W <- weight vector,

        number_of_channels, signal_length = np.shape(data)  # n1-cycles, n2-samples in cycle,
        exponent_param = 2.0  # exponent parameter
        i = 100  # number of iterations
        epsilon = 1e-6  # criterion decreasing
        averaged_signal = np.zeros((1, signal_length))
        weights = np.zeros((1, number_of_channels))

        for i in range(i):  # main loop
            weights_old = weights

            DD = np.sum(np.square(data - averaged_signal).transpose(), axis=0)
            w_new = np.power(DD, 1 / (1 - exponent_param))  # update W
            weights = w_new / np.sum(w_new)

            weights_exp = np.power(weights, exponent_param)
            averaged_signal = weights_exp.dot(data) / np.sum(weights_exp)
            if i > 0:
                if np.std(weights - weights_old) < epsilon:
                    break

        print('No of iteration: {}'.format(i + 1))
        return self._to_raw(weights.dot(data)/sum(weights))

    def absolute(self, data: np.ndarray):
        # [V,W] = AVE1(data,V,Vt)
        # Weighted averaging - quadratic loss function.
        # data <- data set cycles in rows,
        # V <- averaged signal (row vector),
        # W <- weight vector,

        number_of_channels, signal_length = np.shape(data)  # n1-cycles, n2-samples in cycle,
        exponent_param = 2.0  # exponent parameter
        i = 100  # number of iterations
        epsilon = 1e-8  # criterion decreasing
        averaged_signal = np.zeros((1, signal_length))
        weights = np.zeros((1, number_of_channels))

        for i in range(i):  # main loop
            weights_old = weights

            DD = np.sum(np.abs(data - averaged_signal).transpose(), axis=0)
            w_new = np.power(DD, 1 / (1 - exponent_param))  # update W
            weights = w_new / np.sum(w_new)

            weights_exp = np.power(weights, exponent_param)
            weights_exp1 = weights_exp / DD
            averaged_signal = weights_exp1.dot(data) / np.sum(weights_exp1)
            if i > 0:
                if np.std(weights - weights_old) < epsilon:
                    break

        print('No of iteration: {}'.format(i + 1))
        return self._to_raw(weights.dot(data) / sum(weights))

    def _to_raw(self, data: np.ndarray):
        info = mne.create_info(
            ch_names=['RWA'],
            ch_types=['eeg'],
            sfreq=128)

        data = data.reshape((1, len(data)))

        raw = mne.io.RawArray(data, info)
        raw.pick_types(meg=False, eeg=True, eog=False)
        return raw
