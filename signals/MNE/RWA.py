import numpy as np
import mne


class RobustWeightedAveraging:
    def quadratic(self, data: np.ndarray):
        # [V,W] = AVE1(data,V,Vt)
        # Weighted averaging - quadratic loss function.
        # data <- data set cycles in rows,
        # V <- averaged signal (row vector),
        # W <- weight vector,

        number_of_epochs, number_of_channels, signal_length = np.shape(data)
        result = np.zeros((number_of_channels,signal_length))
        for channel_index in range(number_of_channels):
            channel_data = data[:, channel_index, :].reshape(number_of_epochs,signal_length)

            exponent_param = 2.0  # exponent parameter
            iter = 100  # number of iterations
            epsilon = 1e-6  # criterion decreasing
            averaged_signal = np.zeros((1, signal_length))
            weights = np.zeros((1, number_of_epochs))

            for i in range(iter):  # main loop
                weights_old = weights

                DD = np.sum(np.square(channel_data - averaged_signal).transpose(), axis=0)
                w_new = np.power(DD, 1 / (1 - exponent_param))  # update W
                weights = w_new / np.sum(w_new)

                weights_exp = np.power(weights, exponent_param)
                averaged_signal = weights_exp.dot(channel_data) / np.sum(weights_exp)
                if i > 0:
                    if np.std(weights - weights_old) < epsilon:
                        break

            print('No of iteration: {}'.format(i + 1))
            vec = weights.dot(channel_data)/sum(weights)
            if(i + 1 >= iter):
                result[channel_index, :] = np.zeros(signal_length)
            else:
                result[channel_index, :] = vec
        return result

    def absolute(self, data: np.ndarray):
        # [V,W] = AVE1(data,V,Vt)
        # Weighted averaging - quadratic loss function.
        # data <- data set cycles in rows,
        # V <- averaged signal (row vector),
        # W <- weight vector,
        number_of_epochs, number_of_channels, signal_length = np.shape(data)
        result = np.zeros((number_of_channels, signal_length))

        for channel_index in range(number_of_channels):
            channel_data = data[:, channel_index, :].reshape(number_of_epochs,signal_length)

            exponent_param = 2.0  # exponent parameter
            iter = 100  # number of iterations
            epsilon = 1e-8  # criterion decreasing
            averaged_signal = np.zeros((1, signal_length))
            weights = np.zeros((1, number_of_epochs))

            for i in range(iter):  # main loop
                weights_old = weights

                DD = np.sum(np.abs(channel_data - averaged_signal).transpose(), axis=0)
                w_new = np.power(DD, 1 / (1 - exponent_param))  # update W
                weights = w_new / np.sum(w_new)

                weights_exp = np.power(weights, exponent_param)
                weights_exp1 = weights_exp / DD
                averaged_signal = weights_exp1.dot(channel_data) / np.sum(weights_exp1)
                if i > 0:
                    if np.std(weights - weights_old) < epsilon:
                        break

            print('No of iteration: {}'.format(i + 1))
            vec = weights.dot(channel_data) / sum(weights)
            if (i + 1 >= iter):
                result[channel_index, :] = np.zeros(signal_length)
            else:
                result[channel_index, :] = vec
        return result
