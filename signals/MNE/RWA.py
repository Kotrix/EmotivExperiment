import numpy as np


class RobustWeightedAveraging:
    def quadratic(self, data: np.ndarray, input_signal_index):
        # [V,W] = AVE1(data,V,Vt)
        # Weighted averaging - quadratic loss function.
        # data <- data set cycles in rows,
        # V <- averaged signal (row vector),
        # W <- weight vector,

        number_of_epochs, number_of_channels, signal_length = np.shape(data)
        result = np.zeros((number_of_channels,signal_length))
        for channel_index in range(number_of_channels):
            channel_data = data[:, channel_index, :].reshape(number_of_epochs, signal_length)
            result[channel_index, :] = self._quadratic_for_one_channel(channel_data, number_of_epochs, input_signal_index)
        return result

    def _quadratic_for_one_channel(self, channel_data, number_of_epochs, input_signal_index):
        exponent_param = 2.0  # exponent parameter
        iter = 25  # number of iterations
        epsilon = 1e-3  # criterion decreasing
        averaged_signal = self._input_signal(channel_data, input_signal_index)
        weights = np.zeros(number_of_epochs)

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
        vec = weights.dot(channel_data) / sum(weights)
        if (i + 1 >= iter):
            return np.zeros(channel_data.shape[1]) # np.ones(channel_data.shape[1])*4200 #self._quadratic_for_one_channel(channel_data, signal_length, number_of_epochs) #
        else:
            return vec

    def absolute(self, data: np.ndarray, input_signal_index):
        # [V,W] = AVE1(data,V,Vt)
        # Weighted averaging - quadratic loss function.
        # data <- data set cycles in rows,
        # V <- averaged signal (row vector),
        # W <- weight vector,
        number_of_epochs, number_of_channels, signal_length = np.shape(data)
        result = np.zeros((number_of_channels, signal_length))

        for channel_index in range(number_of_channels):
            channel_data = data[:, channel_index, :].reshape(number_of_epochs,signal_length)
            result[channel_index, :] = self.absolute_for_one_channel(channel_data, number_of_epochs, input_signal_index)
        return result

    def absolute_for_one_channel(self, channel_data, number_of_epochs, input_signal_index):
        exponent_param = 2.0  # exponent parameter
        iter = 25  # number of iterations
        epsilon = 1e-3  # criterion decreasing
        averaged_signal = self._input_signal(channel_data, input_signal_index)
        weights = np.zeros(number_of_epochs)
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
            return np.zeros(channel_data.shape[1]) #
        else:
            return vec

    def _input_signal(self, vector_data: np.ndarray, index):
        if index == 0:
            return self._find_signal_with_minimal_distance_to_others(vector_data)
        elif index == 1:
            return np.reshape(np.median(vector_data, axis=0), (1, vector_data.shape[1]))
        elif index == 2:
            return np.reshape(np.mean(vector_data, axis=0), (1, vector_data.shape[1]))
        elif index == 3:
            return np.ones((1, vector_data.shape[1]))*4200
        elif index == 4:
            return np.zeros((1, vector_data.shape[1]))
        else:
            return np.random.uniform(np.random.rand(1)*-10000, np.random.rand(1)*10000, (1, vector_data.shape[1]))

    def _find_signal_with_minimal_distance_to_others(self, data: np.ndarray):
        signals_amount, signals_length = data.shape

        signals_distances = np.zeros(signals_amount)

        for i in range(signals_amount):
            for j in range(signals_amount):
                signals_distances[i] += self._dist(data[i, :], data[j, :])

        first_min, second_min = signals_distances.argsort()[:2]
        return np.mean(np.array([data[first_min], data[second_min]]), axis=0)

    def _dist(self, vec_a, vec_b):
        return np.sqrt(np.sum((vec_a-vec_b)**2))

