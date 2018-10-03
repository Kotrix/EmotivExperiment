import numpy as np


def add_gaussian(signal: np.ndarray):
    mean = 0
    var = 10
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma, signal.shape)
    noisy = signal + gauss
    return noisy


def add_salt_and_pepper(signal: np.ndarray):
      s_vs_p = 0.5
      amount = 0.0004
      out = np.copy(signal)
      # Salt mode
      num_salt = np.ceil(amount * signal.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in signal.shape]
      out[coords] = 6000

      # Pepper mode
      num_pepper = np.ceil(amount* signal.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in signal.shape]
      out[coords] = 0
      return out


def add_poisson(signal: np.ndarray):
      vals = len(np.unique(signal))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(signal * vals) / float(vals)
      return noisy

def add_speckle(signal: np.ndarray):
      row,col = signal.shape
      gauss = np.random.randn(row,col) * 0.005
      noisy = signal + signal * gauss
      return noisy

def snr(data: np.ndarray):
    axis = 0
    data = np.transpose(data)

    # positive_data = (data - np.min(data, axis=axis))
    # normalized_data = positive_data/np.max(positive_data, axis=axis)
    normalized_data = data

    m = np.max(normalized_data, axis=axis) - np.min(normalized_data, axis=axis)
    sd = np.std(normalized_data, axis=axis, ddof=0)

    cos = np.transpose(normalized_data)
    mean =  cos.mean(axis=1)
    lala = np.transpose(cos) - mean
    v = np.transpose(lala)
    a = (np.sum((lala) ** 2, axis=axis))
    b = normalized_data[0, :].size - 1
    c = a/b

    std = np.sqrt(c)
    return m/sd