import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

def fft_bandpass_filter(signal, lower_limit=0, upper_limit=50, fs=128):
    assert(lower_limit < upper_limit)
    f_signal = rfft(signal)

    W = fftfreq(len(signal), d=1 / fs)
    cut_f_signal = f_signal.copy()
    cut_f_signal[W < lower_limit] = 0
    cut_f_signal[W > upper_limit] = 0

    # plt.figure()
    # plt.plot(W,f_signal)

    return irfft(cut_f_signal)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.divide(cutoff, nyq)
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_filter(data, cutoff, fs, order=5):
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_freq_response(b, a, fs, cutoff):
    
    w, h = freqz(b, a, worN=800)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.show()