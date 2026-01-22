import numpy as np

def snr(signal, noise):
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
