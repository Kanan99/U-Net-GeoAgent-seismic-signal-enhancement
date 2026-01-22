
import numpy as np

def normalize(trace):
    m = np.mean(trace)
    s = np.std(trace) + 1e-8
    return (trace - m) / s

def clip_amplitude(trace, limit=3.0):
    return np.clip(trace, -limit, limit)
