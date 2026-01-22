import segyio
import numpy as np

def read_segy(path):
    with segyio.open(path, ignore_geometry=True) as f:
        data = np.stack([f.trace[i] for i in range(f.tracecount)], axis=1)
    return data
