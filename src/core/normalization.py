import numpy as np

def restore_amplitude(reference, enhanced, low=2, high=98):
    r_low, r_high = np.percentile(reference, [low, high])
    e_low, e_high = np.percentile(enhanced, [low, high])

    scaled = (enhanced - e_low) / (e_high - e_low + 1e-8)
    scaled = scaled * (r_high - r_low) + r_low

    return scaled + reference.mean() - scaled.mean()
