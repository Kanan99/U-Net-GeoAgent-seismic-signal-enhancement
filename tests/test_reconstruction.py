import numpy as np
from src.core.reconstruction import overlap_average

def test_overlap():
    patch = np.ones((4,4))
    out = overlap_average([patch], [(0,0)], (4,4), 4)
    assert out.shape == (4,4)
