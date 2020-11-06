import pytest

import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal)

from daltonize.daltonize import gamma_correction, inverse_gamma_correction

def test_gamma_correction():
    rgb = np.array([[[0, 10, 11, 25, 128, 255]]]).reshape((-1, 1, 3))
    expected = np.array([[[0.      , 0.003035, 0.003347]],
                         [[0.00972 , 0.2158  , 1.      ]]], dtype=np.float16)
    linear_rgb = gamma_correction(rgb)
    assert_array_almost_equal(linear_rgb, expected)
    expected = np.array([[[0.      , 0.003035, 0.093   ]],
                         [[0.145   , 0.528   , 1.      ]]], dtype=np.float16)
    linear_rgb = gamma_correction(rgb, gamma=1)
    assert_array_almost_equal(linear_rgb, expected)

def test_inverse_gamma_correction():
    rgb = np.array([[[0, 10, 11, 25, 128, 255]]]).reshape((-1, 1, 3))
    assert_array_almost_equal(rgb, inverse_gamma_correction(gamma_correction(rgb)))

