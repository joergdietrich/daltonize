import pytest

from pathlib import Path

from PIL import Image

import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal)

from daltonize.daltonize import gamma_correction, inverse_gamma_correction, simulate, array_to_img

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

@pytest.mark.parametrize("type, ref_img_path", [("d", Path("daltonize/tests/data/colored_crayons_d.png")),
                                                ("p", Path("daltonize/tests/data/colored_crayons_p.png")),
                                                ("t", Path("daltonize/tests/data/colored_crayons_t.png"))])
def test_simulation(type, ref_img_path):
    gamma = 2.4
    orig_img_path = Path("daltonize/tests/data/colored_crayons.png")
    orig_img = np.asarray(Image.open(orig_img_path).convert("RGB"), dtype=np.float16)
    orig_img = gamma_correction(orig_img, gamma)
    simul_rgb = simulate(orig_img, type)
    simul_img = np.asarray(array_to_img(simul_rgb, gamma=gamma))
    ref_img = np.asarray(Image.open(ref_img_path).convert("RGB"))
    assert_array_almost_equal(simul_img, ref_img)