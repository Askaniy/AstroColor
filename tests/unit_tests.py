import unittest
import numpy as np
#import matplotlib.pyplot as plt
from astrocolor.auxiliary import linear_interp, spectral_binning

np.random.seed(42)


class TestAuxiliary(unittest.TestCase):

    def test_linear_interpolation(self):
        x0 = np.array([3, 5, 8, 9, 11])
        y0 = np.array([3, 4, 7, 0, 2])
        x1 = np.arange(0, 15, 0.5)
        y1_constant = linear_interp(x0, y0, x1, extrap_mode='constant')
        y1_linear = linear_interp(x0, y0, x1, extrap_mode='linear')
        #plt.plot(x1, y1_constant)
        #plt.plot(x1, y1_linear)
        #plt.scatter(x0, y0)
        #plt.savefig('tests/test_linear_interpolation.png')
        np.testing.assert_allclose(y1_constant, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 3.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], rtol=1e-5)
        np.testing.assert_allclose(y1_linear, [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 3.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], rtol=1e-5)

    def test_spectral_binning(self):
        nm0_len = 100
        nm0 = np.sort(np.linspace(402, 650, nm0_len) + np.random.normal(0, 5, nm0_len))
        br0 = nm0 / 100
        step = 5 # nm
        nm1 = np.arange(400, 700, step)
        nm0_diff = np.diff(nm0)
        br1, std1 = spectral_binning(nm0, br0, None, nm1, step, nm0_diff)
        np.testing.assert_allclose(br1, nm1 / 100, rtol=0.1)


if __name__ == '__main__':
    unittest.main()
