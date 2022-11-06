import unittest
import numpy as np
import main


# testing class - Task 1.2
class TestMethods(unittest.TestCase):
    image = np.array([[1, 2, 2, 2], [0, 1, 4, 3],
                      [0, 2, 2, 2], [0, 1, 1, 0]])

    kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

    # test convolution with zero padding border
    def test_convolution_zeros(self):
        expected_convoluted_image = np.array([[3., 6., 7., 1.], [2., 5., 2., -6.],
                                              [3., 3., -4., -9.],
                                              [1., -1., -5., -5.]])
        np.testing.assert_array_equal(expected_convoluted_image, main.convolve2d(self.image, self.kernel, "zeros"))

    # test convolution with mirror padding border
    def test_convolution_reflect(self):
        expected_convoluted_image = np.array([[0., 5., 2., 0.],
                                              [-1., 5., 2., 0.],
                                              [0., 3., -4., -6.],
                                              [0., 3., -1., 0.]])
        np.testing.assert_array_equal(expected_convoluted_image, main.convolve2d(self.image, self.kernel, "reflect"))



if __name__ == '__main__':
    unittest.main()
