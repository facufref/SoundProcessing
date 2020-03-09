from SoundProcessor import *
import numpy as np
import unittest


class SoundProcessorTest(unittest.TestCase):

    def test_pre_emphasize_signal_should_return_valid_emphasized_signal(self):
        signal = np.array([[0, 0], [0, -1], [1, 3], [-2, -3], [3, 3]])
        emphasized_signal = pre_emphasize_signal(signal, 0.95)
        self.assertEqual(10, len(emphasized_signal))
        self.assertEqual(0, emphasized_signal[0])
        self.assertEqual(0, emphasized_signal[1])
        self.assertEqual(signal[1][0] - 0.95 * signal[0][0], emphasized_signal[2])
        self.assertEqual(signal[1][1] - 0.95 * signal[0][1], emphasized_signal[3])
        self.assertEqual(signal[2][0] - 0.95 * signal[1][0], emphasized_signal[4])
        self.assertEqual(signal[2][1] - 0.95 * signal[1][1], emphasized_signal[5])
        self.assertEqual(signal[3][0] - 0.95 * signal[2][0], emphasized_signal[6])
        self.assertEqual(signal[3][1] - 0.95 * signal[2][1], emphasized_signal[7])
        self.assertEqual(signal[4][0] - 0.95 * signal[3][0], emphasized_signal[8])
        self.assertEqual(signal[4][1] - 0.95 * signal[3][1], emphasized_signal[9])
