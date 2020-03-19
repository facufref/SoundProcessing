from SoundProcessor import *
import numpy as np
import unittest


class SoundProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

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

    def test_frame_signal_should_return_valid_frames_and_length(self):
        emphasized_signal = [0.0, 0.0, 0.0, -1.0, 1.0, 3.95, -2.95, -5.85, 4.9, 5.85]
        sample_rate = 100
        frame_length, frames = frame_signal(emphasized_signal, sample_rate)
        self.assertEqual(frame_length, 2)
        self.assertEqual(frames[0][0], 0.0)
        self.assertEqual(frames[0][1], 0.0)
        self.assertEqual(frames[1][0], 0.0)
        self.assertEqual(frames[1][1], 0.0)
        self.assertEqual(frames[2][0], 0.0)
        self.assertEqual(frames[2][1], -1.0)
        self.assertEqual(frames[3][0], -1.0)
        self.assertEqual(frames[3][1], 1.0)
        self.assertEqual(frames[4][0], 1.0)
        self.assertEqual(frames[4][1], 3.95)
        self.assertEqual(frames[5][0], 3.95)
        self.assertEqual(frames[5][1], -2.95)
        self.assertEqual(frames[6][0], -2.95)
        self.assertEqual(frames[6][1], -5.85)
        self.assertEqual(frames[7][0], -5.85)
        self.assertEqual(frames[7][1], 4.9)

    def test_apply_hamming_window_should_return_valid_frames(self):
        frame_length = 2
        frames = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0], [-1.0, 1.0], [1.0, 3.95], [3.95, -2.95], [-2.95, -5.85], [-5.85, 4.9]])
        frames = apply_hamming_window(frame_length, frames)
        self.assertEqual(round(frames[0][0], 3), 0.0)
        self.assertEqual(round(frames[0][1], 3), 0.0)
        self.assertEqual(round(frames[1][0], 3), 0.0)
        self.assertEqual(round(frames[1][1], 3), 0.0)
        self.assertEqual(round(frames[2][0], 3), 0.0)
        self.assertEqual(round(frames[2][1], 3), -0.08)
        self.assertEqual(round(frames[3][0], 3), -0.08)
        self.assertEqual(round(frames[3][1], 3), 0.08)
        self.assertEqual(round(frames[4][0], 3), 0.08)
        self.assertEqual(round(frames[4][1], 3), 0.316)
        self.assertEqual(round(frames[5][0], 3), 0.316)
        self.assertEqual(round(frames[5][1], 3), -0.236)
        self.assertEqual(round(frames[6][0], 3), -0.236)
        self.assertEqual(round(frames[6][1], 3), -0.468)
        self.assertEqual(round(frames[7][0], 3), -0.468)
        self.assertEqual(round(frames[7][1], 3), 0.392)

    def test_apply_stft_should_return_valid_frames(self):
        frames = np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        nfft, pow_frames = apply_stft(frames, 4)
        self.assertEqual(pow_frames.shape, (5, 3))
        self.assertEqual(round(pow_frames[0][0], 5), 0.00250)
        self.assertEqual(round(pow_frames[1][0], 5), 0.02250)
        self.assertEqual(round(pow_frames[2][0], 5), 0.06250)
        self.assertEqual(round(pow_frames[3][0], 5), 0.12250)
        self.assertEqual(round(pow_frames[4][0], 5), 0.20250)
        self.assertEqual(round(pow_frames[0][2], 5), 0.00250)
        self.assertEqual(round(pow_frames[4][2], 5), 0.00250)

    def test_apply_filter_banks_should_return_valid_filter_banks(self):
        frames = np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        nfft, pow_frames = apply_stft(frames, 4)
        filter_banks = apply_filter_banks(nfft, pow_frames, 100, 4)
        self.assertEqual(filter_banks.shape, (5, 4))
        self.assertEqual(round(filter_banks[0][0], 5), -313.07120)
        self.assertEqual(round(filter_banks[0][1], 5), -52.04120)
        self.assertEqual(round(filter_banks[0][2], 5), -313.07120)
        self.assertEqual(round(filter_banks[0][3], 5), -52.04120)
        self.assertEqual(round(filter_banks[4][0], 5), -313.07120)
        self.assertEqual(round(filter_banks[4][1], 5), -13.87150)
        self.assertEqual(round(filter_banks[4][2], 5), -313.07120)
        self.assertEqual(round(filter_banks[4][3], 5), -19.78552)

    def test_apply_sinusoidal_liftering_should_return_valid_mfcc(self):
        mfcc = np.array([[1.0, 2.0], [3.0, 4.0]])
        mfcc = apply_sinusoidal_liftering(mfcc, 10)
        self.assertEqual(mfcc.shape, (2, 2))
        self.assertEqual(round(mfcc[0][0], 5), 1.00000)
        self.assertEqual(round(mfcc[0][1], 5), 5.09017)
        self.assertEqual(round(mfcc[1][0], 5), 3.00000)
        self.assertEqual(round(mfcc[1][1], 5), 10.18034)

    def test_apply_mfcc_should_return_valid_mfcc(self):
        frames = np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        nfft, pow_frames = apply_stft(frames, 4)
        filter_banks = apply_filter_banks(nfft, pow_frames, 100, 4)
        mfcc = apply_mfcc(filter_banks)
        self.assertEqual(mfcc.shape, (5, 3))

    def test_mean_normalize_should_return_normalized_frames(self):
        mfcc = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean_normalize(mfcc)
        self.assertEqual(mfcc.shape, (2, 2))
        self.assertEqual(round(mfcc[0][0], 5), -1.00000)
        self.assertEqual(round(mfcc[0][1], 5), -1.00000)
        self.assertEqual(round(mfcc[1][0], 5), 1.00000)
        self.assertEqual(round(mfcc[1][1], 5), 1.00000)





