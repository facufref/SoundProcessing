from SoundFileManager import *
import numpy as np
import pandas as pd
import unittest


class SoundProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_target_should_return_valid_target(self):
        df = pd.read_csv('wavfiles/Test/labels_SoundFileManagerTest.csv')
        target = get_target(df)
        self.assertEqual(4, len(target))
        self.assertEqual(1, target[0])
        self.assertEqual(1, target[1])
        self.assertEqual(0, target[2])
        self.assertEqual(0, target[3])

    def test_get_data_and_filenames_should_return_data_and_filenames(self):
        df = pd.read_csv('wavfiles/Test/labels_SoundFileManagerTest.csv')
        data, filenames = get_data_and_filenames(df, 'wavfiles/Test/')
        self.assertEqual((4, 12), data.shape)
        self.assertEqual(4, len(filenames))
        self.assertEqual('Violin1.wav', filenames[0])
        self.assertEqual('Violin2.wav', filenames[1])
        self.assertEqual('Monster1.wav', filenames[2])
        self.assertEqual('Monster2.wav', filenames[3])
