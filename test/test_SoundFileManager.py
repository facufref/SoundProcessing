from SoundDataManager import *
import pandas as pd
import unittest


class SoundProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_data_and_filenames_should_return_data_and_filenames(self):
        df = pd.read_csv('wavfiles/labels_SoundFileManagerTest.csv')
        data, target, filenames = get_data_target_filenames(df, 'wavfiles/')
        self.assertEqual((21, 12), data.shape)
        self.assertEqual(21, len(filenames))
        self.assertEqual(21, len(target))
        self.assertEqual('Violin1.wav', filenames[0])
        self.assertEqual('Violin1.wav', filenames[11])
        self.assertEqual('Violin2.wav', filenames[12])
        self.assertEqual('Violin2.wav', filenames[18])
        self.assertEqual('Monster1.wav', filenames[19])
        self.assertEqual('Monster1.wav', filenames[20])
        self.assertEqual([1], target[0])
        self.assertEqual([1], target[11])
        self.assertEqual([1], target[12])
        self.assertEqual([1], target[18])
        self.assertEqual([0], target[19])
        self.assertEqual([0], target[20])
