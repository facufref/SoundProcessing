from SoundProcessor import get_processed_mfcc, get_mean_frames
from scipy.io import wavfile
import pandas as pd
import numpy as np


def get_dataset_from_wavfile(root, file_name):
    df = pd.read_csv(root + file_name)
    target = get_target(df)
    data = get_data(df, root)
    return data, target


def get_data(df, root):
    list_mfcc = []
    for f in df.index:
        file = wavfile.read(root + f)
        mfcc = get_processed_mfcc(file)
        mean_mfcc = get_mean_frames(mfcc)  # Mean of cepstral coefficients, maybe there's a better way to fit the data
        list_mfcc.append(mean_mfcc)
    data = np.vstack(list_mfcc)
    return data


def get_target(df):
    target = np.vstack(df.__getitem__('class'))
    df.set_index('fname', inplace=True)
    return target

#  TODO: Unit tests