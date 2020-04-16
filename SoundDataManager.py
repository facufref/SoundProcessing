from SoundProcessor import get_processed_mfcc, get_mean_frames
from scipy.io import wavfile
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def get_train_test(data, filenames, target):
    indices = np.arange(len(filenames))
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(data, target, indices, random_state=0, shuffle=True, train_size=0.75)
    return X_test, X_train, idx1, idx2, y_test, y_train


def get_dataset_from_wavfile(root, file_name):
    df = pd.read_csv(root + file_name)
    target = get_target(df)
    data, filenames = get_data_and_filenames(df, root)
    return data, target, filenames


def get_data_and_filenames(df, root):
    list_mfcc = []
    filenames = []
    df.set_index('fname', inplace=True)
    for f in df.index:
        filenames.append(f)
        file = wavfile.read(root + f)
        mfcc = get_processed_mfcc(file)
        mean_mfcc = get_mean_frames(mfcc)  # Mean of cepstral coefficients, maybe there's a better way to fit the data
        list_mfcc.append(mean_mfcc)
    data = np.vstack(list_mfcc)
    return data, filenames


def get_target(df):
    target = df.__getitem__('class')
    return target


def pre_process(X_test, X_train):
    """One way to pre process I found on "Introduction to machine learning with Python: a guide for data scientists." Chapter 2
        There must be a better way to do this"""
    # compute the mean value per feature on the training set
    mean_on_train = X_train.mean(axis=0)
    # compute the standard deviation of each feature on the training set
    std_on_train = X_train.std(axis=0)
    # subtract the mean, scale by inverse standard deviation
    # afterwards, mean=0 and std=1
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    # use THE SAME transformation (using training mean and std) on the test set
    X_test_scaled = (X_test - mean_on_train) / std_on_train
    return X_test_scaled, X_train_scaled

