import os
from librosa import(
    feature,
    logamplitude,
    feature,
    load
    )

import numpy as np



def load_files(path):
    # print 'reading audio'
    y, sr = load(path, sr=44100)

    return y, sr


def mfcc(path):
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    # We use a small hop length of 64 here so that the
    # frames line up with the beat tracker example below.

    y, sr = load_files(path)

    print 'claculating mfcc ' + path
    S = feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=64, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = logamplitude(S, ref_power=np.max)

    mfcc_v = feature.mfcc(S=log_S, n_mfcc=20)

    return np.sum(mfcc_v, axis=1)/mfcc_v.shape[1]


def chromagram(path):
    y, sr = load_files(path)
    C = feature.chromagram(y, sr)
    return np.mean(C, axis=1)


def calc_mfcc_features(path):
    '''
    function calcuates features for all song in songs
    returns mean for each song and songs id
    '''
    print 'calcualting features for ' + path

    ids = []
    fetures = np.array([])
    i = 0

    for filename in os.listdir(path):
        ids.append(int(filename[:3]))
        mfcc_feat = mfcc(os.path.join(path, filename))
        # print fetures
        if len(fetures) == 0:
            fetures = mfcc_feat.ravel()
        else:
            fetures = np.vstack((fetures, mfcc_feat.ravel()))
        print i
        i += 1

    return ids, fetures


def calc_chroma_features(path):
    '''
    function calcuates shape features for all song in path
    returns mean for each song and songs id
    '''
    print 'calcualting shape features for ' + path

    ids = []
    fetures = np.array([])
    i = 0

    for filename in os.listdir(path):
        ids.append(int(filename[:3]))
        mfcc_feat = chromagram(os.path.join(path, filename))
        # print fetures
        if len(fetures) == 0:
            fetures = mfcc_feat.ravel()
        else:
            fetures = np.vstack((fetures, mfcc_feat.ravel()))
        print i
        i += 1

    return ids, fetures