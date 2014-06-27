import os
import librosa
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_files(path):
    # print 'reading audio'
    y, sr = librosa.load(path, sr=44100)
    return y, sr


def mfcc(path):
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    # We use a small hop length of 64 here so that the
    # frames line up with the beat tracker example below.

    y, sr = load_files(path)

    print 'claculating mfcc ' + path
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=64, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=S.max())

    mfcc_v = librosa.feature.mfcc(log_S, n_mfcc=1)

    return mfcc_v


def calc_features(path):
    '''
    function calcuates features for all song in songs
    returns mean for each song and songs id
    '''
    print 'calcualting features for ' + path

    ids = []
    fetures = []

    for filename in os.listdir(path):
        ids.append(int(filename[:3]))
        mfcc_feat = mfcc(os.path.join(path, filename))[0]
        fetures.append(sum(mfcc_feat) / float(len(mfcc_feat)))

    return ids, fetures


def csv_2_dict(path):
    '''
    Function creates 2 dictionaries for valence and arousal
    for each song in csv
    key - song id
    value - array of values
    '''

    print 'rendering csv file'

    # read csv
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    val = {}
    aro = {}

    # parse csv
    for row in reader:
        # print row[81]
        key = int(row[81])
        if key not in val:
            val[key] = []
            aro[key] = []

        for i in range(82, 129, 2):
            if float(row[i]) != 9:
                val[key].append(float(row[i]))
                aro[key].append(float(row[i+1]))

    return val, aro


def find_a_v_mens(ids, val, aro):
    '''
    calcultes means of arousal and valence for songs in ids
    returns v and a array of means
    '''
    val_m = []
    aro_m = []

    for id_s in ids:
        val_m.append(sum(val[id_s])/float(len(val[id_s])))
        aro_m.append(sum(aro[id_s])/float(len(aro[id_s])))

    return val_m, aro_m


def regression(features, valence_m, arousal_m):
    '''
    calculates c and n with regression for both
    '''
    print 'regression'
    x = np.array(features)
    y_v = np.array(valence_m)
    y_a = np.array(arousal_m)

    A = np.vstack([x, np.ones(len(x))]).T

    n_v, c_v = np.linalg.lstsq(A, y_v)[0]
    n_a, c_a = np.linalg.lstsq(A, y_a)[0]

    return n_v, c_v, n_a, c_a


def show_for_id(song_id, valence, arousal, ids, valence_dict, arousal_dict):
    idx = ids.index(song_id)
    plt.plot(valence_dict[song_id], arousal_dict[song_id], 'o', color='green', markersize=1)
    plt.plot(valence[idx], arousal[idx], 'o', color='blue')
    plt.plot(sum(valence_dict[song_id])/float(len(valence_dict[song_id])), sum(arousal_dict[song_id])/float(len(arousal_dict[song_id])), arousal[idx], 'o', color='red')
    plt.show()


# y, sr = load_files('audio/101.mp3')
# mfcc_v = mfcc(y, sr)
# get exsisting valence and arousal data
valence, arousal = csv_2_dict('csv/survery2data.csv')

# calculate fetures for song in train set
train_ids, train_feat = calc_features('audio/train')

# calcultae valence and arousal find_a_v_mens
val_mean, aro_mean = find_a_v_mens(train_ids, valence, arousal)

# use regression
n_v, c_v, n_a, c_a = regression(train_feat, val_mean, aro_mean)

# calculating features for whole dataset
all_ids, all_feat = calc_features('audio/full')

# use linera function to calculate v and a
all_val = np.array(all_feat) * n_v + c_v
all_aro = np.array(all_feat) * n_a + c_a

show_for_id(536, all_val, all_aro, all_ids, valence, arousal)
