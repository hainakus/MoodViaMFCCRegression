import os
from librosa import(
    feature,
    logamplitude,
    feature,
    load
    )
import csv
import numpy as np
import matplotlib.pyplot as plt


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

    return np.sum(mfcc_v, axis=1)


def calc_features(path):
    '''
    function calcuates features for all song in songs
    returns mean for each song and songs id
    '''
    print 'reading features for ' + path

    ids = []
    fetures = np.array([])

    with open('eric_dataset/mfccs', 'r') as f:
        for line in f:
            linesp = line.split()
            ids.append(int(linesp[0][:-4]))
            mfccs = []

            for word in linesp[1:]:
                mfccs.append(float(word))

            if len(fetures) == 0:
                fetures = (np.array(mfccs)).ravel()
            else:
                fetures = np.vstack((fetures, np.array(mfccs)))

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

    # print x

    # A = np.vstack([x]).T

    # print x

    X_v = np.linalg.lstsq(x, y_v)[0]
    X_a = np.linalg.lstsq(x, y_a)[0]

    return X_v, X_a


def show_for_id(song_id, valence, arousal, ids, valence_dict, arousal_dict):
    idx = ids.index(song_id)
    plt.clf()
    plt.plot(valence_dict[song_id], arousal_dict[song_id], 'o', color='green', markersize=1)
    plt.plot(valence[idx], arousal[idx], 'o', color='blue')
    plt.plot(sum(valence_dict[song_id])/float(len(valence_dict[song_id])), sum(arousal_dict[song_id])/float(len(arousal_dict[song_id])), 'o', color='red')
    plt.axis([-1, 1, -1, 1])
    plt.savefig('results/' + str(song_id) + '.png')


def plot_all(all_val, all_aro, all_ids, valence, arousal):
    for id_s in all_ids:
        show_for_id(id_s, all_val, all_aro, all_ids, valence, arousal)


# y, sr = load_files('audio/101.mp3')
# mfcc_v = mfcc(y, sr)
# get exsisting valence and arousal data
valence, arousal = csv_2_dict('csv/survery2data.csv')

# calculate fetures for song in train set
train_ids, train_feat = calc_features('audio/train')

train_ids = train_ids[0::4]
train_feat = train_feat[0::4]
print len(train_ids)
print train_feat.shape

exit()
# calcultae valence and arousal find_a_v_mens
val_mean, aro_mean = find_a_v_mens(train_ids, valence, arousal)

# use regression
X_v, X_a = regression(train_feat, val_mean, aro_mean)

# calculating features for whole dataset
all_ids, all_feat = calc_features('audio/full')
#print all_feat.shape

# use linera function to calculate v and a
all_val = np.sum(np.array(all_feat) * X_v, axis=1)
all_aro = np.sum(all_feat * X_a, axis=1)

#print all_val.shape
#print all_aro.shape

plot_all(all_val, all_aro, all_ids, valence, arousal)
