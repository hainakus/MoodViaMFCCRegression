import numpy as np
import math
import os

from librosa import(
    feature,
    logamplitude,
    feature,
    load
    )

def find_a_v_mens_va(ids, val, aro):
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

    X_v = np.linalg.lstsq(x, y_v)[0]
    X_a = np.linalg.lstsq(x, y_a)[0]

    return X_v, X_a


def average_distance_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))

    return sum/float(len(ids))


def average_distance_to_nearest_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from nearest value and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        min_dist = 9 # no distance biger than this
        for j in range(len(valence[ids[i]])):
            dist = math.sqrt(math.pow(valence_calc[i] - valence[ids[i]][j], 2) + math.pow(arousal_calc[i] - arousal[ids[i]][j], 2))
            min_dist = dist if dist < min_dist else min_dist
        sum += min_dist

    return sum/float(len(ids))


def average_distance_std_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        stdev = (np.std(valence[ids[i]]) + np.std(arousal[ids[i]])) / 2
        adist = math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))
        sum += adist/stdev

    return sum/float(len(ids))


def valence_distance_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += abs(valence_calc[i] - np.mean(valence[ids[i]]))
    return sum/float(len(ids))


def arousal_distance_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += abs(arousal_calc[i] - np.mean(arousal[ids[i]]))

    return sum/float(len(ids))


def nearest_dist_average_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    claculates average distance to nearest song
    '''
    sum = 0
    for i in range(len(ids)):
        min_dist = 9  # this is value greater than all distances
        for j in range(len(valence[ids[i]])):
            dist = math.sqrt(math.pow(valence_calc[i] - valence[ids[i]][j], 2) + math.pow(arousal_calc[i] - arousal[ids[i]][j], 2))
            min_dist = dist if min_dist > dist else min_dist
        sum += min_dist

    return sum/float(len(ids))


def no_stdev_average_va(valence_calc, arousal_calc, valence_mean, arousal_mean, valence, arousal, ids):
    '''
    calculates aveerage factor claculated deistance / no_stdev_average
    '''
    sum = 0
    for i in range(len(ids)):
#        distance = math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))
        stdev = (np.std(valence[ids[i]]) + np.std(arousal[ids[i]])) / 2
        sum += math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))/stdev

    return sum/float(len(ids))


def find_in_dict(dict, ids):

    results = np.array([])
    for idx in ids:
        if len(results) == 0:
            results = (np.array(dict[idx])).ravel()
        else:
            results = np.vstack((results, np.array(dict[idx])))

    return results

def mfcc(path):
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    # We use a small hop length of 64 here so that the
    # frames line up with the beat tracker example below.

    y, sr = load_files(path)

    print 'claculating mfcc ' + path
    S = feature.melspectrogram(y, sr=sr, n_fft=2048, hop_length=64, n_mels=128)
    
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = logamplitude(S, ref_power=np.max)
    mfcc_v = feature.mfcc(S=log_S, n_mfcc=14)
    
    return np.sum(mfcc_v, axis=1)/mfcc_v.shape[1]


def load_files(path):
    # print 'reading audio'
    y, sr = load(path, sr=44100)

    return y, sr

def calc_mfcc_features(path):
    '''
    function calcuates features for all song in songs
    returns mean for each song and songs id
    '''
    print 'calcualting features for ' + path

    ids = []
    fetures = {}
    i = 0

    for filename in os.listdir(path):
        key = int(filename[:3])
        ids.append(key)
        mfcc_feat = mfcc(os.path.join(path, filename))
        # print fetures
        fetures[key] = mfcc_feat
        print i
        i += 1

    return ids, fetures


def feature_matrix_by_id(idx, feature):
    '''
    function from dict feature select rows
    and concat 
    it in new matrix using idx
    '''

    feats = np.array(feature[idx[0]])

    for id in idx[1:]:
        feats = np.vstack((feats, feature[id]))

    return feats

from random import shuffle


def shufle_same(X, Yv, Ya, ids):
    X_shuf = []
    Yv_shuf = []
    Ya_shuf = []
    ids_shuf = []
    index_shuf = range(len(Yv))
    shuffle(index_shuf)
    for i in index_shuf:
        # print i
        X_shuf.append(X[i])
        Yv_shuf.append(Yv[i])
        Ya_shuf.append(Ya[i])
        ids_shuf.append(ids[i])
        # print i

    # print X.shape
    # print len(Ya)

    return X, Yv, Ya, ids_shuf


def averagedist(xa, ya, xb, yb):
    '''
    calculate mean of distances
    beetwen arrays of point
    '''
    sum = 0
    for i in range(len(xa)):
        a = np.array([xa[i], ya[i]])
        b = np.array([xb[i], yb[i]])
        sum += np.linalg.norm(a-b)
    return sum/len(xa)


