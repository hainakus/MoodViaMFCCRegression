import os
from librosa import(
    feature,
    logamplitude,
    feature,
    load
    )
from utils.plot import plot_all_va
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from utils.calc_utils import find_in_dict
import sys


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

    mfcc_v = feature.mfcc(S=log_S, n_mfcc=14)
    
    return np.sum(mfcc_v, axis=1)/mfcc_v.shape[1]


def calc_features(path):
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

        for i in range(82+20, 129, 2):
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


def average_distance(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))
        if (math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))) > 0.8:
            print ids[i]


    return sum/float(len(ids))


def valence_distance(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += abs(valence_calc[i] - np.mean(valence[ids[i]]))
    return sum/float(len(ids))


def arousal_distance(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += abs(arousal_calc[i] - np.mean(arousal[ids[i]]))

    return sum/float(len(ids))


def nearest_dist_average(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    claculates average distance to nearest song
    '''
    sum=0
    for i in range(len(ids)):
        min_dist = 9 # this is value greater than all distances
        for j in range(len(valence[ids[i]])):
            dist = math.sqrt(math.pow(valence_calc[i] - valence[ids[i]][j], 2) + math.pow(arousal_calc[i] - arousal[ids[i]][j], 2))
            min_dist = dist if min_dist > dist else min_dist
        sum += min_dist

    return sum/float(len(ids))


def no_stdev_average(valence_calc, arousal_calc, valence_mean, arousal_mean, valence, arousal, ids):
    '''
    calculates aveerage factor claculated deistance / no_stdev_average
    '''
    sum = 0
    for i in range(len(ids)):
        distance = math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))
        stdev = (np.std(valence[ids[i]]) + np.std(arousal[ids[i]])) / 2
        sum += math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))/stdev

    return sum/float(len(ids))


# y, sr = load_files('audio/101.mp3')
# mfcc_v = mfcc(y, sr)
# get exsisting valence and arousal data
valence, arousal = csv_2_dict('csv/survery2dataMin1.csv')

ids, feat = calc_features('audio/full')
# repeate 50 times and take the best

best_avg = sys.maxint
best_near = sys.maxint
best_std = sys.maxint

best_val = sys.maxint 
best_aro = sys.maxint

for i in range(50):
    train_ids = ids
    random.shuffle(train_ids)
    all_ids = train_ids[141:]
    train_ids = train_ids[0:140]

    # calcultae valence and arousal find_a_v_mens
    val_mean, aro_mean = find_a_v_mens(train_ids, valence, arousal)
    train_feat = find_in_dict(feat, train_ids)
    test_feat = find_in_dict(feat, all_ids)

    # use regression
    X_v, X_a = regression(train_feat, val_mean, aro_mean)

    # calculating features for whole dataset
    #print all_feat.shape

    # use linera function to calculate v and a
    all_val = np.sum(np.array(test_feat) * X_v, axis=1)
    all_aro = np.sum(test_feat * X_a, axis=1)

    #print all_val.shape
    #print all_aro.shape

    #plot_all(all_val, all_aro, all_ids, valence, arousal)
    print "ATTEMPT" + str(i)
    avg = average_distance(all_val, all_aro, valence, arousal, all_ids)
    nearest = nearest_dist_average(all_val, all_aro, valence, arousal, all_ids)
    standdev = no_stdev_average(all_val, all_aro, val_mean, aro_mean, valence, arousal, all_ids)

    valence_dist = valence_distance(all_val, all_aro, valence, arousal, all_ids)
    arousal_dist = arousal_distance(all_val, all_aro, valence, arousal, all_ids)

    print 'Average distance: ' + str(avg) 
    print 'Nearest distance: ' + str(nearest) 
    print 'Nearest distance: ' + str(standdev) 

    if avg < best_avg:
        best_avg = avg
        best_near = nearest
        best_std = standdev

    if best_val > valence_dist:
        best_val = valence_dist

    if best_aro > arousal_dist:
        best_aro = arousal_dist

plot_all_va(all_val, all_aro, all_ids, valence, arousal)

print "BEST"
print 'Average distance: ' + str(best_avg)
print 'Nearest distance: ' + str(best_near)
print 'Std distance: ' + str(best_std)

print best_val
print best_aro
