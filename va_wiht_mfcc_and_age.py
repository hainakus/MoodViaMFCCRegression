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
from utils.calc_utils import(
    find_in_dict,
    calc_mfcc_features,
    find_a_v_mens_va,
    find_in_dict, 
    regression,
    valence_distance_va,
    arousal_distance_va,
    average_distance_va,
    nearest_dist_average_va,
    no_stdev_average_va,
) 
from utils.read import(
    csv_2_dict_va
)
import sys


# y, sr = load_files('audio/101.mp3')
# mfcc_v = mfcc(y, sr)
# get exsisting valence and arousal data
valence, arousal = csv_2_dict_va('csv/survery2dataMin1.csv')

ids, feat = calc_mfcc_features_years('audio/full')
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
    val_mean, aro_mean = find_a_v_mens_va(train_ids, valence, arousal)
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
    avg = average_distance_va(all_val, all_aro, valence, arousal, all_ids)
    nearest = nearest_dist_average_va(all_val, all_aro, valence, arousal, all_ids)
    standdev = no_stdev_average_va(all_val, all_aro, val_mean, aro_mean, valence, arousal, all_ids)

    valence_dist = valence_distance_va(all_val, all_aro, valence, arousal, all_ids)
    arousal_dist = arousal_distance_va(all_val, all_aro, valence, arousal, all_ids)

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

#plot_all_va(all_val, all_aro, all_ids, valence, arousal)

print "BEST"
print 'Average distance: ' + str(best_avg)
print 'Nearest distance: ' + str(best_near)
print 'Std distance: ' + str(best_std)

print best_avg
print best_near
print best_std

print best_val
print best_aro
