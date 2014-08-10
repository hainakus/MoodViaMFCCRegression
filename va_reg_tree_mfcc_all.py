from feature_extract.features import(
    calc_chroma_features,
    calc_chroma_features_dict,
    calc_mfcc_features_dict
)
from utils.read import(
    mean_va,
    csv_2_dict_va
)

from utils.calc_utils import(
    find_a_v_mens_va,
    regression,
    average_distance_va,
    nearest_dist_average_va,
    no_stdev_average_va,
    find_in_dict,
    valence_distance_va,
    arousal_distance_va,
    feature_matrix_by_id,
    shufle_same,
    averagedist
)
from utils.plot import(
    plot_all_va
)

import random
import sys

import numpy as np

from sklearn.tree import DecisionTreeRegressor

''' 
regression tree for each response
'''


# y, sr = load_files('audio/101.mp3')
# mfcc_v = mfcc(y, sr)
# get exsisting valence and arousal data
all_ids, all_val, all_aro = mean_va('csv/survery2dataMin1.csv')
valence, arousal = csv_2_dict_va('csv/survery2dataMin1.csv')

print len(all_ids)

# calculate fetures for song in train set
ids, feat = calc_mfcc_features_dict('audio/full')

X = feature_matrix_by_id(all_ids, feat)
Yv = all_val
Ya = all_aro
# print X

best_avg = sys.maxint
best_near = sys.maxint
best_std = sys.maxint

best_val = sys.maxint
best_aro = sys.maxint

print "check 3"
print X.shape
print len(Yv)
print len(Ya)

for i in range(50):
    X, Yv, Ya, all_ids = shufle_same(X, Yv, Ya, all_ids)
    

    print "check 4"
    print X.shape
    print len(Yv)
    print len(Ya)

    trainlen = int(len(Yv) * 0.7)

    Xtrain = X[1:trainlen]
    Yvtrain = Yv[1:trainlen]
    Yatrain = Ya[1:trainlen]

    Xtest = X[trainlen+1:]
    Yvtest = Yv[trainlen+1:]
    Yatest = Ya[trainlen+1:]
    idstest = all_ids[trainlen+1:]

    print "check 5"
    print X.shape
    print len(Yv)
    print len(Ya)

    clf_1 = DecisionTreeRegressor(max_depth=2)
    clf_2 = DecisionTreeRegressor(max_depth=2)
    clf_1.fit(Xtrain, Yvtrain)
    clf_2.fit(Xtrain, Yatrain)

    Yvpred = clf_1.predict(Xtest)
    Yapred = clf_2.predict(Xtest)
    # print len(Yvpred)
    # print len(Yvtest)
    # print Yvpred.shape
    # print Yvtest.shape
    # avg = averagedist(Yvpred, Yapred, Yvtest, Yatest)
    avg = average_distance_va(Yvpred, Yapred, valence, arousal, idstest)

    if avg < best_avg:
        best_avg = avg

print "BEST"
print 'Average distance: ' + str(best_avg)
# print 'Nearest distance: ' + str(best_near)
# print 'Std distance: ' + str(best_std)

# print best_val
# print best_aro
