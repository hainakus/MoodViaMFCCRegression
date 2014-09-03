from feature_extract.features import(
    calc_chroma_features,
    calc_chroma_features_dict,
    calc_mfcc_features_dict
)
from utils.read import(
    mean_va,
    csv_2_dict_va,
    uniform_va_hsv,
    csv_2_dict_hsv,
    uniform_va_hsv_polar
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
    averagedist,
    regression_one,
    average_vector_dist
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


# get exsisting valence and arousal data
ids, va, aro, h, s, v = uniform_va_hsv_polar('csv/surveydatahsv.csv')
h_dict, s_dict, v_dict = csv_2_dict_hsv('csv/surveydatahsv.csv')

# calculate fetures for song in train set

X = np.column_stack((va,aro))
Yh = h
Ys = s
Yv = v

best_avg = sys.maxint
best_h = sys.maxint
best_s = sys.maxint
best_v = sys.maxint

for i in range(100):
    X, Yh, Ys, Yv, ids = shufle_same(X, Yh, Ys, Yv, ids)   
    
    trainlen = int(len(Yh) * 0.7)

    Xtrain = X[1:trainlen]
    Yhtrain = Yh[1:trainlen]
    Ystrain = Ys[1:trainlen]
    Yvtrain = Yv[1:trainlen]

    Xtest = X[trainlen+1:]
    Yhtest = Yh[trainlen+1:]
    Ystest = Ys[trainlen+1:]
    Yvtest = Yv[trainlen+1:]
    idstest = ids[trainlen+1:]


    rh = regression_one(Xtrain, Yhtrain)
    rs = regression_one(Xtrain, Ystrain)
    rv = regression_one(Xtrain, Yvtrain)
    Yhpred = np.sum(Xtest * rh, axis=1)
    Yspred = np.sum(Xtest * rs, axis=1)
    Yvpred = np.sum(Xtest * rv, axis=1)
  
    h_avg = average_vector_dist(Yhpred, h_dict, idstest)
    s_avg = average_vector_dist(Yspred, s_dict, idstest)
    v_avg = average_vector_dist(Yvpred, v_dict, idstest)

    avg = (h_avg + s_avg + v_avg)/3
    if (avg < best_avg):
        best_avg = avg
        best_h = h_avg
        best_s = s_avg
        best_v = v_avg


print "BEST"
print "h: " + str(best_h)
print "s: " + str(best_s)
print "v: " + str(best_v)

