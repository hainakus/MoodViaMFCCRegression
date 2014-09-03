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
    uniform_va_rgb,
    csv_2_dict_rgb,
    uniform_va_rgb_polar
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
ids, va, aro, r, g, b= uniform_va_rgb_polar('csv/surveydatahsv.csv')
r_dict, g_dict, b_dict = csv_2_dict_rgb('csv/surveydatahsv.csv')

# calculate fetures for song in train set

X = np.column_stack((va,aro))
Yr = r
Yg = g
Yb = b

best_avg = sys.maxint
best_r = sys.maxint
best_g = sys.maxint
best_b = sys.maxint

for i in range(100):
    X, Yr, Yg, Yb, ids = shufle_same(X, Yr, Yg, Yb, ids)   
    
    trainlen = int(len(Yr) * 0.7)

    Xtrain = X[1:trainlen]
    Yrtrain = Yr[1:trainlen]
    Ygtrain = Yg[1:trainlen]
    Ybtrain = Yb[1:trainlen]

    Xtest = X[trainlen+1:]
    Yrtest = Yr[trainlen+1:]
    Ygtest = Yg[trainlen+1:]
    Ybtest = Yb[trainlen+1:]
    idstest = ids[trainlen+1:]
    
    rr = regression_one(Xtrain, Yrtrain)
    rg = regression_one(Xtrain, Ygtrain)
    rb = regression_one(Xtrain, Ybtrain)
    Yrpred = np.sum(Xtest * rr, axis=1)
    Ygpred = np.sum(Xtest * rg, axis=1)
    Ybpred = np.sum(Xtest * rb, axis=1)

  
    r_avg = average_vector_dist(Yrpred, r_dict, idstest)
    g_avg = average_vector_dist(Ygpred, g_dict, idstest)
    b_avg = average_vector_dist(Ybpred, b_dict, idstest)

    avg = (r_avg + g_avg + b_avg)/3
    if (avg < best_avg):
        best_avg = avg
        best_r = r_avg
        best_g = g_avg
        best_b = b_avg



print "BEST"
print "r: " + str(best_r)
print "g: " + str(best_g)
print "b: " + str(best_b)

