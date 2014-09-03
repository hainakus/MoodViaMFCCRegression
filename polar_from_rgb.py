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
    csv_2_dict_va_polar,
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
    average_vector_dist,
    average_vector_dist_polar
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
va_dict, aro_dict = csv_2_dict_va_polar('csv/survery2dataMin1.csv')

# calculate fetures for song in train set

X = np.column_stack((r,g,b))
Yva = va
Yaro = aro


best_avg = sys.maxint
best_aro = sys.maxint
best_va = sys.maxint

for i in range(100):
    X, Yva, Yaro, ids = shufle_same(X, Yva, Yaro, ids)   
    
    trainlen = int(len(Yaro) * 0.7)

    Xtrain = X[1:trainlen]
    Yvatrain = Yva[1:trainlen]
    Yarotrain = Yaro[1:trainlen]
    

    Xtest = X[trainlen+1:]
    Yvatest = Yva[trainlen+1:]
    Yarotest = Yaro[trainlen+1:]
    idstest = ids[trainlen+1:]
    #print Yarotest[0]

    rva = regression_one(Xtrain, Yvatrain)
    raro = regression_one(Xtrain, Yarotrain)
    
    Yvapred = np.sum(Xtest * rva, axis=1)
    Yaropred = np.sum(Xtest * raro, axis=1)
  
    va_avg = average_vector_dist(Yvapred, va_dict, idstest)
    aro_avg = average_vector_dist_polar(Yaropred, aro_dict, idstest)

    avg = (va_avg + aro_avg)/2
    if (avg < best_avg):
        best_avg = avg
        best_va = va_avg
        best_aro = aro_avg



print "BEST"
print "r: " + str(best_va)
print "theta: " + str(best_aro)

