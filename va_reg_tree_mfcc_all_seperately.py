from feature_extract.features import(
    calc_chroma_features,
    calc_chroma_features_dict,
    calc_mfcc_features_dict
)
from utils.read import(
    mean_va,
    csv_2_dict_va,
    read_csv_col,
    read_csv_col_with_row,
    read_feature_from_json,
    seperate_va
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
    average_distance_std_va,
    average_distance_to_nearest_va
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
all_ids, all_val, all_aro, rows = seperate_va('csv/survery2dataMin1.csv')
valence, arousal = csv_2_dict_va('csv/survery2dataMin1.csv')

#data by columns
    # 3 - age
    # 4 - sex
    # 5 - place of living
    # 6 - music school
    # 7 - medicines
    # 8 - drugs
    # 12 - amount of listening music
    # 13 - playing instrument or singing
    # 17\18 - mood in VA
    # 19\20 - mood color
    # 21:40 - preception of 10 labels
    # 41:57 - presence of mood
    # 58:77 - color perception for 10 lables
    # 81 - song id
    # 82:101 - induced labels positions
    # 102:129 - perceived labels positions
    # 130/131 - color for song
    # ni 134:136 - HSV for song

# calculate fetures for song in train set
#ids, feat = calc_mfcc_features_dict('audio/full')
feat = read_feature_from_json('features/mfcc_our_dataset_20.json')

X = feature_matrix_by_id(all_ids, feat)
Yv = all_val
Ya = all_aro
# add data to X

#hsv = np.array(read_csv_col('csv/survery2dataMin1.csv', 134, 136))
color = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 130, 131, rows))
vamood = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 17, 18, rows))
musicschool = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 6, 6, rows))
sex = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 4, 4, rows))
listening = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 12, 12, rows))
moodcolor = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 19, 20, rows))
moodperception = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 21, 40, rows))
presencemood = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 41, 57, rows))
colorperception = np.array(read_csv_col_with_row('csv/survery2dataMin1.csv', 58, 77, rows))
# X = np.hstack((X, moodperception))
X = np.hstack((X, colorperception))

print "check 3"
print X.shape
print len(Yv)
print len(Ya)

best_avg = sys.maxint
best_near = sys.maxint
best_std = sys.maxint

best_val = sys.maxint
best_aro = sys.maxint

for j in range(1, 2):
    for i in range(100):
        X, Yv, Ya, all_ids = shufle_same(X, Yv, Ya, all_ids)

        trainlen = int(len(Yv) * 0.7)

        Xtrain = X[1:trainlen]
        Yvtrain = Yv[1:trainlen]
        Yatrain = Ya[1:trainlen]

        Xtest = X[trainlen+1:]
        Yvtest = Yv[trainlen+1:]
        Yatest = Ya[trainlen+1:]
        idstest = all_ids[trainlen+1:]

        # print "check 5"
        # print X.shape
        # print len(Yv)
        # print len(Ya)

        clf_1 = DecisionTreeRegressor(max_depth=j)
        clf_2 = DecisionTreeRegressor(max_depth=j)
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
        std = average_distance_std_va(Yvpred, Yapred, valence, arousal, idstest)
        near = average_distance_to_nearest_va(Yvpred, Yapred, valence, arousal, idstest)

        if avg < best_avg:
            best_avg = avg
            best_near = near
            best_std = std

    print 'j ' + str(j)
    #print "BEST"
    print 'Average distance: ' + str(best_avg)
    # print 'Nearest distance: ' + str(best_near)
    # print 'Std distance: ' + str(best_std)

    # print best_val
    # print best_aro
print 'BEST'
print 'Average distance: ' + str(best_avg)
print 'Std distance: ' + str(best_std)
print 'Neares distance: ' + str(best_near)
