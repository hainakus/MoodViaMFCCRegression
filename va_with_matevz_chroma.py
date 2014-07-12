from feature_extract.features import(
	calc_chroma_features
)
from utils.read import(
	csv_2_dict_va,
	read_csv_song_features, 
	read_eric_va,
	read_fake_chroma
)

from utils.calc_utils import(
	find_a_v_mens_va,
	regression,
	average_distance_va,
	nearest_dist_average_va,
	no_stdev_average_va,
	find_in_dict

)
from utils.plot import(
	plot_all_va
)

import numpy as np
import random


# y, sr = load_files('audio/101.mp3')
# mfcc_v = mfcc(y, sr)
# get exsisting valence and arousal data
valence, arousal = csv_2_dict_va('csv/survery2dataMin1.csv')

# calculate fetures for song in train set
ids, feat = read_fake_chroma('features/fakechroma')
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

# use regresion function to calculate v and a
all_val = np.sum(np.array(test_feat) * X_v, axis=1)
all_aro = np.sum(test_feat * X_a, axis=1)

#print all_val.shape
#print all_aro.shape

#plot_all_va(all_val, all_aro, all_ids, valence, arousal)

print 'Average distance: ' + str(average_distance_va(all_val, all_aro, valence, arousal, all_ids)) 
print 'Nearest distance: ' + str(nearest_dist_average_va(all_val, all_aro, valence, arousal, all_ids)) 
print 'Std distance: ' + str(no_stdev_average_va(all_val, all_aro, val_mean, aro_mean, valence, arousal, all_ids)) 