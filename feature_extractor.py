from feature_extract.features import calc_mfcc_features_dict
import json

'''
program calc mfcc feature for all song in dir audio/full
and write it to json file
'''

# read mfcc
ids, feat = calc_mfcc_features_dict('audio/full', 14)

for key in feat.keys():
    feat[key] = feat[key].tolist()


with open('features/mfcc_our_dataset_14.json', 'wb') as fp:
    json.dump(feat, fp)
