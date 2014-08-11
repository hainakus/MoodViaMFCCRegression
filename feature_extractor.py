from feature_extract.features import calc_mfcc_features_dict
import json

'''
program calc mfcc feature for all song in dir audio/full
and write it to json file
'''

# read mfcc
ids, feat = calc_mfcc_features_dict('audio/full')

with open('features/mfcc_our_dataset_20.json', 'wb') as fp:
    json.dump(data, fp)