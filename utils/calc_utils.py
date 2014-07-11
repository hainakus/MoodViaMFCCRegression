import numpy as np
import math

def find_a_v_mens_va(ids, val, aro):
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


def average_distance_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    calculates average distace from mesured averages and calcualted averages
    '''
    sum = 0
    for i in range(len(ids)):
        sum += math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))

    return sum/float(len(ids))


def nearest_dist_average_va(valence_calc, arousal_calc, valence, arousal, ids):
    '''
    claculates average distance to nearest song
    '''
    sum = 0
    for i in range(len(ids)):
        min_dist = 9  # this is value greater than all distances
        for j in range(len(valence[ids[i]])):
            dist = math.sqrt(math.pow(valence_calc[i] - valence[ids[i]][j], 2) + math.pow(arousal_calc[i] - arousal[ids[i]][j], 2))
            min_dist = dist if min_dist > dist else min_dist
        sum += min_dist

    return sum/float(len(ids))


def no_stdev_average_va(valence_calc, arousal_calc, valence_mean, arousal_mean, valence, arousal, ids):
    '''
    calculates aveerage factor claculated deistance / no_stdev_average
    '''
    sum = 0
    for i in range(len(ids)):
#        distance = math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))
        stdev = (np.std(valence[ids[i]]) + np.std(arousal[ids[i]])) / 2
        sum += math.sqrt(math.pow(valence_calc[i] - np.mean(valence[ids[i]]), 2) + math.pow(arousal_calc[i] - np.mean(arousal[ids[i]]), 2))/stdev

    return sum/float(len(ids))
    