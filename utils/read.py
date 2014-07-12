import csv
import numpy as np
import os


def csv_2_dict_va(path):
    '''
    Function creates 2 dictionaries for valence and arousal
    for each song in csv
    key - song id
    value - array of values
    '''

    print 'rendering csv file'

    # read csv
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    val = {}
    aro = {}

    # parse csv
    for row in reader:
        # print row[81]
        key = int(row[81])
        if key not in val:
            val[key] = []
            aro[key] = []

        for i in range(82+20, 129, 2):
            if float(row[i]) != 9:
                val[key].append(float(row[i]))
                aro[key].append(float(row[i+1]))

    return val, aro


def read_csv_song_features(path):
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    ids = []
    features = np.array([])

    for row in reader:
        ids.append(int(row[0][0:-4]))

        #print row[1:]

        [float(x) for x in row[1:]]

        if len(features) == 0:
            features = (np.array([float(x) for x in row[1:]])).ravel()
        else:
            features = np.vstack((features, np.array([float(x) for x in row[1:]])))

    return ids, features


def read_eric_va(pathv, patha):
    '''
    Function creates 2 dictionaries for valence and arousal
    for each song in csv
    key - song id
    value - array of values
    '''

    print 'rendering csv file'

    # read csv
    ifilev = open(pathv, "rb")
    ifilea = open(patha, "rb")
    valcsv = csv.reader(ifilev)
    arocsv = csv.reader(ifilea)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    val = {}
    aro = {}

    # parse csv
    for row in valcsv:
        # print row[81]

        key = int(row[0])

        val[key] = map(float, row[1:])

    for row in arocsv:
        # print row[81]

        key = int(row[0])

        aro[key] = map(float, row[1:])

    return val, aro


def single_fake_chroma_read(path):
    ifile = open(path, "rb")
    reader = csv.reader(ifile)
    x = list(reader)
    features = np.array(x).astype('float')
    return np.mean(features, axis=0)


def read_fake_chroma(path):

    ids = []
    features = np.array([])
    i = 0

    for filename in os.listdir(path):
        ids.append(int(filename[:3]))
        vector = single_fake_chroma_read(os.path.join(path, filename))
        # print fetures
        if len(features) == 0:
            features = vector.ravel()
        else:
            features = np.vstack((features, vector.ravel()))
        print i
        i += 1

    return ids, features
