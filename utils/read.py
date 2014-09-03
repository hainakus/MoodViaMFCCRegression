import csv
import numpy as np
import os
import json
import colorsys
from calc_utils import cart2polar

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
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                val[key].append(float(row[i]))
                aro[key].append(float(row[i+1]))

    return val, aro


def csv_2_dict_va_polar(path):
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
    r = {}
    theta = {}

    # parse csv
    for row in reader:
        # print row[81]
        key = int(row[81])
        if key not in r:
            r[key] = []
            theta[key] = []

        for i in range(82+20, 129, 2):
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                rr, ttheta = cart2polar(float(row[i]), float(row[i+1]))
                r[key].append(rr)
                theta[key].append(ttheta)

    return r, theta


def csv_2_dict_hsv(path):
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
    h = {}
    s = {}
    v = {}

    # parse csv
    for row in reader:
        # print row[81]
        key = int(row[81])
        if key not in h:
            h[key] = []
            s[key] = []
            v[key] = []

        h[key].append(float(row[132]))
        s[key].append(float(row[133]))
        v[key].append(float(row[134]))

    return h, s, v


def csv_2_dict_rgb(path):
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
    r = {}
    g = {}
    b = {}

    # parse csv
    for row in reader:
        # print row[81]
        key = int(row[81])
        if key not in r:
            r[key] = []
            g[key] = []
            b[key] = []

        red, green, blue = colorsys.hsv_to_rgb(float(row[132]), float(row[133]), float(row[134]))
        r[key].append(red)
        g[key].append(green)
        b[key].append(blue)

    return r, g, b


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


def read_csv_song_features_dict(path):
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    ids = []
    features = {}

    for row in reader:
        key = int(row[0][0:-4])
        ids.append(key)

        #print row[1:]

        features[key] = [float(x) for x in row[1:]]

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
    features = {}
    i = 0

    for filename in os.listdir(path):
        key = int(filename[:3])
        ids.append(key)
        vector = single_fake_chroma_read(os.path.join(path, filename))
        # print fetures
        features[key] = vector

        i += 1

    return ids, features


def mean_va(path):
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
    val = []
    aro = []
    id = []

    for row in reader:
        id.append(int(row[81]))
        v = 0
        a = 0
        count = 0

        for i in range(82+20, 129, 2):
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                v += float(row[i])
                a += float(row[i+1])
                count += 1

        val.append(v/count)
        aro.append(a/count)
    return id, val, aro


def uniform_va_hsv(path):
    '''
    Function returns va and hsv for this responses where
    only one mood was selected 
    '''

    print 'rendering csv file'

    # read csv
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    val = []
    aro = []
    id = []
    h = []
    s = [] 
    v = []

    for row in reader:
        va=0
        ar=0
        count = 0

        for i in range(82+20, 129, 2):
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                va = float(row[i])
                ar = float(row[i+1])
                count += 1

        if count == 1:                   
            val.append(va)
            aro.append(ar)
            id.append(int(row[81]))
            h.append(float(row[132]))
            s.append(float(row[133]))
            v.append(float(row[134]))
    return id, val, aro, h, s, v


def uniform_va_rgb(path):
    '''
    Function returns va and rgb for this responses where
    only one mood was selected 
    '''

    print 'rendering csv file'

    # read csv
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    val = []
    aro = []
    id = []
    r = []
    g = [] 
    b = []

    for row in reader:
        va=0
        ar=0
        count = 0

        for i in range(82+20, 129, 2):
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                va = float(row[i])
                ar = float(row[i+1])
                count += 1

        if count == 1:                   
            val.append(va)
            aro.append(ar)
            id.append(int(row[81]))
            red, green, blue = colorsys.hsv_to_rgb(float(row[132]), float(row[133]), float(row[134]))
            r.append(red)
            g.append(green)
            b.append(blue)
    return id, val, aro, r, g, b


def uniform_va_hsv_polar(path):
    '''
    Function returns va and hsv for this responses where
    only one mood was selected 
    '''

    print 'rendering csv file'

    # read csv
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    rrr = []
    tetha = []
    id = []
    h = []
    s = [] 
    v = []

    for row in reader:
        rr=0
        ttetha=0
        count = 0

        for i in range(82+20, 129, 2):
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                va = float(row[i])
                ar = float(row[i+1])
                rr, ttheta = cart2polar(va,ar)
                count += 1

        if count == 1:                   
            tetha.append(ttheta)
            rrr.append(rr)
            id.append(int(row[81]))
            h.append(float(row[132]))
            s.append(float(row[133]))
            v.append(float(row[134]))
    return id, rrr, tetha, h, s, v


def uniform_va_rgb_polar(path):
    '''
    Function returns va and rgb for this responses where
    only one mood was selected 
    '''

    print 'rendering csv file'

    # read csv
    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    # dict: key => song id,
    # value => array of valence (1.dict) arousal (2. dict)
    rrr = []
    tetha = []
    id = []
    r = []
    g = [] 
    b = []

    for row in reader:
        rr=0
        ttetha=0
        count = 0

        for i in range(82+20, 129, 2):
            if float(row[i]) <= 1 and float(row[i+1]) <= 1:
                va = float(row[i])
                ar = float(row[i+1])
                rr, ttheta = cart2polar(va,ar)
                count += 1

        if count == 1:                   
            rrr.append(rr)
            tetha.append(ttheta)
            id.append(int(row[81]))
            red, green, blue = colorsys.hsv_to_rgb(float(row[132]), float(row[133]), float(row[134]))
            r.append(red)
            g.append(green)
            b.append(blue)
    return id, rrr, tetha, r, g, b


def read_csv_col(path, colfrom, colto):
    '''
    function reads columns colfrom:colto(included)
    from csv file on path path
    '''

    ifile = open(path, "rb")
    reader = csv.reader(ifile)

    col_data = []

    for row in reader:
        #print row[colfrom:colto]
        col_data.append([float(i) for i in row[colfrom:colto]])
    return col_data


def read_feature_from_json(path):
    '''
    read json to dict
    '''
    with open(path, 'rb') as fp:
        data = json.load(fp)

    # convert keys from string to int
    for key in data.keys():
        data[int(key)] = data.pop(key)

    return data
