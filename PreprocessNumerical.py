import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import log_loss

from sklearn.preprocessing import normalize


def PreprocessNumericalFeatures():
    rawdata_bots = pd.read_csv('bots_data.csv', sep=",", encoding='latin1')
    rawdata_nonbots = pd.read_csv('nonbots_data.csv', sep=",", encoding='latin1')
    rawdata = pd.concat([rawdata_bots, rawdata_nonbots], ignore_index=True)
    rawdata.fillna('', inplace=True)
    print('total: {}'.format(rawdata.shape))


    # First manually filter twitter accounts which have bot in description
    description = rawdata['description']
    drop_row = []
    for i in range(len(description)):
        if 'bot' in description[i]:
            drop_row.append(i)


    # Need to delete the rows with which 'bot' is in its description
    rawdata.drop(drop_row, inplace=True)


    # Need to delete the columns which are not related to the prediction
    # id, id_str, screen_name, location, description, lang, status, default_profile, default_profile_image, has_extended_profile
    # url needs to be set as boolean, if there's a url, there's big chance it is a bot
    rawdata.drop(['id'], axis=1, inplace=True)
    rawdata.drop(['id_str'], axis=1, inplace=True)
    rawdata.drop(['screen_name'], axis=1, inplace=True)
    rawdata.drop(['location'], axis=1, inplace=True)
    rawdata.drop(['description'], axis=1, inplace=True)
    rawdata.drop(['lang'], axis=1, inplace=True)
    rawdata.drop(['created_at'], axis=1, inplace=True)
    rawdata.drop(['status'], axis=1, inplace=True)
    rawdata.drop(['default_profile'], axis=1, inplace=True)
    rawdata.drop(['default_profile_image'], axis=1, inplace=True)
    rawdata.drop(['has_extended_profile'], axis=1, inplace=True)
    rawdata.drop(['name'], axis=1, inplace=True)


    # Reset index
    rawdata = rawdata.reset_index(drop=True)


    # url, verified, followers_count,friends_count, listedcount, favourites_count, statuses_count
    # url = rawdata['url']
    # url_boolean = []
    # for i in range(len(url)):
    #     if url[i]:
    #         url_boolean.append(1)
    #     else:
    #         url_boolean.append(0)
    rawdata.drop(['url'], axis=1, inplace=True)

    verified = rawdata['verified']
    verified_boolean = []
    for i in range(len(verified)):
        if verified[i] == False:
            verified_boolean.append(0)
        else:
            verified_boolean.append(1)


    bot = rawdata['bot']
    bot_boolean = []
    for i in range(len(bot)):
        if bot[i] == 1:
            bot_boolean.append(1)
        else:
            bot_boolean.append(0)


    # Normalize the numerical data
    numerical_data = rawdata[['followers_count','friends_count', 'listedcount', 'favourites_count', 'statuses_count']]
    numerical_data_l1 = normalize(numerical_data, norm='l1')
    numerical_data_l2 = normalize(numerical_data, norm='l2')


    data = []
    # url_boolean is list
    # numerical_data_l2 is numpy.ndarray
    for i in range(len(rawdata)):
        data.append([verified_boolean[i], numerical_data_l2[i][0], numerical_data_l2[i][1], numerical_data_l2[i][2], numerical_data_l2[i][3], numerical_data_l2[i][4], bot_boolean[i]])

    data = np.asarray(data)


    print len(data)
    # Split into training and testing dataset (80 / 20) randomly
    # Save them as train_data.csv and test_data.csv
    split = np.random.rand(len(data)) < 0.8
    train_data = data[split]

    print train_data[0:5, :]

    X_train = train_data[:, 0:6]
    y_train = train_data[:, 6:]
    y_train = y_train.ravel()

    test_data = data[~split]
    X_test = test_data[:, 0:6]
    y_test = test_data[:, 6:]
    y_test = y_test.ravel()

    return X_train, y_train, X_test, y_test
