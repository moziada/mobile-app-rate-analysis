import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os

dataSet_path = 'datasets'


def Normalize_df(ColumnFeatures):
    ColumnFeatures = np.expand_dims(ColumnFeatures.to_numpy(), axis=1)
    scaler = preprocessing.MinMaxScaler().fit(ColumnFeatures)
    return scaler.transform(ColumnFeatures)


def dataPreparation(mainData, bonus=True, reduce_features=False):
    if bonus:
        desc_df = pd.read_csv(os.path.join(dataSet_path, 'appleStore_description.csv'))
        desc_df.drop(['track_name', 'size_bytes'], axis=1, inplace=True)

        mainData = pd.merge(mainData, desc_df, 'left', on='id')
        mainData['app_desc'] = mainData['app_desc'].apply(lambda x: len(x.split()))

        # mainData.to_csv('AppleStore_training_ExtraFeature.csv', index=False, header=True)

    data = mainData.drop(['Unnamed: 0', 'id', 'track_name', 'currency', 'vpp_lic', 'ver'], axis=1)
    data.dropna(how='any', inplace=True)  # delete entire row if a cell is missing

    data['size_bytes'] = Normalize_df(data['size_bytes'])  # size is normalized now, ranges from 0 to 1
    data['price'] = Normalize_df(data['price'])
    data['rating_count_tot'] = Normalize_df(data['rating_count_tot'])
    data['rating_count_ver'] = Normalize_df(data['rating_count_ver'])
    data['user_rating_ver'] = Normalize_df(data['user_rating_ver'])
    data['sup_devices.num'] = Normalize_df(data['sup_devices.num'])
    data['ipadSc_urls.num'] = Normalize_df(data['ipadSc_urls.num'])
    data['lang.num'] = Normalize_df(data['lang.num'])
    if bonus:
        data['app_desc'] = Normalize_df(data['app_desc'])

    contentRatingOneHot = pd.get_dummies(data['cont_rating'])
    primeGenreOneHot = pd.get_dummies(data['prime_genre'])
    data = data.drop(['cont_rating', 'prime_genre'], axis=1)
    data = data.join(contentRatingOneHot)
    data = data.join(primeGenreOneHot)

    if reduce_features:
        significant_cols = ['size_bytes', 'rating_count_tot', 'rating_count_ver', 'user_rating_ver', 'ipadSc_urls.num',
                            'lang.num', 'app_desc', '17+', 'Games', 'user_rating']
        data = data[significant_cols]
        return data

    tmp = data['user_rating']
    data = data.drop(['user_rating'], axis=1)
    data = data.join(tmp)  # making sure that user_rating is last column
    return data


def findCorr(data):
    corr = data.corr(method='spearman')
    top_feature = corr.index[abs(corr['user_rating']) > 0.1]
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr(method='spearman')
    sns.heatmap(top_corr, annot=True)
    plt.show()
