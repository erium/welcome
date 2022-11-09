import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=UserWarning)

from joblib import dump, load

def test_normality(df, df_uni, alpha):
    normal_variables = []
    for column in df_uni.columns:
        k2, p = stats.normaltest(df[column].values)
        if p < alpha:
            normal_variables.append(column)
    return normal_variables

def run_uni_model(model, threshold, df, normal_variables, df_uni):
    print('Running ' + model)
    print('Threshold: ' + str(threshold))
    if model == 'iqr':
        print('Corresponds to std of: ' + str(0.675 + threshold * 1.35))
    outliers = []
    prediction = list(range(df.shape[0]))
    if model == 'z_score':
        for variable in normal_variables:
            mean = df[variable].mean()
            std = df[variable].std()
            upper = mean + threshold * std
            lower = mean - threshold * std
            outlier = list(df.index[df[variable] < lower]) + list(df.index[df[variable] > upper])
            outliers += outlier
    elif model == 'iqr':
        for column in df_uni.columns:
            percentile25 = df[column].quantile(0.25)
            percentile75 = df[column].quantile(0.75)
            iqr = percentile75 - percentile25
            upper = percentile75 + threshold * iqr
            lower = percentile25 - threshold * iqr
            outlier = list(df.index[df[column] < lower]) + list(df.index[df[column] > upper])
            outliers += outlier
    elif model == 'percentile':
        for column in df_uni.columns:
            upper = df[column].quantile(threshold)
            lower = df[column].quantile(1 - threshold)
            outlier = list(df.index[df[column] < lower]) + list(df.index[df[column] > upper])
            outliers += outlier
    elif model == 'stl':
        for column in df_uni.columns:
            result = seasonal_decompose(df[column], model='additive') # additive or multiplicative
            fig = result.plot()
            resid = result.resid
            resid = resid.dropna()
            mean = resid.values.mean()
            std = resid.values.std()
            upper = mean + threshold * std
            lower = mean - threshold * std
            outlier = list(resid.index[resid.values < lower]) + list(resid.index[resid.values > upper])
            outliers += outlier
    elif model == 'arima':
        for column in df_uni.columns:
            model = ARIMA(df[column], order=threshold)
            res = model.fit()
            print(res.summary())
            resid = res.resid
            resid_df = pd.DataFrame(resid)
            resid_df.plot()
            mean = resid.values.mean()
            std = resid.values.std()
            upper = mean + num_std * std
            lower = mean - num_std * std
            outlier = list(resid.index[resid.values < lower]) + list(resid.index[resid.values > upper])
            outliers += outlier
    outliers = list(set(outliers))
    outliers.sort()
    outliers_index = []
    for outlier in outliers:
        outliers_index.append(df.index.get_loc(outlier))
    outliers = outliers_index

    for p in prediction:
        if p in outliers:
            prediction[p] = -1
        else:
            prediction[p] = 1

    if prediction.count(prediction[0]) == len(prediction): # No outliers detected
        sil = None
        ch = None
        db = None
    else:
        sil = silhouette_score(df, prediction, metric='euclidean')
        ch = calinski_harabasz_score(df, prediction)
        db = davies_bouldin_score(df, prediction)

    print(str(len(outliers)) + ' outliers found')
    return [outliers, threshold, [sil, ch, db]]

def run_multi_model(model, contamination, df):
    n_estimators = 100 # Number of base estimators in the ensemble
    n_neighbors = 20 # Number of neighbors for kneighbors queries. 20 default
    leaf_size = 30 # Affects the speed of construction and query. 30 default
    multi_models = {'elliptic': EllipticEnvelope(contamination=contamination),
    'svm': OneClassSVM(nu = contamination),
    'sgd_svm': SGDOneClassSVM(nu = contamination),
    'iso': IsolationForest(n_estimators=n_estimators, contamination=contamination),
    'lof': LocalOutlierFactor(n_neighbors=20, leaf_size=30, contamination=contamination)}

    clf_predict = LocalOutlierFactor(n_neighbors=20, leaf_size=30, contamination=contamination, novelty=True)
    print('Running ' + model)
    outliers = []
    clf = multi_models[model].fit(df)
    if model == 'lof':
        prediction = clf.fit_predict(df)
        clf_predict.fit(df)
    else:
        prediction = clf.predict(df)

    if list(prediction).count(prediction[0]) == len(prediction): # No outliers detected
        sil = None
        ch = None
        db = None
    else:
        sil = silhouette_score(df, prediction, metric='euclidean')
        ch = calinski_harabasz_score(df, prediction)
        db = davies_bouldin_score(df, prediction)

    for i in range(len(prediction)):
        if prediction[i] == -1:
            outliers.append(i)
    print(str(len(outliers)) + ' outliers found')

    if model == 'iso':
        return [outliers, clf, [sil, ch, db], n_estimators, contamination]
    elif model == 'lof':
        return [outliers, clf, [sil, ch, db], n_neighbors, leaf_size, contamination]
    return [outliers, clf, [sil, ch, db], contamination]