import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from joblib import dump, load

def scale(X_train, X_test, y_train, y_test, method, path):
    scaler_x = StandardScaler() if method == 'standard' else MinMaxScaler()
    scaler_y = StandardScaler() if method == 'standard' else MinMaxScaler()

    scaler_x.fit(X_train)
    X_train = pd.DataFrame(scaler_x.transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(scaler_x.transform(X_test), columns = X_test.columns)

    scaler_y.fit(y_train)
    y_train = pd.DataFrame(scaler_y.transform(y_train)).squeeze()
    y_test = pd.DataFrame(scaler_y.transform(y_test)).squeeze()

    dump(scaler_x, path + '/scaler_x.joblib')
    dump(scaler_y, path + '/scaler_y.joblib')

    return X_train, X_test, y_train, y_test