import os
import shutil
from distutils.dir_util import copy_tree

import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import classification_report, plot_roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=FutureWarning)

from joblib import dump, load

def normalise(X_train, X_test, path):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

    dump(scaler, path + '/scaler.joblib')

    return scaler, X_train, X_test

def run_model(model_to_run, X_train, y_train, X_test, y_test):
    n_neighbors = 5
    gamma=2
    c = 1
    factor = 1
    kernel_factor = 1
    max_depth = 5
    n_estimators = 10
    max_features = 1
    alpha = 1
    max_iter = 1000
    if model_to_run == "Linear SVM":
        c = 0.025

    print("Running " + model_to_run)
    model_dict = {"Nearest Neighbors": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Linear SVM": SVC(kernel="linear", C=c), "RBF SVM": SVC(gamma=gamma, C=c),
    "Gaussian Process": GaussianProcessClassifier(factor * RBF(kernel_factor)),
    "Decision Tree": DecisionTreeClassifier(max_depth=max_depth),
    "Random Forest": RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features),
    "Neural Net": MLPClassifier(alpha=alpha, max_iter=max_iter),
    "AdaBoost": AdaBoostClassifier(), "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis()}

    params_dict = {"Nearest Neighbors": [n_neighbors],
    "Linear SVM": [c], "RBF SVM": [gamma, c],
    "Gaussian Process": [factor, kernel_factor],
    "Decision Tree": [max_depth],
    "Random Forest": [max_depth, n_estimators, max_features],
    "Neural Net": [alpha, max_iter],
    "AdaBoost": [], "Naive Bayes": [],
    "QDA": []}

    model = model_dict[model_to_run]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    if model_to_run == "Nearest Neighbors":
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, model.predict(X_test))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=['Non-outlier', 'Outlier'])
    plot_roc_curve(model, X_test, y_test)
    plt.show()

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(pd.Series([tn, fp, fn, tp], index = ['True Negatives (Non-outliers)', 'False Positives (Non-outliers predicted as outliers)', 'False Negatives (Outliers predicted as non-outliers', 'True Positives (Outliers)']))
    print(report)
    return [model, params_dict[model_to_run], [accuracy, roc_auc, precision, recall, fscore], report] # model, parameters, metrics, report