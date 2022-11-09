import numpy as np
import pandas as pd
import halerium.core as hal
from halerium import CausalStructure

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def run_model(dependencies, data, outlier_threshold, test_data):
    causal_structure = CausalStructure(dependencies)
    causal_structure.train(data)
    prediction = causal_structure.predict(data=test_data)
    prediction_mean, prediction_std = causal_structure.predict(
        data=test_data, return_std=True)
    prediction.loc[prediction['(outlier|age,height)'] < outlier_threshold, '(outlier|age,height)'] = 0
    prediction.loc[prediction['(outlier|age,height)'] >= outlier_threshold, '(outlier|age,height)'] = 1
    return prediction

# <halerium id="6a35ae62-a925-4e36-8e0a-7586227a67a1">
def show_results(df_test, prediction):
# </halerium id="6a35ae62-a925-4e36-8e0a-7586227a67a1">
    y_test = df_test['outlier']
    y_pred = prediction['(outlier|age,height)']
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=['Non-outlier', 'Outlier'])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(pd.Series([tn, fp, fn, tp], index = ['True Negatives (Non-outliers)', 'False Positives (Non-outliers predicted as outliers)', 'False Negatives (Outliers predicted as non-outliers', 'True Positives (Outliers)']))
    print(report)