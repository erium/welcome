import os
import warnings
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

from etl import etl
from features import features
from train import train
from visualize import visualize

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

output_path = './../../output'

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Read the wine-quality csv file from the URL
csv_url = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)

data = pd.read_csv(csv_url)

train, test = train_test_split(data)

with mlflow.start_run():

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lr, "model")
