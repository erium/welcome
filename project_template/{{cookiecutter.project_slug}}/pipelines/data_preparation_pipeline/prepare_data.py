import os
import sys

import pandas as pd
import numpy as np

prepared_data_path = './../../data/2_prepared/'

def prepare(data):
    df = pd.read_csv(data)
    df = df[0:2]
    df.to_csv(prepared_data_path + 'example_prepared_data.csv')