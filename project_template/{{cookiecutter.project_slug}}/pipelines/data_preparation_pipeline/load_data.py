import os
import sys

import pandas as pd

raw_data_path = './../../data/0_raw/'

def load(data):
    df = pd.read_csv(data, delimiter=';')
    df.to_csv(raw_data_path + 'example_raw_data.csv')