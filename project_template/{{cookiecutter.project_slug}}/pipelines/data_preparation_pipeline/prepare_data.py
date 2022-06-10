import os
import sys

import pandas as pd

sys.path.append('../..')
from {{ cookiecutter.module_name }} import data_io

interim_data_path = './../../data/1_interim/'
prepared_data_path = './../../data/2_prepared/'


# <halerium id="3cf4aacd-2c92-4432-8e2d-b06d589a2f64">
def load(file_name):
    df = pd.read_csv(interim_data_path + file_name, delimiter=';')
    # df = data_io.some_function(df)
    df.to_csv(prepared_data_path + file_name)
# </halerium id="3cf4aacd-2c92-4432-8e2d-b06d589a2f64">