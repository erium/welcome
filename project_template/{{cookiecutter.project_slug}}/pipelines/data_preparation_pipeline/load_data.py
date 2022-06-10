import os
import sys

import pandas as pd

sys.path.append('../..')
from {{ cookiecutter.module_name }} import data_io

raw_data_path = './../../data/0_raw/'
interim_data_path = './../../data/1_interim/'


# <halerium id="2155529e-86b4-4958-bbf5-780f3b09a030">
def load(file_name):
    df = pd.read_csv(raw_data_path + file_name, delimiter=';')
    # df = data_io.some_function(df)
    df.to_csv(interim_data_path + file_name)
# </halerium id="2155529e-86b4-4958-bbf5-780f3b09a030">