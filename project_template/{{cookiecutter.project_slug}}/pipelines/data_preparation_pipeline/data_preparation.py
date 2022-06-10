import os
import sys

from load_data import load
from prepare_data import prepare

raw_file = "raw_file_name"
interim_file = "interim_file_name"

load(raw_file)
prepare(interim_file)