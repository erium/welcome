import os
import sys

from load_data import load
from prepare_data import prepare

load(sys.argv[1])
prepare(sys.argv[2])