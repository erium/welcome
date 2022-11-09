import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

def show_results(predicted, true):
    plt.title('Difference between true and predicted values')
    plt.plot(true-predicted)
    plt.show()

    mse = mean_squared_error(true, predicted)
    r2 = r2_score(true, predicted)
    print('mse:', mse, 'r2', r2)