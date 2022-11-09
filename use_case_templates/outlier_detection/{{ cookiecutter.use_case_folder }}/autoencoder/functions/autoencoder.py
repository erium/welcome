import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

from joblib import dump, load

def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def export_model(model, scaler, TIME_STEPS, threshold, time_series, df):
    model.save('./out/autoencoder_model')
    dump([[scaler, TIME_STEPS, threshold], scaler.inverse_transform(df), time_series], './out/autoencoder_model_data.joblib')