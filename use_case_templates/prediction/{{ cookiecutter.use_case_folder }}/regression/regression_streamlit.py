import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from joblib import dump, load
import streamlit as st

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.title("Regression Model Prediction")

df = pd.read_csv('./out/dataset.csv', index_col = 0)
model_data = load('./out/model.joblib')
model_name = model_data[0]
model_params = model_data[1][2]
labels = model_data[2]
pca = model_data[1][3]
model = model_data[1][4]
target = model_data[3]

scaler_x = load('./out/scaler_x.joblib')
scaler_y = load('./out/scaler_y.joblib')

with st.expander("Original Dataframe"):
    st.dataframe(df)
with st.expander("Trained Model"):
    st.subheader("Model name")
    st.write(model_name)
    st.write("Model Training Score (r2)")
    st.write(model_data[1][0])
    st.write("Model Training Time (seconds)")
    st.write(model_data[1][1])
    st.subheader("Model parameters")
    st.write(model_params)

df_input = df[labels]
df_target = df[target]
col_minmax = {}
for col in df_input:
    col_minmax[col] = [min(df_input[col]), max(df_input[col])]

st.header('Input data')

st.write('You may manually input data, use sliders, or upload a CSV file.')

option = st.selectbox(
     'How would you like to input data?',
     ('Manual Input', 'Sliders', 'Upload CSV'))

inputs = dict.fromkeys(df.columns)

if option == 'Manual Input': # Inputs in original features then transform
    for col in df_input.columns:
        number = st.number_input(col, step = 0.0001)
        inputs[col] = [number]
        if (number < col_minmax[col][0]):
            st.caption('Value is lower than minimum value in original dataset. Prediction may be very inaccruate.')
        if (number > col_minmax[col][1]):
            st.caption('Value is higher than maximum value in original dataset. Prediction may be very inaccruate.')

if option == 'Sliders':
    st.write('Note: Sliders only give you a range between double the range of min and max values around the centre for each feature in the dataset.')
    for col in df_input.columns:
        low = col_minmax[col][0]
        high = col_minmax[col][1]
        span = high - (high + low)/2
        number = st.slider(col, low - span, high + span, low - span)
        inputs[col] = [number]

if option == 'Upload CSV':
    st.write('Note: CSV feature dimension must match the original dataset dimensions with manual reduction.')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        upload_raw = pd.read_csv(uploaded_file)
        inputs = upload_raw.copy()

inputs = pd.DataFrame(inputs, columns = df_input.columns)

if option != 'Upload CSV' or uploaded_file is not None:
    if option == 'Upload CSV':
        st.success("CSV Uploaded Successfully")
    st.write('Output Predictions')
    input_col, output_col = st.columns(2)

    with input_col:
        st.caption('Your inputs')
        st.write(inputs)

    inputs = scaler_x.transform(inputs)
    inputs = pca.transform(inputs)
    if model_name == 'poly':
        poly_transform = PolynomialFeatures(degree=model_params["poly__degree"])
        inputs = poly_transform.fit_transform(inputs)

    outputs = pd.DataFrame(model.predict(inputs))
    outputs = pd.DataFrame(scaler_y.inverse_transform(outputs))
    outputs.columns = ['Predicted ' + x for x in target]

    with output_col:
        st.caption('Predicted outputs')
        st.write(outputs)

    if option == 'Upload CSV':
        manual_csv = convert_df(pd.concat([upload_raw, outputs], axis = 1))
        st.download_button(
            label="Download data with prediction as CSV",
            data=manual_csv,
            file_name='uploaded_file_prediction.csv',
            mime='text/csv',
        )
