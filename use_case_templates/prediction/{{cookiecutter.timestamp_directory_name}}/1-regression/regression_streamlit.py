import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from joblib import dump, load
import streamlit as st

pca = False
manual = False
poly = False
normal = False

paths = []

for root, dirs, files in os.walk("./"):
    for dir in dirs:
        for r, d, f in os.walk("./" + dir):
            for file in f:
                if file == "model.joblib":
                    paths.append(r)

st.title("Regression Model Prediction")
if len(paths) > 1:
    st.warning('Multiple models detected in root directory')
    path = st.selectbox('Which model do you want to use?', (paths))
else:
    path = './out'

# Import data
data_df = pd.read_csv(path + '/dataset.csv')
data_df.drop(data_df.columns[0], axis=1, inplace=True)

X_concat = pd.read_csv(path + '/X_concat.csv')
X_concat.drop(X_concat.columns[0], axis=1, inplace=True)
X_columns = X_concat.columns

if os.path.exists(path + '/X_concat_pca.csv'):
    X_concat_pca = pd.read_csv(path + '/X_concat_pca.csv')
    X_concat_pca.drop(X_concat_pca.columns[0], axis=1, inplace=True)
    pca = True

y_concat = pd.read_csv(path + '/y_concat.csv')
y_concat.drop(y_concat.columns[0], axis=1, inplace=True)
y_columns = y_concat.columns

if os.path.exists(path + '/pcaresults.csv'):
    pca_data = pd.read_csv(path + '/pcaresults.csv')

if os.path.exists(path + '/poly_transform.joblib'):
    poly_transform = load(path + '/poly_transform.joblib')
    poly = True
if os.path.exists(path + '/pca_model.joblib'):
    pca_model = load(path + '/pca_model.joblib')
if os.path.exists(path + '/scaler_x.joblib'):
    scaler_x = load(path + '/scaler_x.joblib')
    normal = True
if os.path.exists(path + '/scaler_y.joblib'):
    scaler_y = load(path + '/scaler_y.joblib')
model = load(path + '/model.joblib')

if (not pca and pd.concat([X_concat, y_concat], axis = 1).shape[1] < len(data_df.columns)):
    manual = True

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Prepare data depending on dimensionality reduction and normalisation
if pca:
    X_concat_reduce = X_concat_pca
elif manual:
    X_concat_reduce = X_concat
else:
    X_concat_reduce = X_concat

if normal:
    X_concat = pd.DataFrame(scaler_x.inverse_transform(X_concat))
    X_concat.columns = X_columns

col_minmax = {}
for col in X_concat:
    col_minmax[col] = [min(X_concat[col]), max(X_concat[col])]

st.header('Input data')

st.write('You may manually input data, use sliders, or upload a CSV file.')

option = st.selectbox(
     'How would you like to input data?',
     ('Manual Input', 'Sliders', 'Upload CSV'))

inputs = dict.fromkeys(X_concat.columns)

if option == 'Manual Input': # Inputs in original features then transform
    for col in X_concat.columns:
        number = st.number_input(col, step = 0.0001)
        inputs[col] = [number]
        if (number < col_minmax[col][0]):
            st.caption('Value is lower than minimum value in original dataset. Prediction may be very inaccruate.')
        if (number > col_minmax[col][1]):
            st.caption('Value is higher than maximum value in original dataset. Prediction may be very inaccruate.')

if option == 'Sliders':
    st.write('Note: Sliders only give you a range between double the range of min and max values around the centre for each feature in the dataset.')
    for col in X_concat.columns:
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

inputs = pd.DataFrame(inputs, columns = X_columns)

if option != 'Upload CSV' or uploaded_file is not None:
    if option == 'Upload CSV':
        st.success("CSV Uploaded Successfully")
    st.write('Output Predictions')
    input_col, output_col = st.columns(2)

    with input_col:
        st.caption('Your inputs')
        st.write(inputs)
    if normal:
        inputs = scaler_x.transform(inputs)
    if pca:
        inputs = pca_model.transform(inputs)
    if poly:
        inputs = poly_transform.fit_transform(inputs)

    outputs = pd.DataFrame(model.predict(inputs))
    outputs = pd.DataFrame(scaler_y.inverse_transform(outputs))
    outputs.columns = ['Predicted ' + x for x in y_columns]

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


info = st.checkbox('Display dataset features and model diagnostics')

if info:
    st.subheader("Dataset")
    with st.expander('Full Dataset for model training/testing'):
        data_df
        st.write(str(data_df.shape[0]) + " rows, " + str(data_df.shape[1]) + " columns")


    st.subheader("Features")
    with st.expander('Independent Features'):
        
        X, X_pca = st.columns(2)
        with X:
            st.caption("Original Independent Features")
            st.write(data_df.drop(columns=y_columns))
            st.write(str(len(data_df.columns) - len(y_columns)) + ' dimensions')

        with X_pca:
            if pca or manual:
                st.caption("Reduced Independent Features")
                st.write(X_concat_reduce)
                st.write(str(len(X_concat_reduce.columns)) + ' dimensions')
            else:
                st.write('No reduction')
        
        st.write("Note: Index numbers do not correspond to the full dataset index above")

    if poly:
        X_concat_reduce = poly_transform.fit_transform(X_concat_reduce)

    y_pred = pd.DataFrame(model.predict(X_concat_reduce))

    if normal:
        y_concat = pd.DataFrame(scaler_y.inverse_transform(y_concat))
        y_pred = pd.DataFrame(scaler_y.inverse_transform(y_pred))

    y_concat.columns = y_columns
    y_pred.columns = ['Predicted ' + x for x in y_columns]

    with st.expander('Target and Predicted Features'):
        y, y_predicted = st.columns(2)
        with y:
            st.caption("Target Features")
            st.write(y_concat)
            st.write(str(len(y_concat.columns)) + ' dimensions')
        with y_predicted:
            st.caption("Predicted Target Features")
            st.write(y_pred)
            st.write(str(len(y_pred.columns)) + ' dimensions')

        st.write("Note: Index numbers do not correspond to the full dataset index above")


        fig, ax = plt.subplots()
        ax.scatter(y_concat, y_pred)
        ax.set_title('Original vs Predicted target')
        ax.set_xlabel('Original target')
        ax.set_ylabel('Predicted target')
        st.pyplot(fig)

    full_df = pd.concat([X_concat, y_concat, y_pred], axis = 1)

    st.subheader('Original Data with Prediction')
    st.write(full_df)

    csv = convert_df(full_df)

    st.download_button(
        label="Download original data with prediction as CSV",
        data=csv,
        file_name='regression_prediction.csv',
        mime='text/csv',
    )
