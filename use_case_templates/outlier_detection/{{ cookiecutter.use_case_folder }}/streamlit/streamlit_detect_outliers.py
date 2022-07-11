import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import halerium.core as hal

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

from tensorflow import keras

from sklearn.metrics import classification_report, plot_roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support

from joblib import dump, load

import streamlit as st

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

autoencoder = False
unsupervised = False
unsupervised_bayesian = False
supervised_bayesian = False

if os.path.exists('./../out/unsupervised_model_data.joblib'):
    unsupervised = True
    data = load('./../out/unsupervised_model_data.joblib')
    original_df = data[1]
    original_col = list(original_df.columns)[:-1]
    time_series = data[2]
    run_models_data = data[0]
    run_models = [k for (k, v) in run_models_data.items()]
    num_models = len(run_models)
    run_multi_models = [model for model in run_models if model not in ['z_score', 'iqr', 'percentile', 'stl']]
    num_multi_models = len(run_multi_models)
elif os.path.exists('./../out/autoencoder_model'):
    autoencoder = True
    model = keras.models.load_model('./../out/autoencoder_model')
    data = load('./../out/autoencoder_model_data.joblib')
    scaler, TIME_STEPS, threshold = data[0]
    original_df = data[1]
    time_series = data[2]
elif os.path.exists('./../out/supervised_model.joblib'):
    data = load('./../out/supervised_model.joblib')
    original_df = data[1]
    time_series = data[2]
    scaler = data[3]
    run_models_data = data[0]
    run_models = run_models_data[0]
elif os.path.exists('./../out/unsupervised_bayesian.joblib'):
    unsupervised_bayesian = True
    data = load('./../out/unsupervised_bayesian.joblib')
    original_df = data[0]
    time_series = data[1]
    graph = hal.Graph.from_specification(file="./../out/graph.json")
    trained_graph = hal.Graph.from_specification(file="./../out/trained_graph.json")
elif os.path.exists('./../out/supervised_bayesian.joblib'):
    supervised_bayesian = True
    data = load('./../out/supervised_bayesian.joblib')
    train_data = data[0]
    dependencies = data[1]
    original_df = data[2]
    time_series = data[3]
    outlier_threshold = data[4]

st.title("Detect Outliers")
with st.expander('Original Training Dataset'):
    st.write(original_df)
    st.write(original_df.shape)
    st.write('Models run:')
    if autoencoder:
        st.write('autoencoder')
    elif unsupervised_bayesian:
        st.write('unsupervised_bayesian')
    elif supervised_bayesian:
        st.write('supervised_bayesian')
    else:
        st.write(run_models)

st.subheader("Upload a file")
uploaded_file = st.file_uploader('')
if uploaded_file is not None:
    if time_series:
        df = pd.read_csv(uploaded_file, parse_dates=['date'], index_col = 'date')
    else:
        df = pd.read_csv(uploaded_file)
    with st.expander('Uploaded Data'):
        st.write(df)
        st.write(df.shape)
    if unsupervised:
        df = df[original_col]
    num_col = len(df.columns)
    plt_v = 3
    plt_h = 5

    # Visualise Decision Boundary if 2 variables
    if autoencoder:
        df_normal = pd.DataFrame(scaler.transform(df), index = df.index, columns = df.columns)
        X_test = create_sequences(df_normal.values, time_steps=TIME_STEPS)
        X_test_pred = model.predict(X_test)
        test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
        anomalies = (test_mae_loss > threshold)
        for i, anomaly in enumerate(anomalies):
            anomalies[i] = True in anomaly
        test_mae_loss = test_mae_loss.reshape((-1))
        outliers = []
        for data_idx in range(TIME_STEPS - 1, len(df_normal) - TIME_STEPS + 1):
            if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
                outliers.append(data_idx)

        if outliers:
            feature = st.selectbox('Choose feature to visualise', df.columns)
            df_subset = df.copy() * np.nan
            df_subset[feature][outliers] = df[feature][outliers]
            fig, ax = plt.subplots()
            df[feature].plot(legend=False, ax=ax)
            df_subset[feature].plot(legend=False, ax=ax, color="r")
            st.pyplot(fig)

    elif unsupervised:
        with st.expander('Outlier Decision Boundaries (for bivariate models)'):
            if num_col == 2:
                xx_min = min(df[df.columns[0]])
                xx_max = max(df[df.columns[0]])
                xx_range = xx_max - xx_min
                yy_min = min(df[df.columns[1]])
                yy_max = max(df[df.columns[1]])
                yy_range = yy_max - yy_min
                xx, yy = np.meshgrid(np.linspace(xx_min - (yy_range), xx_max + (yy_range), 200), np.linspace(yy_min - (yy_range//2 + 1), yy_max + (yy_range//2 + 1), 200))

                fig, axs = plt.subplots(1, num_multi_models, sharex=True, sharey=True)
                fig.suptitle('Outlier Decision Boundaries (for bivariate models)')
                for i, model in enumerate(run_multi_models):
                    if model != 'lof':
                        Z = run_models_data[model][1].predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        axs[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors="red")
                        y_pred = run_models_data[model][1].predict(df)
                    else:
                        Z = run_models_data[model][3].decision_function(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        axs[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors="red")
                        y_pred = run_models_data[model][3].predict(df)
                    colors = np.array(["#377eb8", "#ff7f00"])
                    axs[i].scatter(df[df.columns[0]], df[df.columns[1]], s=10, color=colors[(y_pred + 1) // 2])
                    axs[i].set_xlabel(df.columns[0])
                    axs[i].set_ylabel(df.columns[1])
                    axs[i].set_title(model)
                st.pyplot(fig)
            else:
                st.write('No visualisation available for non-bivariate data')

        selected_model = st.selectbox('Select the model to receive outlier predictions', run_models)
        st.caption(selected_model + ' selected')
        if selected_model in run_multi_models:
            if selected_model == 'lof':
                model = run_models_data[selected_model][3]
            else:
                model = run_models_data[selected_model][1]
            pred = model.predict(df)
            outliers = [i for i, j in enumerate(pred) if j == -1]
        elif selected_model in ['z_score', 'iqr', 'percentile', 'stl']:
            parameter = run_models_data[selected_model][1]
            outliers = []
            if selected_model == 'z_score':
                alpha = 1e-3 # You may change this alpha to adjust the strictness of the normal test
                normal_variables = []
                for column in df.columns:
                    k2, p = stats.normaltest(df[column].values)
                    if p < alpha:
                        normal_variables.append(column)
                for variable in normal_variables:
                    mean = df[variable].mean()
                    std = df[variable].std()
                    upper = mean + parameter * std
                    lower = mean - parameter * std
                    outlier = list(df.index[df[variable] < lower]) + list(df.index[df[variable] > upper])
                    outliers += outlier
            elif selected_model == 'iqr':
                for column in df.columns:
                    percentile25 = df[column].quantile(0.25)
                    percentile75 = df[column].quantile(0.75)
                    iqr = percentile75 - percentile25
                    upper = percentile75 + parameter * iqr
                    lower = percentile25 - parameter * iqr
                    outlier = list(df.index[df[column] < lower]) + list(df.index[df[column] > upper])
                    outliers += outlier
            elif selected_model == 'percentile':
                for column in df.columns:
                    upper = df[column].quantile(parameter)
                    lower = df[column].quantile(1 - parameter)
                    outlier = list(df.index[df[column] < lower]) + list(df.index[df[column] > upper])
                    outliers += outlier
            elif selected_model == 'stl':
                for column in df.columns:
                    result = seasonal_decompose(df[column], model='additive') # additive or multiplicative
                    fig = result.plot()
                    resid = result.resid
                    resid = resid.dropna()
                    mean = resid.values.mean()
                    std = resid.values.std()
                    upper = mean + parameter * std
                    lower = mean - parameter * std
                    outlier = list(resid.index[resid.values < lower]) + list(resid.index[resid.values > upper])
                    outliers += outlier
            elif selected_model == 'arima':
                for column in df.columns:
                    model = ARIMA(df[column], order=(p, d, q))
                    res = model.fit()
                    resid = res.resid
                    mean = resid.values.mean()
                    std = resid.values.std()
                    upper = mean + parameter * std
                    lower = mean - parameter * std
                    outlier = list(resid.index[resid.values < lower]) + list(resid.index[resid.values > upper])
                    outliers += outlier
            outliers = list(set(outliers))
            outliers.sort()
            outliers_index = []
            for outlier in outliers:
                outliers_index.append(df.index.get_loc(outlier))
            outliers = outliers_index
    elif unsupervised_bayesian:
        out_detector = hal.objectives.OutlierDetector(trained_graph, data={trained_graph.age: df["Age"], trained_graph.height: df["Height"]})
        outlier_flags = out_detector()['graph']
        outliers = []
        for i, outlier in enumerate(outlier_flags):
            if outlier:
                outliers.append(i)
    elif supervised_bayesian:
        outliers = []
        test_data = pd.DataFrame(data={"(age)": df["Age"],
                          "(height|age)": df["Height"]})
        causal_structure = hal.CausalStructure(dependencies)
        causal_structure.train(train_data)
        prediction = causal_structure.predict(data=test_data)
        prediction.loc[prediction['(outlier|age,height)'] < outlier_threshold, '(outlier|age,height)'] = 0
        prediction.loc[prediction['(outlier|age,height)'] >= outlier_threshold, '(outlier|age,height)'] = 1
        for i, outlier in enumerate(prediction['(outlier|age,height)']):
            if outlier == 1:
                outliers.append(i)
    else:
        df_normal = pd.DataFrame(scaler.transform(df), index = df.index, columns = df.columns)
        y_pred = run_models.predict(df_normal)
        outliers = []
        for i, pred in enumerate(y_pred):
            if pred == 1:
                outliers.append(i)

    st.write(str(len(outliers)) + ' outliers')

    df['outlier'] = 0
    df.loc[df.index[outliers], 'outlier'] = 1


    with st.expander('Dataset with outliers labeled'):
        palette ={0: "C0", 1: "C3"}
        fig = sns.pairplot(df, hue = 'outlier', palette=palette)
        st.pyplot(fig)
        st.write(df)
        csv = convert_df(df)
        st.download_button(
            label="Download labeled data as CSV",
            data=csv,
            file_name= ('autoencoder' if autoencoder else (selected_model if unsupervised else 'supervised')) + '_labeled_outliers.csv',
            mime='text/csv',
        )

    