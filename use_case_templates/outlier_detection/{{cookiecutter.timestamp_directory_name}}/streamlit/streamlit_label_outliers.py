import os
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st

if 'outliers' not in st.session_state:
    st.session_state.outliers = []

if 'outlier_loaded' not in st.session_state:
    st.session_state.outlier_loaded = False

def change_time_label():
    if time_index in st.session_state.outliers:
        st.session_state.outliers.remove(time_index)
    else:
        st.session_state.outliers.append(time_index)

def change_time_threshold_label():
    if all(index in st.session_state.outliers for index in threshold_indexes):
        for index in threshold_indexes:
            st.session_state.outliers.remove(index)
    else:
        for index in threshold_indexes:
            if index not in st.session_state.outliers:
                st.session_state.outliers.append(index)

def change_discrete_compare_label():
    if compare_index in st.session_state.outliers:
        st.session_state.outliers.remove(compare_index)
    else:
        st.session_state.outliers.append(compare_index)

def change_discrete_category_label():
    if all(index in st.session_state.outliers for index in category_indexes):
        for index in category_indexes:
            st.session_state.outliers.remove(index)
    else:
        for index in category_indexes:
            if index not in st.session_state.outliers:
                st.session_state.outliers.append(index)

def change_discrete_threshold_label():
    if all(index in st.session_state.outliers for index in threshold_indexes):
        for index in threshold_indexes:
            st.session_state.outliers.remove(index)
    else:
        for index in threshold_indexes:
            if index not in st.session_state.outliers:
                st.session_state.outliers.append(index)

def change_data():
    st.session_state.outliers = []
    st.session_state.outlier_loaded = False

def clear_label():
    st.session_state.outliers = []

def clear_specific_label():
    st.session_state.outliers.remove(clear_index)

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

st.title("Label Outliers")

paths = []

for root, dirs, files in os.walk("./../data"):
    for file in files:
        if file.endswith(".csv"):
            paths.append(file)

dataset_col, type_col = st.columns(2)
with dataset_col:
    path = st.selectbox('Select dataset to label', (paths), on_change=change_data)
with type_col:
    data_type = st.radio("Select the type of data", ("Discrete/Continuous", "Time Series"), on_change=change_data)
st.caption("Note: Changing the dataset and type will remove all previously labeled points.")

df_check = pd.read_csv('./../data/' + path)
date_present = 'date' in df_check
outlier_present = 'outlier' in df_check

if not date_present and data_type == "Time Series":
    st.error('Error: No column labeled date present')

if data_type == "Discrete/Continuous":
    df = pd.read_csv('./../data/' + path)
    rows = df.shape[0]

    if outlier_present and not st.session_state.outlier_loaded:
        st.info('Outlier column present in dataset. These will be parsed and added')
        st.session_state.outliers = list(df_check[df_check['outlier'] == 1].index)
        st.session_state.outlier_loaded = True
    else:
        df['outlier'] = 0

    for outlier in st.session_state.outliers:
            df.loc[outlier, 'outlier'] = 1

    discrete_method = st.selectbox('Choose labeling method', ('Individual Comparison', 'Whole value/category'))

    target = st.selectbox('Pick the target variable', (df.columns))

    if discrete_method == 'Individual Comparison':
        discrete_features_compare = st.multiselect('Select the features to compare (may not display correct if >4)', list(df.columns[:-1]), [])

        num_compare = len(discrete_features_compare)

        target_indexes = list(df.sort_values(by=[target]).index)
        if num_compare >= 1:
            compare_select = st.slider(
                    "Select index (sorted for current target variable)",
                    min_value = 0,
                    max_value = rows-1,
                    value = (0),
                    key = "discrete_compare_slider"
                )
            compare_index = target_indexes[compare_select]
            outlier_x = df.loc[st.session_state.outliers][target]

            if num_compare == 1:
                outlier_y = df.loc[st.session_state.outliers][discrete_features_compare[0]]
                fig, ax = plt.subplots()
                fig.suptitle(target + " comparison")
                ax.scatter(df[target][compare_index], df[discrete_features_compare[0]][compare_index], marker='o', color='g', label='selected', zorder = 3)
                ax.scatter(outlier_x, outlier_y, marker = 'o', color='r', label='outliers', zorder = 2)
                ax.scatter(df[target], df[discrete_features_compare[0]], zorder = 1)
                ax.set_xlabel(target)
                ax.set_ylabel(discrete_features_compare[0])
                ax.legend()
            elif num_compare > 1:
                fig, axs = plt.subplots(num_compare, sharex=True)
                fig.suptitle(target + " comparison")
                for i in range(num_compare):
                    outlier_y = df.loc[st.session_state.outliers][discrete_features_compare[i]]
                    axs[i].scatter(df[target][compare_index], df[discrete_features_compare[i]][compare_index], marker='o', color='g', label='selected', zorder = 3)
                    axs[i].scatter(outlier_x, outlier_y, marker = 'o', color='r', label='outliers', zorder = 2)
                    axs[i].scatter(df[target], df[discrete_features_compare[i]], zorder = 1)
                    axs[i].set_xlabel(target)
                    axs[i].set_ylabel(discrete_features_compare[i])
                    handles, labels = axs[i].get_legend_handles_labels()
                fig.legend(handles, labels)

            st.pyplot(fig)
            st.caption("Current selected point")
            st.write(df.iloc[compare_index])

            label_button_text = "Unlabel as outlier" if compare_index in st.session_state.outliers else "Label as outlier"
            label_button = st.button(label_button_text, on_click=change_discrete_compare_label, key="discrete_compare_label_button")
        


    if discrete_method == 'Whole value/category':
        st.subheader('Distribution of variable')
        counts = df[target].value_counts()
        outlier_counts = df.loc[df['outlier']==1][target].value_counts()
        fig, ax = plt.subplots()
        ax.bar(outlier_counts.index, outlier_counts.values, color='r', zorder=2, label='outlier')
        ax.bar(counts.index, counts.values, zorder=1, label='non-outlier')
        if st.session_state.outliers:
            ax.legend()
        st.pyplot(fig)

        category, divider, threshold = st.columns([5, 1, 5])
        with category:
            outlier_category = st.selectbox('Pick the category to label as outlier', (sorted(set(df[target].values))))
            category_indexes = df.loc[df[target]==outlier_category].index
            
            label_button_text = "Unlabel as outlier" if all(index in st.session_state.outliers for index in category_indexes) else "Label as outlier"
            label_button = st.button(label_button_text, on_click=change_discrete_category_label, key="discrete_category_label_button")
            st.caption('Note: You may only unlabel a category if all values of that category have been labeled as outliers')
        with divider:
            st.subheader("OR")
        with threshold:
            threshold_select = st.slider(
                "Select threshold range (values outside this range will be considered as outliers)",
                min_value = min(counts.index),
                max_value = max(counts.index),
                value = (min(counts.index), max(counts.index)),
                key = "discrete_threshold_slider"
            )

            threshold_indexes = list(df.loc[df[target] < threshold_select[0]].index) + list(df.loc[df[target] > threshold_select[1]].index)

            if threshold_select == (min(counts.index), max(counts.index)):
                st.caption("Full Range Selected")
            else:
                label_button_text = "Unlabel outliers not in range" if all(index in st.session_state.outliers for index in threshold_indexes) else "Label outliers not in range"
                label_button = st.button(label_button_text, on_click=change_discrete_threshold_label, key="discrete_threshold_label_button")
            st.caption("Note: You may only unlabel outliers not in range if all values in that range have been labeled as outliers")



# For Time Series Data
if data_type == "Time Series" and date_present:
    df = pd.read_csv('./../data/' + path, parse_dates=['date'], index_col = 'date')
    rows = df.shape[0]

    if outlier_present and not st.session_state.outlier_loaded:
        st.info('Outlier column present in dataset. These will be parsed and added')
        st.session_state.outliers = list(df[df['outlier'] == 1].index)
        st.session_state.outlier_loaded = True
    else:
        df['outlier'] = 0
    for outlier in st.session_state.outliers:
            df.loc[outlier, 'outlier'] = 1

    time_method = st.selectbox("Select the method to label outliers", ("Single point", "Threshold"))

    if time_method == "Single point":

        time_targets = st.multiselect('Select the features to compare (may not display correct if >4)', list(df.columns[:-1]), [])

        num_compare_time = len(time_targets)

        if num_compare_time >=1:
            time_range = st.slider(
                "Select your time range (to zoom)",
                min_value = df.index[0].to_pydatetime(),
                max_value = df.index[rows-1].to_pydatetime(),
                value = (df.index[0].to_pydatetime(), df.index[rows-1].to_pydatetime()),
                key = "time_range_slider")

            time_index = st.slider(
                "Select your time (to pinpoint a single point)",
                min_value = time_range[0],
                max_value = time_range[1],
                value = time_range[0],
                key = "time_select_slider")

            outliers_in_time_range = [x for x in st.session_state.outliers if x >= time_range[0] and x <= time_range[1]]

            outlier_x = df.loc[outliers_in_time_range].index

            if num_compare_time == 1:
                outlier_y = df.loc[outliers_in_time_range][time_targets[0]]
                fig, ax = plt.subplots()
                fig.suptitle("Time series comparison")
                ax.scatter(time_index, df[time_targets[0]][time_index], marker='o', color='g', label='selected', zorder = 3)
                ax.scatter(outlier_x, outlier_y, marker = 'o', color='r', label='outliers', zorder = 2)
                ax.plot(df[time_targets[0]].loc[time_range[0]:time_range[1]], zorder = 1)
                ax.set_ylabel(time_targets[0])
                ax.legend()
            elif num_compare_time > 1:
                fig, axs = plt.subplots(num_compare_time, sharex=True)
                fig.suptitle("Time series comparison")
                for i in range(num_compare_time):
                    outlier_y = df.loc[outliers_in_time_range][time_targets[i]]
                    axs[i].scatter(time_index, df[time_targets[i]][time_index], marker='o', color='g', label='selected', zorder = 3)
                    axs[i].scatter(outlier_x, outlier_y, marker = 'o', color='r', label='outliers', zorder = 2)
                    axs[i].plot(df[time_targets[i]].loc[time_range[0]:time_range[1]], zorder = 1)
                    axs[i].set_ylabel(time_targets[i])
                    handles, labels = axs[i].get_legend_handles_labels()
                fig.legend(handles, labels)

            st.pyplot(fig)
            st.caption("Current selected point")
            st.write(df.loc[time_index])

            label_button_text = "Unlabel as outlier" if time_index in st.session_state.outliers else "Label as outlier"
            label_button = st.button(label_button_text, on_click=change_time_label, key="time_label_button")

    if time_method == "Threshold":
        time_target = st.selectbox("Select the feature to label", list(df.columns[:-1]))

        time_range = st.slider(
                "Select your time range (to zoom)",
                min_value = df.index[0].to_pydatetime(),
                max_value = df.index[rows-1].to_pydatetime(),
                value = (df.index[0].to_pydatetime(), df.index[rows-1].to_pydatetime()),
                key = "time_range_slider")

        threshold_range = st.slider(
                "Select your threshold range",
                min_value = min(df[time_target]),
                max_value = max(df[time_target]),
                value = (min(df[time_target]), max(df[time_target])),
                key = "threshold_range_slider")

        outliers_in_time_range = [x for x in st.session_state.outliers if x >= time_range[0] and x <= time_range[1]]

        outlier_x = df.loc[outliers_in_time_range].index
        outlier_y = df.loc[outliers_in_time_range][time_target]
        fig, ax = plt.subplots()
        fig.suptitle("Time series comparison")
        ax.scatter(outlier_x, outlier_y, marker = 'o', color='r', label='outliers', zorder = 2)
        ax.plot(df[time_target].loc[time_range[0]:time_range[1]], zorder = 1)
        high = ax.axhline(threshold_range[1], color='r',ls=':')
        high.set_label('threshold high')
        low = ax.axhline(threshold_range[0], color='r',ls='--')
        low.set_label('threshold low')
        ax.set_ylabel(time_target)
        ax.legend()

        st.pyplot(fig)

        threshold_indexes = list(df.loc[df[time_target] < threshold_range[0]].index) + list(df.loc[df[time_target] > threshold_range[1]].index)

        if threshold_range == (min(df[time_target]), max(df[time_target])):
            st.caption("Full Range Selected")
        else:
            label_button_text = "Unlabel outliers not in range" if all(index in st.session_state.outliers for index in threshold_indexes) else "Label outliers not in range"
            label_button = st.button(label_button_text, on_click=change_time_threshold_label, key="discrete_time_label_button")
        st.caption("Note: You may only unlabel outliers not in range if all values in that range have been labeled as outliers")

with st.expander('Outliers'):
    st.subheader("Outliers")
    st.write(df.loc[st.session_state.outliers])
    st.caption(str(df.loc[df['outlier']==1].shape[0]) + " outliers" if st.session_state.outliers else "No outliers labeled")

    st.subheader("Clear outliers")
    clear_specific_col, divider, clear_all_col = st.columns([5, 1, 5])
    with clear_specific_col:
        clear_index = st.selectbox("Select index to clear", st.session_state.outliers)
        st.button("Clear specified outlier", on_click=clear_specific_label)
    with divider:
        st.subheader("OR")
    with clear_all_col:
        st.button("Clear all outliers", on_click=clear_label)
        


with st.expander("Dataset with outliers"):
    if st.session_state.outliers:
        st.dataframe(df.style.highlight_max())
    else:
        st.dataframe(df)
    st.write(str(rows) + " rows " + str(df.shape[1]) + " columns (with outlier column)")

    csv = convert_df(df)
    st.download_button(
        label="Download labeled data as CSV",
        data=csv,
        file_name='labeled_' + path,
        mime='text/csv',
    )