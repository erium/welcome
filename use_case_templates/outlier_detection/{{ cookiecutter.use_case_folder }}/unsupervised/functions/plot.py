import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def plot_features(time_series, df, num_col):
    n_bins = 50
    plt_v = 3
    plt_h = 5

    if time_series:
        suptitle = 'Time Series, Frequency, and Box plots of features'
        plt_row = 3
        plt_v *= 3
    else:
        suptitle = 'Frequency and Box plots of features'
        plt_row = 2
        plt_v *= 2


    if num_col == 1:
        fig, axs = plt.subplots(plt_row, num_col, figsize=(plt_h*num_col, plt_v))
        fig.suptitle(suptitle)
        axs[0].hist(df[df.columns[0]], bins = n_bins)
        axs[0].set_ylabel('Frequency')
        axs[1].boxplot(df[df.columns[0]], vert=False)
        axs[1].set_xlabel(df.columns[0])
        if time_series:
            axs[2].plot(df)
            axs[2].set_xlabel('time')
            axs[2].set_ylabel(df.columns[0])
    elif num_col > 1:
        fig, axs = plt.subplots(plt_row, num_col, figsize=(plt_h*num_col, plt_v))
        fig.suptitle(suptitle)
        for i in range(num_col):
            axs[0][i].hist(df[df.columns[i]], bins = n_bins)
            axs[0][i].set_ylabel('Frequency')
            axs[1][i].boxplot(df[df.columns[i]], vert=False)
            axs[1][i].set_xlabel(df.columns[i])
            if time_series:
                axs[2][i].plot(df[df.columns[i]])
                axs[2][i].set_xlabel('time')
                axs[2][i].set_ylabel(df.columns[i])    
    plt.show()

def plot_uni_results(run_models_data, time_series, features_to_consider, num_col, df):
    n_bins = 50
    plt_v = 3
    plt_h = 5
    rows = 3
    run_models_data = {k:v[0] for (k, v) in run_models_data.items()}
    z_score_outliers = run_models_data['z_score'][0]
    iqr_outliers = run_models_data['iqr'][0]
    percentile_outliers = run_models_data['percentile'][0]
    univariate_approaches = ['z_score', 'iqr', 'percentile']
    univariate_outliers = [z_score_outliers, iqr_outliers, percentile_outliers]
    if time_series:
        stl_outliers = run_models_data['stl'][0]
        arima_outliers = run_models_data['arima'][0]
        univariate_approaches = ['z_score', 'iqr', 'percentile'].extend(['stl', 'arima'])
        univariate_outliers.extend([stl_outliers, arima_outliers])
        rows = 5

    print('features labeled: ' + str(features_to_consider))
    print('z_score outliers: ' + str(len(z_score_outliers)))
    print('z_score threshold: ' + str(run_models_data['z_score'][1]))
    print('iqr outliers: ' + str(len(iqr_outliers)))
    print('iqr threshold: ' + str(run_models_data['iqr'][1]))
    print('percentile outliers: ' + str(len(percentile_outliers)))
    print('percentile threshold: ' + str(run_models_data['percentile'][1]))
    if time_series:
        print('stl outliers: ' + str(len(stl_outliers)))
        print('stl threshold: ' + str(run_models_data['stl'][1]))
        print('stl outliers: ' + str(len(arima_outliers)))
        print('stl threshold: ' + str(run_models_data['arima'][1]))

    if num_col == 1:
        fig, axs = plt.subplots(rows, num_col, figsize=(plt_h*num_col, plt_v))
        fig.suptitle("Univariate approaches results")
        axs[0].hist(df.iloc[z_score_outliers][df.columns[0]], color='r', bins = n_bins, zorder=2, label='outlier')
        axs[0].hist(df[df.columns[0]], bins = n_bins, zorder=1)
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('z-score outliers')
        axs[1].hist(df.iloc[iqr_outliers][df.columns[0]], color='r', bins = n_bins, zorder=2)
        axs[1].hist(df[df.columns[0]], bins = n_bins, zorder=1)
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('iqr outliers')
        axs[2].hist(df.iloc[percentile_outliers][df.columns[0]], color='r', bins = n_bins, zorder=2)
        axs[2].hist(df[df.columns[0]], bins = n_bins, zorder=1)
        axs[2].set_ylabel('Frequency')
        axs[2].set_xlabel(df.columns[0])
        axs[2].set_title('percentile outliers')
        if time_series:
            axs[3].hist(df.iloc[stl_outliers][df.columns[0]], color='r', bins = n_bins, zorder=2)
            axs[3].hist(df[df.columns[0]], bins = n_bins, zorder=1)
            axs[3].set_ylabel('Frequency')
            axs[3].set_xlabel(df.columns[0])
            axs[3].set_title('stl outliers')
            axs[4].hist(df.iloc[arima_outliers][df.columns[0]], color='r', bins = n_bins, zorder=2)
            axs[4].hist(df[df.columns[0]], bins = n_bins, zorder=1)
            axs[4].set_ylabel('Frequency')
            axs[4].set_xlabel(df.columns[0])
            axs[4].set_title('arima outliers')
    elif num_col > 1:
        fig, axs = plt.subplots(rows, num_col, figsize=(plt_h*num_col, 2 * plt_v))
        fig.suptitle("Univariate approaches results")
        for i in range(num_col):
            axs[0][i].hist(df.iloc[z_score_outliers][df.columns[i]], color='r', bins = n_bins, zorder=2, label='outlier')
            axs[0][i].hist(df[df.columns[i]], bins = n_bins, zorder=1)
            axs[0][i].set_ylabel('Frequency')
            axs[0][i].set_title('z-score outliers')
            axs[1][i].hist(df.iloc[iqr_outliers][df.columns[i]], color='r', bins = n_bins, zorder=2)
            axs[1][i].hist(df[df.columns[i]], bins = n_bins, zorder=1)
            axs[1][i].set_ylabel('Frequency')
            axs[1][i].set_title('iqr outliers')
            axs[2][i].hist(df.iloc[percentile_outliers][df.columns[i]], color='r', bins = n_bins, zorder=2)
            axs[2][i].hist(df[df.columns[i]], bins = n_bins, zorder=1)
            axs[2][i].set_ylabel('Frequency')
            axs[2][i].set_xlabel(df.columns[i])
            axs[2][i].set_title('percentile outliers')
            if time_series:
                axs[3][i].hist(df.iloc[stl_outliers][df.columns[i]], color='r', bins = n_bins, zorder=2)
                axs[3][i].hist(df[df.columns[0]], bins = n_bins, zorder=1)
                axs[3][i].set_ylabel('Frequency')
                axs[3][i].set_xlabel(df.columns[0])
                axs[3][i].set_title('stl outliers')
                axs[4][i].hist(df.iloc[arima_outliers][df.columns[i]], color='r', bins = n_bins, zorder=2)
                axs[4][i].hist(df[df.columns[0]], bins = n_bins, zorder=1)
                axs[4][i].set_ylabel('Frequency')
                axs[4][i].set_xlabel(df.columns[0])
                axs[4][i].set_title('arima outliers')
            handles, labels = axs[0][i].get_legend_handles_labels()
        fig.legend(handles, labels)
    plt.show()

    if time_series:
        return [run_models_data, z_score_outliers, iqr_outliers, percentile_outliers, stl_outliers, arima_outliers]
    return [run_models_data, z_score_outliers, iqr_outliers, percentile_outliers]

def plot_uni_visual(uni_approach_visual, uni_outliers, time_series, df):
    if time_series:
        run_models_data, z_score_outliers, iqr_outliers, percentile_outliers, stl_outliers, arima_outliers = uni_outliers
    else:
        run_models_data, z_score_outliers, iqr_outliers, percentile_outliers = uni_outliers

    if uni_approach_visual == 'z_score':
        out = z_score_outliers
    elif uni_approach_visual == 'iqr':
        out = iqr_outliers
    elif uni_approach_visual == 'percentile':
        out = percentile_outliers
    elif uni_approach_visual == 'stl':
        out = stl_outliers
    elif uni_approach_visual == 'arima':
        out = arima_outliers

    print(uni_approach_visual + ' approach')
    df_uni_outlier = df.copy()
    df_uni_outlier['outlier'] = 'Non Outlier'
    df_uni_outlier.loc[df.index[out], 'outlier'] = 'Outlier'
    palette ={"Non Outlier": "C0", "Outlier": "C3"}
    sns.pairplot(df_uni_outlier, hue = 'outlier', palette=palette)
    plt.show()

def plot_multi_results(run_models, run_models_data, num_models, num_col, df):
    n_bins = 50
    plt_v = 3
    plt_h = 5
    print('models used: ' + str(run_models))

    for model in run_models:
        print(model + ' outliers: ' + str(len(run_models_data[model][0])))

    if num_col == 1:
        fig, axs = plt.subplots(num_models, num_col, figsize=(plt_h*num_col, 2 * plt_v * num_models))
        fig.suptitle("Multivariate approaches results")
        for i, model in enumerate(run_models):
            model_outliers = run_models_data[model][0]
            axs[i].hist(df.iloc[model_outliers][df.columns[0]], color='r', bins = n_bins, zorder=2, label='outlier')
            axs[i].hist(df[df.columns[0]], bins = n_bins, zorder=1)
            axs[i].set_ylabel('Frequency')
            axs[i].set_title(model + ' outliers')
            axs[i].set_xlabel(df.columns[0])
    elif num_col > 1:
        fig, axs = plt.subplots(num_models, num_col, figsize=(plt_h*num_col, plt_v * num_models))
        fig.suptitle("Multivariate approaches results")
        for i, model in enumerate(run_models):
            model_outliers = run_models_data[model][0]
            for j in range(num_col):
                axs[i][j].hist(df.iloc[model_outliers][df.columns[j]], color='r', bins = n_bins, zorder=2, label='outlier')
                axs[i][j].hist(df[df.columns[j]], bins = n_bins, zorder=1)
                axs[i][j].set_ylabel('Frequency')
                axs[i][j].set_title(model + ' outliers')
                axs[i][j].set_xlabel(df.columns[j])
            handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels)

def plot_decision_boundary(num_col, run_models, run_models_data, df):
    if num_col == 2:
        multi_models = [model for model in run_models if model not in ['z_score', 'iqr', 'percentile', 'stl', 'arima']]
        num_multi_models = len(multi_models)

        xx_min = min(df[df.columns[0]])
        xx_max = max(df[df.columns[0]])
        xx_range = xx_max - xx_min
        yy_min = min(df[df.columns[1]])
        yy_max = max(df[df.columns[1]])
        yy_range = yy_max - yy_min
        xx, yy = np.meshgrid(np.linspace(xx_min - (yy_range), xx_max + (yy_range), 200), np.linspace(yy_min - (yy_range//2 + 1), yy_max + (yy_range//2 + 1), 200))
            
        fig, axs = plt.subplots(1, num_multi_models, figsize=(plt_h*num_multi_models, plt_v), sharex=True, sharey=True)
        fig.suptitle('Outlier Decision Boundaries')
        for i, model in enumerate(multi_models):
            if model != 'lof':
                Z = run_models_data[model][1].predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                axs[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors="red")
                y_pred = run_models_data[model][1].predict(df)
            else:
                Z = run_models_data[model][3].decision_function(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                axs[i].contour(xx, yy, Z, levels=[0], linewidths=2, colors="red")
                y_pred = run_models_data[model][1]
            colors = np.array(["#377eb8", "#ff7f00"])
            axs[i].scatter(df[df.columns[0]], df[df.columns[1]], s=10, color=colors[(y_pred + 1) // 2])
            axs[i].set_xlabel(df.columns[0])
            axs[i].set_ylabel(df.columns[1])
            axs[i].set_title(model + ' outliers')
    plt.show()

def plot_multi_visual(multi_approach_visual, run_models_data, df):
    if multi_approach_visual == 'elliptic':
        out = run_models_data['elliptic'][0]
    elif multi_approach_visual == 'svm':
        out = run_models_data['svm'][0]
    elif multi_approach_visual == 'sgd_svm':
        out = run_models_data['sgd_svm'][0]
    elif multi_approach_visual == 'iso':
        out = run_models_data['iso'][0]
    elif multi_approach_visual == 'lof':
        out = run_models_data['lof'][0]

    print(multi_approach_visual + ' approach')
    df_multi_outlier = df.copy()
    df_multi_outlier['outlier'] = 'Non Outlier'
    df_multi_outlier.loc[df.index[out], 'outlier'] = 'Outlier'
    palette ={"Non Outlier": "C0", "Outlier": "C3"}
    sns.pairplot(df_multi_outlier, hue = 'outlier', palette=palette)
    plt.show()

# <halerium id="9e6dc3c9-b29e-482a-af94-93614299120e">
def plot_clustering_metrics(multi_model_scores):
# </halerium id="9e6dc3c9-b29e-482a-af94-93614299120e">
    n_bins = 50
    plt_v = 3
    plt_h = 5
    multi_model_scores = {k:v for (k, v) in multi_model_scores.items() if None not in v}
    fig, axs = plt.subplots(1, 3, figsize=(plt_h*5, plt_v))
    fig.suptitle('Unsupervised Clustering Metrics')
    axs[0].bar([k for (k, v) in multi_model_scores.items()], [v[0] for (k, v) in multi_model_scores.items()])
    axs[0].set_title('Silhouette - Higher: Better Defined Clusters')
    axs[1].bar([k for (k, v) in multi_model_scores.items()], [v[1] for (k, v) in multi_model_scores.items()])
    axs[1].set_title('Calinski-Harabasz - Higher: Better Defined Clusters')
    axs[2].bar([k for (k, v) in multi_model_scores.items()], [v[2] for (k, v) in multi_model_scores.items()])
    axs[2].set_title('Davies-Bouldin - Lower: Better Separated Clusters')
    plt.show()