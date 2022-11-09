import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_features(df, time_series, num_col):
    n_bins = 50
    plt_v = 3
    plt_h = 6
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
            axs[2].set_xlabel('Time')
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
                axs[2][i].set_xlabel('Time')
                axs[2][i].set_ylabel(df.columns[i])
    plt.show()

def plot_train_loss(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

def plot_mae_loss(model, X_train, df):
    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.legend(labels = df.columns)
    plt.show()

    # Get reconstruction loss threshold.
    threshold = np.amax(train_mae_loss, axis=0)
    print("Reconstruction error threshold: ", threshold)

    return X_train_pred, train_mae_loss, threshold

def plot_first_sequence(X_train, X_train_pred, df):
    plt.plot(X_train[0], label=df.columns)
    plt.plot(X_train_pred[0], label=['learnt ' + col for col in df.columns])
    plt.legend()
    plt.show()