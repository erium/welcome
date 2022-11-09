import scipy.stats as stats

import matplotlib.pyplot as plt

def plot_linear_corr(df, x_cont, y_cont):
    if x_cont and y_cont:
        fig_h = 5
        fig_w = 8

        fig, ax = plt.subplots(len(x_cont), len(y_cont))
        fig.set_figheight(fig_h * len(x_cont))
        fig.set_figwidth(fig_w * len(y_cont))
        if len(x_cont) == 1 and len(y_cont) == 1:
            slope, intercept, r, p, stderr = stats.linregress(df[x_cont[0]], df[y_cont[0]])
            ax.scatter(df[x_cont[0]], df[y_cont[0]])
            ax.plot(df[x_cont[0]], intercept + slope * df[x_cont[0]], color='r')
            ax.set_xlabel(x_cont[0])
            ax.set_ylabel(y_cont[0])
        elif len(x_cont) == 1:
            for i in range(len(y_cont)):
                slope, intercept, r, p, stderr = stats.linregress(df[x_cont[0]], df[y_cont[i]])
                ax[i].scatter(df[x_cont[0]], df[y_cont[i]])
                ax[i].plot(df[x_cont[0]], intercept + slope * df[x_cont[0]], color='r')
                ax[i].set_xlabel(x_cont[0])
                ax[i].set_ylabel(y_cont[i])
        elif len(y_cont) == 1:
            for i in range(len(x_cont)):
                slope, intercept, r, p, stderr = stats.linregress(df[x_cont[i]], df[y_cont[0]])
                ax[i].scatter(df[x_cont[i]], df[y_cont[0]])
                ax[i].plot(df[x_cont[i]], intercept + slope * df[x_cont[i]], color='r')
                ax[i].set_xlabel(x_cont[i])
                ax[i].set_ylabel(y_cont[0])
        else:
            for i in range(len(x_cont)):
                for j in range(len(y_cont)):
                    slope, intercept, r, p, stderr = stats.linregress(df[x_cont[i]], df[y_cont[j]])
                    ax[i, j].scatter(df[x_cont[i]],df[y_cont[j]])
                    ax[i, j].plot(df[x_cont[i]], intercept + slope * df[x_cont[i]], color='r')
                    ax[i, j].set_xlabel(x_cont[i])
                    ax[i, j].set_ylabel(y_cont[j])
        plt.show()
    elif not x_cont:
        print("No continuous x features")
    else:
        print("No continuous y features")