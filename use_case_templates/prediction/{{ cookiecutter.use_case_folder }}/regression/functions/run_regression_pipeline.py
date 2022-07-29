import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def run_regression(X_train, X_test, y_train, y_test, run_models, models, models_param_grid, labels):
    model_results = {}

    for model in run_models:
        print("Running:", model)
        start = time.time()
        pca = PCA()

        pipe = Pipeline(steps=[("pca", pca), ("model", models[model])])

        if model == 'poly':
            poly = PolynomialFeatures()
            pipe = Pipeline(steps=[("pca", pca), ("poly", poly), ("model", models[model])])

        param_grid = models_param_grid[model]
        search = GridSearchCV(pipe, param_grid, n_jobs=2)
        search.fit(X_train, y_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

        r2 = search.score(X_test, y_test)
        time_taken = time.time() - start
        best_pca = PCA(n_components = search.best_params_["pca__n_components"]).fit(X_train)
        best_pca.fit(X_train)
        best_model = models[model]
        best_model.set_params(**{k[7:]:v for k,v in search.best_params_.items() if k.startswith("model")})
        if model == 'poly':
            poly = PolynomialFeatures(degree=search.best_params_['poly__degree'])
            best_model.fit(poly.fit_transform(best_pca.transform(X_train)), y_train)
        else:
            best_model.fit(best_pca.transform(X_train), y_train)

        print("r2 score", r2)
        print("Time taken", time_taken)

        model_results[model] = [r2, time_taken, search.best_params_, best_pca, best_model]

        # Plot the PCA spectrum
        pca.fit(X_train)

        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
        ax0.plot(
            np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
        )
        ax0.set_ylabel("PCA explained variance ratio")

        ax0.axvline(
            search.best_estimator_.named_steps["pca"].n_components,
            linestyle=":",
            label="n_components chosen",
        )
        ax0.legend(prop=dict(size=12))

        # For each number of components, find the best regression results
        results = pd.DataFrame(search.cv_results_)
        components_col = "param_pca__n_components"
        best_clfs = results.groupby(components_col).apply(
            lambda g: g.nlargest(1, "mean_test_score")
        )

        best_clfs.plot(
            x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
        )
        ax1.set_ylabel("cv score")
        ax1.set_xlabel("n_components")

        plt.xlim(1, len(labels))

        plt.tight_layout()
        plt.show()

        if model == 'linear':
            continue
        # For each alpha, find the best regression results
        components_col = "param_model__ccp_alpha" if model in ['tree', 'forest'] else "param_model__alpha" 
        best_clfs = results.groupby(components_col).apply(
            lambda g: g.nlargest(1, "mean_test_score")
        )
        
        best_clfs.plot(
            x=components_col, y="mean_test_score", yerr="std_test_score", legend=False
        )
        plt.ylabel("cv score")
        plt.xlabel("alpha")
        plt.xscale('log')
        plt.title('Regularisation alpha search')

        plt.xlim(0.0001, 1)
        plt.show()
    return model_results