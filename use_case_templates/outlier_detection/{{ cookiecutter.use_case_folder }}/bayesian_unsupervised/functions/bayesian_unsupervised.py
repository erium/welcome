import numpy as np
import pandas as pd
import halerium.core as hal

from joblib import dump, load

def get_posterior_samples(graph, data):
    posterior_model = hal.get_posterior_model(graph, data=data)
    posterior_model.solve()
    post_samples = posterior_model.get_samples(graph.height_curve_parameters, n_samples=1000)

    return post_samples

def get_outlier_df(df, outlier_flags):
    df_final = df.copy()
    df_final['outlier'] = 0
    outliers = []
    for i, outlier in enumerate(outlier_flags):
        if outlier:
            outliers.append(i)
    df_final.loc[df_final.index[outliers], 'outlier'] = 1
    
    return df_final

def export_model(df_final, graph, trained_graph, df, time_series):
    # Export as CSV
    df_final.to_csv('./out/labeled_data')

    # Exports the trained graph
    graph.dump_file('./out/graph.json')
    trained_graph.dump_file("./out/trained_graph.json")
    dump([df, time_series], './out/unsupervised_bayesian.joblib')