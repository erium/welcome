import numpy as np
import pandas as pd
import halerium.core as hal

from sklearn.model_selection import train_test_split

import itertools
from itertools import chain, combinations

import networkx as nx
import matplotlib.pyplot as plt

# Functions for manual causal structure modelling

def manual_causal_structure(df, dependencies, features_input, features_output, test_size):
    features = list(set([item for sublist in dependencies for item in sublist]))
    data = df[features]
    train, test = train_test_split(data, test_size = test_size)

    causal_structure = hal.CausalStructure(dependencies)
    causal_structure.train(train)
    test_input = test[features_input]
    test_output = test[features_output]
    test_input.reset_index(inplace=True)
    test_output.reset_index(inplace=True)

    influences = []
    for feature in features:
        influence = causal_structure.evaluate_objective(hal.InfluenceEstimator, target=feature)
        influences.append([feature, influence])
    evaluation = causal_structure.evaluate_objective(hal.Evaluator, data=test,
                                        inputs=features_input, metric="r2")
    prediction_mean, prediction_std = causal_structure.predict(data=test_input, return_std=True)

    return [features, evaluation, prediction_mean, prediction_std, features, test]


# Functions for automatic causal structure modelling

# Powerset of sets of dependencies of at least size of number of features
def powerset(iterable):
    s = list(iterable)
    min_set_size = 1
    max_set_size = len(s)
    return chain.from_iterable(list(combinations(s, r)) for r in range(min_set_size, max_set_size))

def auto_causal_structure(df, features, features_input, features_output, test_size):
    # Generate all possible dependencies
    dependencies = []
    for i in itertools.permutations(features, 2):
        dependencies.append(list(i))
    print("Number of dependencies:", len(dependencies))

    # Generate powerset of dependencies
    dependency_powerset = list(powerset(dependencies))
    print("Length of dependency powerset:", len(dependency_powerset))

    # Generate DAGs that include all features
    dag = []
    for dependency_set in dependency_powerset:
        try:
            hal.causal_structure.Dependencies(dependency_set)
        except:
            continue
        else:
            dependencies = list(dependency_set)
            all_dependencies = list(set([item for sublist in dependencies for item in sublist]))
            
            # If it does not include all features specified
            if set(all_dependencies) != set(features):
                continue
            dag.append(dependency_set)
    print("Number of DAGs that include all features:", len(dag))

    results = []
    for count, dependencies in enumerate(dag):
        print('Model ' + str(count + 1) + '/' + str(len(dag)))

        data = df[features]
        train, test = train_test_split(data, test_size = test_size)
        causal_structure = hal.CausalStructure(dependencies)
        causal_structure.train(train)
        test_input = test[features_input]
        test_output = test[features_output]
        test_input.reset_index(inplace=True)
        test_output.reset_index(inplace=True)

        influences = []
        for feature in features:
            influence = causal_structure.evaluate_objective(hal.InfluenceEstimator, target=feature)
            influences.append([feature, influence])
        evaluation = causal_structure.evaluate_objective(hal.Evaluator, data=test,
                                        inputs=features_input, metric="r2")
        prediction_mean, prediction_std = causal_structure.predict(data=test_input, return_std=True)
        
        print("r2 scores")
        for output in features_output:
            print(output, evaluation[output])

        results.append([dependencies, causal_structure, influences, evaluation, prediction_mean, prediction_std, test])
    return results

def model_manual_results(dependencies, features_input, features_output, results):
    features, evaluation, prediction_mean, prediction_std, features, test = results
    
    for feature_out in features_output:
        print("Output Feature:", feature_out)
        columns = list(prediction_mean.columns) + [feature + ' std' for feature in prediction_mean]
        prediction = pd.concat([prediction_mean, prediction_std], axis=1)
        prediction.columns = columns
        print("r2:", evaluation[feature_out])

        for feature_in in features_input:
            prediction.sort_values(by=[feature_in], inplace=True)
            for feature_out in features_output:
                prediction_mean = prediction[features]
                prediction_std = prediction[[feature + ' std' for feature in features]]
                plt.plot(prediction_mean[feature_in], prediction_mean[feature_out], color="red", label='Predicted data points')
                plt.fill_between(prediction_mean[feature_in],
                    (prediction_mean - prediction_std)[feature_out],
                    (prediction_mean + prediction_std)[feature_out],
                    color="red", alpha=0.5)
                plt.scatter(test[feature_in], test[feature_out], label='True data points')
                plt.xlabel(feature_in)
                plt.ylabel(feature_out)
                plt.legend()
                plt.show()

        # Building and displaying the Directed Graph
        G = nx.MultiDiGraph()
        for feature in features:
            G.add_node(feature)
        G.add_edges_from(dependencies)

        color_map = []
        for node in G:
            if node in features_output:
                color_map.append('red')
            else: 
                color_map.append('green')  

        print('Causal Structure')
        nx.draw(G, node_color=color_map, with_labels = True)
        plt.show()

def model_auto_results(features, features_input, features_output, results):
    for feature_out in features_output:
        print("Output Feature:", feature_out)
        best_r2 = max(results, key= lambda x: x[3][feature_out])
        dependencies, causal_structure, influences, evaluation, prediction_mean, prediction_std, test = best_r2
        columns = list(prediction_mean.columns) + [feature + ' std' for feature in prediction_mean]
        prediction = pd.concat([prediction_mean, prediction_std], axis=1)
        prediction.columns = columns
        print("r2:", evaluation[feature_out])

        for feature_in in features_input:
            prediction.sort_values(by=[feature_in], inplace=True)
            for feature_out in features_output:
                prediction_mean = prediction[features]
                prediction_std = prediction[[feature + ' std' for feature in features]]
                plt.plot(prediction_mean[feature_in], prediction_mean[feature_out], color="red", label="Predicted data points")
                plt.fill_between(prediction_mean[feature_in],
                    (prediction_mean - prediction_std)[feature_out],
                    (prediction_mean + prediction_std)[feature_out],
                    color="red", alpha=0.5)
                plt.scatter(test[feature_in], test[feature_out], label='True data points')
                plt.xlabel(feature_in)
                plt.ylabel(feature_out)
                plt.legend()
                plt.show()

        # Building and displaying the Directed Graph
        G = nx.MultiDiGraph()
        for feature in features:
            G.add_node(feature)
        G.add_edges_from(dependencies)

        color_map = []
        for node in G:
            if node in features_output:
                color_map.append('red')
            else: 
                color_map.append('green')  

        print("Best fitting causal structure")
        nx.draw(G, node_color=color_map, with_labels = True)
        plt.show()