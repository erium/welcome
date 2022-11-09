import os

import numpy as np
import pandas as pd
from math import isnan

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

class Plotter:
    def plot_line(df):
        df.plot(subplots=True, figsize=(8, 2 * len(df.columns)))
        plt.show()

    def plot_box(df):  
        df.boxplot(figsize =(8, 8), grid = False)
        plt.show()

    def show_outliers(df, outliers):
        df_outlier = df.copy()
        df_outlier['outlier'] = 0
        df_outlier['outlier'].loc[outliers.index] = 1
        pairplot_hue = 'outlier'
        palette ={0: "C0", 1: "C3"}
        sns.pairplot(df_outlier, hue = pairplot_hue, palette=palette)
        plt.show()
    
class Importer:
    def import_data(filepaths, sheet, sep, datetime_col, join):
        dfs = []

        for filepath in filepaths:
            filename, filetype = os.path.splitext(filepath)
            if not filetype:
                raise ValueError(filepath + " has missing extension")

            if filetype == '.csv':
                df = pd.read_csv(filepath)
            elif filetype == '.pkl':
                df = pd.read_pickle(filepath)
            elif filetype == '.xlsx':
                df = pd.read_excel(open(filepath,'rb'), sheet_name=sheet)
            elif filetype == '.zip':
                df = pd.read_csv(filepath)
            elif filetype == '.txt':
                df = pd.read_csv(filepath, sep=sep, header=None)
            elif filetype == '.json':
                df = pd.read_json(filepath)
            else:
                raise ValueError(filepath + " has invalid/unsupported extension")

            if datetime_col:
                df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
                df = df.set_index(datetime_col)

            dfs.append(df)

        if join == 'vertical':
            df = pd.concat(dfs)
        elif join == 'inner':
            df = pd.concat(dfs, axis=1, join="inner")
        elif join == 'outer':
            df = pd.concat(dfs, axis=1, join="outer")
        return df
            
            

class PopulationSeparator:
    def separate_populations(df, indexes, label_or_position):
        populations = {}
        for name, index_range in indexes:
            if label_or_position == 'label':
                populations[name] = df.loc[index_range[0]: index_range[1]]
            elif label_or_position == 'position':
                populations[name] = df.iloc[index_range[0]: index_range[1]]
        return populations

class Standardizer:
    def standardize_datatype(df, columns, datatype):
        datatype_dict = {}
        for col in list(columns):
            datatype_dict[col] = datatype
        try:
            df = df.astype(datatype_dict)
        except Exception as e:
            print("Error:", e)
        return df

    def standardize_column_names(df, func):
        try:
            df = df.rename(func, axis='columns')
        except Exception as e:
            print("Error:", e)
        return df

class MissingValues:
    def handle_missing(df, method, custom_value=None, n_neighbors=3):
        print("Number of missing cells:", df.isna().sum().sum())
        if df.isna().sum().sum() != 0:
            if method == 'delete':
                df = df.dropna()
            elif method == 'zero':
                df = df.fillna(0)
            elif method == 'custom':
                df = df.fillna(custom_value)
            elif method == 'mean':
                df = df.fillna(df.mean())
            elif method == 'median':
                df = df.fillna(df.median())
            elif method == 'mode':
                df = MissingValues.mode_impute(df)
            elif method == 'linear':
                df = MissingValues.linear_reg_impute(df)
            elif method == 'knn':
                df = MissingValues.knn_impute(df, n_neighbors)
            elif method == 'interpolate':
                df = df.interpolate(limit_direction='both')
            elif method == 'encode':
                df = MissingValues.encode(df)

        print("Number of missing cells remaining:", df.isna().sum().sum())
        return df

    def mode_impute(df):
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for feature in df.columns: 
            if df[feature].isna().sum().sum() != 0:
                df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))

                df[feature] = df_imputed[0].values

        return df

    def linear_reg_impute(df):
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = {}
        model = LinearRegression()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature]) if type(k) == type("String")}

                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])
        for feature in df.columns: 
            test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
            train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])

            if len(test_df.index) != 0:
                pipe = make_pipeline(StandardScaler(), model)

                X_train = train_df.drop(feature, axis=1)
                test_df.drop(feature, axis=1, inplace=True)
                
                y = train_df[feature] # use non-log-transformed data
                model = pipe.fit(X_train, y)
                pred = model.predict(test_df) # predict values

                test_df[feature]= pred

                if (df[feature].fillna(-9999) % 1  == 0).all():
                    # round back to integers, if original data were integers
                    test_df[feature] = test_df[feature].round()
                    test_df[feature] = test_df[feature].astype('Int64')
                    df[feature].update(test_df[feature])                          
                else:
                    df[feature].update(test_df[feature])  
        if df.isna().sum().sum() != 0:
            imp_mean = IterativeImputer()
            df_imputed = pd.DataFrame(imp_mean.fit_transform(df), index = df.index, columns = df.columns)
            for feature in df.columns:
                if (df[feature].fillna(-9999) % 1  == 0).all():
                    df[feature] = df_imputed[feature].values
                    # round back to INTs, if original data were INTs
                    df[feature] = df[feature].round()
                    df[feature] = df[feature].astype('Int64')   
                else:
                    df[feature] = df_imputed[feature].values

        for feature in df.columns: 
            try:   
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df
    
    def knn_impute(df, n_neighbors):
        cols_num = df.select_dtypes(include=np.number).columns 
        imputer = KNNImputer(n_neighbors=n_neighbors)
        # numerical features
        for feature in df.columns: 
            if feature in cols_num:
                if df[feature].isna().sum().sum() != 0:
                    df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))

                    if (df[feature].fillna(-9999) % 1  == 0).all():
                        df[feature] = df_imputed[0].values
                        # round back to INTs, if original data were INTs
                        df[feature] = df[feature].round()
                        df[feature] = df[feature].astype('Int64')                                        
                    else:
                        df[feature] = df_imputed[0].values

        # categorical features
            else:
                if df[feature].isna().sum()!= 0:
                    mapping = {}
                    mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                    mapping[feature] = mappings
                    df[feature] = df[feature].map(mapping[feature])

                    df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])    

                    # round to integers before mapping back to original values
                    df[feature] = df_imputed[feature].values
                    df[feature] = df[feature].round()
                    df[feature] = df[feature].astype('Int64')  

                    # map values back to original
                    mappings_inv = {v: k for k, v in mapping[feature].items()}
                    df[feature] = df[feature].map(mappings_inv)

        return df

    def encode(df):
        for feature in df.columns: 
            if df[feature].isna().sum().sum() != 0:
                df[feature + "_missing"] = 0
                df.loc[df[feature].isna(), feature + "_missing"] = 1

        return df

class Outliers:
    def find_outliers(percentile, upper_or_lower, df):
        cols_num = df.select_dtypes(include=np.number).columns 
        outliers = []
        for column in df.columns:
            if column not in cols_num:
                print(column, "contains non-numeric data. Outlier detection skipped.")
                continue
            if upper_or_lower == 'upper':
                upper = df[column].quantile(percentile)
                outlier = list(df.index[df[column] > upper])
            elif upper_or_lower == 'lower':
                lower = df[column].quantile(1 - percentile)
                outlier = list(df.index[df[column] < lower])
            outliers.extend(outlier)
        print("Outliers found:", len(df.loc[outliers]))
        return df.loc[outliers]

    def remove_outliers(df, outliers):
        df = df.drop(outliers.index)
        return df

class Encoder:
    def encode(df, one_hot_threshold=10):
        # Non-numeric features
        cols_categ = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)
        if cols_categ:
            for feature in cols_categ:
                if df[feature].nunique() <= one_hot_threshold:
                    df = Encoder.to_onehot(df, feature)
                # LABEL encode if not more than thresholded unique values to encode
                elif df[feature].nunique() > one_hot_threshold:
                    df = Encoder.to_label(df, feature)
        return df

    def to_onehot(df, feature):
        one_hot = pd.get_dummies(df[feature], prefix=feature)
        df = df.join(one_hot)
        return df

    def to_label(df, feature):
        le = LabelEncoder()

        df[feature + '_label'] = le.fit_transform(df[feature].values)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        for key in mapping:
            try:
                if isnan(key):               
                    replace = {mapping[key] : key }
                    df[feature].replace(replace, inplace=True)
            except:
                pass
                    
        return df

class Transformer:
    def single_column_operation(df, cols, function):
        df[cols] = df[cols].apply(function)
        return df

    def multi_column_operation(df, multi_cols, multi_cols_operations, result_col_name):
        df[result_col_name] = df[multi_cols[0]]
        for i, operation in enumerate(multi_cols_operations):
            if operation == 'add':
                df[result_col_name] += df[multi_cols[i+1]]
            elif operation == 'subtract':
                df[result_col_name] -= df[multi_cols[i+1]]
            elif operation == 'multiply':
                df[result_col_name] *= df[multi_cols[i+1]]
            elif operation == 'divide':
                df[result_col_name] /= df[multi_cols[i+1]]
        return df
    
    def normalise(df):
        cols_num = df.select_dtypes(include=np.number).columns 
        scaler = StandardScaler()
        df[cols_num] = scaler.fit_transform(df[cols_num])
        return df
        #return pd.DataFrame(scaler.fit_transform(df), index = df.index, columns = df.columns)

    def binning(df, binning_columns, binning_thresholds, binning_labels):
        for column in binning_columns:
            df[column + '_bin'] = pd.qcut(df[column], q = binning_thresholds, labels = binning_labels)
        return df