import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from bioinfokit.analys import stat 

import matplotlib.pyplot as plt

def time_series_test(df, time_features, significance, results):
    time_results = {'stationary': [], 'normal residuals': []}
    for time_feature in time_features:
        df_time = df[time_feature]
        plt.plot(df_time)
        plt.xlabel('Time')
        plt.ylabel(time_feature)
        plt.show()
        stationarity = adfuller(df_time)
        pvalue = stationarity[1]
        print("Stationarity pvalue:", pvalue)

        results['x'].append(time_feature)
        results['y'].append('-Time Series-')
        results['test'].append('stationarity')
        if pvalue <= significance:
            print(time_feature, "is stationary at significance", significance)
            time_results['stationary'].append(True)
            results['passed'].append(True)
        else:
            print(time_feature, "is not stationary at significance", significance)
            time_results['stationary'].append(False)
            results['passed'].append(False)
        
        stl = STL(df_time, period=7)
        res = stl.fit()
        fig = res.plot()
        resid = res.resid
        k2, p = stats.normaltest(resid)
        print("Normal Residuals pvalue:", p)

        results['x'].append(time_feature)
        results['y'].append('-Time Series-')
        results['test'].append('normal residuals')
        if p > significance:
            print(time_feature, "residuals follow a normal distribution at significance", significance)
            time_results['normal residuals'].append(True)
            results['passed'].append(True)
        else:
            print(time_feature, "residuals do not follow a normal distribution at significance", significance)
            time_results['normal residuals'].append(False)
            results['passed'].append(False)
    return time_results

def univariate_lin_corr(df, x_cont, y_cont, significance, results):
    if not x_cont or not y_cont:
        print("No x or y continuous features")
        return
    
    for i in range(len(x_cont)):
        for j in range(len(y_cont)):
            X = sm.add_constant(df[x_cont[i]])
            result = sm.OLS(df[y_cont[j]], X).fit()
            print('Feature:', x_cont[i], 'Compared to:', y_cont[j])
            results_as_html = result.summary().tables[1].as_html()
            results_df = pd.read_html(results_as_html, header=0, index_col=0)[0].iloc[1:]
            significant_corr_features = results_df.loc[results_df['P>|t|'] <= significance]
            results['x'].append(x_cont[i])
            results['y'].append(y_cont[j])
            results['test'].append('uni')
            if not significant_corr_features.empty:
                print("Correlated features at significance:", significance)
                print(significant_corr_features)
                results['passed'].append(True)
            else:
                print("Features not correlated at significance level:", significance)
                results['passed'].append(False)
            print("__________")

def multivariate_lin_corr(df, x_cont, y_cont, significance, results, results_y):
    if not x_cont or not y_cont:
        print("No x or y continuous features")
        return
    
    for j in range(len(y_cont)):
        X = sm.add_constant(df[x_cont])
        result = sm.OLS(df[y_cont[j]], X).fit()
        print('Feature:', y_cont[j])
        results_as_html = result.summary().tables[1].as_html()
        results_df = pd.read_html(results_as_html, header=0, index_col=0)[0].iloc[1:]
        significant_corr_features = results_df.loc[results_df['P>|t|'] <= significance]
        if not significant_corr_features.empty:
            print("Correlated features at significance:", significance)
            print(significant_corr_features)
            results_y[y_cont[j]].append([x_cont, 'multi', True])
            for x_para in list(significant_corr_features.index):
                results['x'].append(x_para)
                results['y'].append(y_cont[j])
                results['test'].append('multi')
                results['passed'].append(True)
            for x_para in list(results_df.loc[results_df['P>|t|'] > significance].index):
                results['x'].append(x_para)
                results['y'].append(y_cont[j])
                results['test'].append('multi')
                results['passed'].append(False)
        else:
            print("Features not correlated at significance level:", significance)
            for x_para in list(results_df.loc[results_df['P>|t|'] > significance].index):
                results['x'].append(x_para)
                results['y'].append(y_cont[j])
                results['test'].append('multi')
                results['passed'].append(False)
            
        print("__________")

def anova(df, x_multi, y_cont, significance, results):
    for x in x_multi:
        discrete_values = list(set(df[x]))
        for y in y_cont:
            discrete_sets = []
            print("Feature:", y, "compared to discrete:", x)
            for value in discrete_values:
                discrete_sets.append(df[y].loc[df[x] == value])
            plt.boxplot(discrete_sets, labels=discrete_values)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.show()
            fvalue, pvalue = stats.f_oneway(*discrete_sets)
            print("fvalue:", fvalue, "pvalue:", pvalue)
            results['x'].append(x)
            results['y'].append(y)
            results['test'].append('anova')
            if pvalue < significance:
                print("Group mean of" , y, "affected at significance:", significance)
                df_discrete = pd.DataFrame(discrete_sets, index=discrete_values).T
                df_discrete = pd.melt(df_discrete.reset_index(),id_vars=['index'], value_vars=discrete_values)
                df_discrete.columns =['index', 'treatments', 'value']
                res = stat()
                res.tukey_hsd(df=df_discrete, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
                print("Pairwise comparison")
                print(res.tukey_summary)
                results['passed'].append(True)
            else:
                print("Group mean of", y, "not affected at significance:", significance)
                results['passed'].append(False)

def t_test(df, x_binary, y_cont, significance, results):
    for x in x_binary:
        discrete_values = list(set(df[x]))
        for y in y_cont:
            discrete_sets = []
            print("Feature:", y, "compared to discrete:", x)
            for value in discrete_values:
                discrete_sets.append(df[y].loc[df[x] == value])
            plt.boxplot(discrete_sets, labels=discrete_values)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.show()
            fvalue, pvalue = stats.ttest_ind(*discrete_sets)
            print("fvalue:", fvalue, "pvalue:", pvalue)
            results['x'].append(x)
            results['y'].append(y)
            results['test'].append('t_test')
            if pvalue < significance:
                print("Group mean of" , y, "affected at significance:", significance)
                results['passed'].append(True)
            else:
                print("Group mean of", y, "not affected at significance:", significance)
                results['passed'].append(False)

def chi_square_test(df, x_multi, y_multi, significance, results):
    for x in x_multi:
        discrete_values = list(set(df[x]))
        for y in y_multi:
            print("Feature:", y, "compared to:", x)
            contingency_table = pd.crosstab(index=df[x], columns=df[y], margins=True)
            print(contingency_table)
            chi2, pvalue, dof, ex = stats.chi2_contingency(contingency_table)
            results['x'].append(x)
            results['y'].append(y)
            results['test'].append('chi2')
            if pvalue < significance:
                print("Group mean of" , y, "affected at significance:", significance)
                results['passed'].append(True)
            else:
                print("Group mean of", y, "not affected at significance:", significance)
                results['passed'].append(False)