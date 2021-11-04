import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
import acquire as a
import prepare as p
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor


################################ Custom Script For Modeling ######################################

def select_rfe(X, y, k, return_rankings=False, model=LinearRegression()):
    '''
    Function to utilize recursive feature elimination and return a list of the best features 
    to be used for regression modeling
    '''

    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    features = X.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X.columns, rfe.ranking_)))
        return features, rankings
    else:
        return features


def get_metrics(df, model_name,rmse_validate,r2_validate):
    '''
    Function designed to create a dataframe of model metrics for easy comparison of RMSE and R-squared
    values when using regression machine learning
    '''
    df = df.append({
        'model': model_name,
        'rmse_outofsample':rmse_validate, 
        'r^2_outofsample':r2_validate}, ignore_index=True)
    return df
