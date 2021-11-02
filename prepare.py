import warnings
warnings.filterwarnings('ignore')

from math import sqrt
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import seaborn as sns


############################# Function File for Preparing Data ############################

def handle_nulls(df):
    '''
    This function drops columns that have over 50 percent null values making them not useable
    for the rest of the pipeline and drops the smaller amount of null values after those 
    columns are removed
    '''
    
    df = df.drop(columns = ['FireplaceQu', 'Fence', 'MiscFeature', 'PoolQC', 'Alley'])
    
    df = df.dropna() ## dropping the other null because they are now 10 percent less than the 
    ## amount of observations
    
    return df

def split_data(df):
    '''
    This function splits the original dataframe into three dataframes: train, validate, and test
    for regression modeling purposes
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123)
    
    print('Making Sure Our Shapes Look Good')
    print(f'Train: {train.shape}, Validate: {validate.shape}, Test: {test.shape}')
    
    return, train, validate, test