# ========== Packages ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from ast import literal_eval
from sklearn.metrics import mean_squared_log_error

# ========== Functions ==========
def get_num_cols(df):
    numerical_cols = []
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64' :
            numerical_cols.append(col)
    return numerical_cols       
  
def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    if pd.isna(x) is False:
        year = x.split('/')[2]
        if int(year) <= 19:
            return x[:-2] + '20' + year
        else:
            return x[:-2] + '19' + year
    else:
        return 0
    
    
def process_date(df):
    date_parts = ["year", "weekday", "month", 'day']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)
    
    return df

def get_words (text):
    text_words = []
    for t in text:
        string = t[-1]         
        words = string.split() 
        words = set(words)     
        text_words += [words] 
    return text_words


def millions(x, pos):
    """The two args are the value and tick position."""
    return '${:1.1f}M'.format(x*1e-6)

def plot_importance(importance,names, max_num_features, model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df = fi_df.head(max_num_features)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
    

def __init__(self):
        pass