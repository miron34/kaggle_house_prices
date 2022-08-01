import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt


def custom_multiclass_accuracy(y_test, y_pred):
    """
    y_pred list of probabilities to belonging to category 
    """
    results = []
    for t, p in list(zip([x.nonzero()[0] for x in y_test], y_pred.argmax(axis=1))):
        if p in set(t):
            results.append(1)
        else:
            results.append(0)
    return sum(results)/len(results)

def multiclass_accuracy_grouped(y_true, y_pred, *group_by):
    '''
    y_true - многомерный массив с тру мульти-таргетом
    y_pred - многомерный массив с предсказнным мульти-таргетом
    Подсчет точностей попадания топ y_pred в y_true в различных разрезах *group_by
    '''
    display(pd.DataFrame().append({
        'group':'TOTAL',
        'accuracy':custom_multiclass_accuracy(y_true, y_pred)
    }, ignore_index=True))
    
    if group_by:
        for group in group_by:
            total_df = pd.DataFrame()
            for idx, series in pd.DataFrame(s, columns=['group']).reset_index().groupby('group').agg(list).iterrows():
                total_df = total_df.append({
                    'group':idx,
                    'accuracy':custom_multiclass_accuracy(y_true[series['index']],
                                                          y_pred[series['index']])
                }, ignore_index=True)
            display(total_df)


def singleclass_accuracy_grouped(y_true, y_pred, *group_by):
    '''
    y_true - 1-D массив с тру сингл-таргетом
    y_pred - 1-D с предсказнным сингл-таргетом
    Подсчет точностей y_true и y_pred в различных разрезах *group_by
    '''
    y_true, y_pred = list(y_true), list(y_pred)
    
    display(pd.DataFrame().append({
        'group':'TOTAL',
        'accuracy':(pd.Series(y_true) == pd.Series(y_pred)).sum() / len(y_true)
    }, ignore_index=True))
    
#     group_by += [y_true, y_pred]
    if group_by:
        for group in group_by:
            df = pd.DataFrame([y_true, y_pred, group], index=['y_true', 'y_pred', 'group']).T

            total_df = pd.DataFrame()
            for slice_name, slice_ in df.groupby('group'):
                total_df = total_df.append({
                    'group':slice_name,
                    'accuracy':(slice_.y_true == slice_.y_pred).sum()/len(slice_)
                }, ignore_index=True)
            display(total_df)
            
            
            
            
def completeness(df, show_plot = True, return_df=False): 
    ''' 
    Estimates nulls amount by each feature in df
    - show_plot - plotting a bar 
    - return_df - returns df 
    '''
    
    nan_info = df.isnull().sum().sort_values() 
    nan_info_percent = nan_info / len(df) * 100
    if show_plot:
        sns.set(rc={'figure.figsize':(8,8)})
        ax = sns.barplot(x = nan_info_percent, y=nan_info_percent.index)
        ax.tick_params(labelsize=20)
        ax.set_title('% of Nulls', fontsize=20)
        plt.show()
    if return_df:
        return nan_info

def uniqueness(df, show_plot=True, return_df=False):
    '''
    Estimates uniqueness for each feature
    - show_plot - plotting a bar 
    - return_df - returns df 
    '''
    
    nunique_df = df.nunique().reset_index().rename(columns={
        'index':'name', 
        0:'nunique'
    }).sort_values('nunique').append({
        'name':'WHOLE DF',
        'nunique':len(df)
    }, ignore_index=True)
    
    nunique_df_percent = nunique_df.set_index('name', drop=True)['nunique'] / len(df) * 100
    
    if show_plot:
        sns.set(rc={'figure.figsize':(8,8)})
        ax = sns.barplot(x = nunique_df_percent.values, y=nunique_df_percent.index.to_list())
        ax.tick_params(labelsize=20)
        ax.set_title('% of uniqueness', fontsize=20)
        plt.show()
    if return_df:
        return nunique_df    

def distribution_series(series, threshold=20, return_df=False):
    '''
    Plot a distribution of selected series without counting NULLS
    - return_df - returns df 
    - threshold - border of top features amount 
    '''
    # float types recognizes in a wrong way
    if series.dtype != 'str':
        distr_info = series.astype(str).value_counts()
    else:
        distr_info = series.value_counts()
    distr_info = distr_info[distr_info.index.map(lambda x: x not in [np.NaN, None, 'nan'])].copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax = sns.barplot(x = distr_info.values[:threshold], y=distr_info.index[:threshold])
    ax.tick_params(labelsize=20)
    ax.set_title(f'{series.name} - number of records', fontsize=20)
    plt.show()
    if return_df:
        return distr_info

def distribution(features, df, threshold=20, return_df=False, orient='h'):
    '''
    features str or list
    '''
    if type(features) == str:
        features = [features]
        columns_amount = 1
    else:
        columns_amount = 2
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        num = 0
        rows_amount = len(features)//2 + len(features)%2
        heigh = rows_amount*10
        return_df_arr = {}

        fig, axes = plt.subplots(rows_amount, columns_amount, figsize=(20, heigh))
        sns.set()

        for feature in features:
            row = num//2
            column = num%2
            
            # float types recognizes in a wrong way
            if df[feature].dtype != 'str':
                distr_info = df[feature].astype(str).value_counts()
            else:
                distr_info = df[feature].value_counts()
            distr_info = distr_info[distr_info.index.map(lambda x: x not in [np.NaN, None, 'nan'])].copy()
            
            if orient == 'h':
                x_plot = distr_info.values[:threshold]
                y_plot = distr_info.index[:threshold]
            elif orient == 'v':
                x_plot = distr_info.index[:threshold]
                y_plot = distr_info.values[:threshold]
            else:
                print('Choose the orientation: horizonatal or vertical')
                return
            if columns_amount == 1:
                sns.barplot(ax=axes, x = x_plot, y=y_plot)
                axes.tick_params(labelsize=20)
                axes.set_title(f'{df[feature].name} - number of records', fontsize=20)
            else:
                if rows_amount > 1:
                    sns.barplot(ax=axes[row, column], x = x_plot, y=y_plot)
                    axes[row, column].tick_params(labelsize=20)
                    axes[row, column].set_title(f'{df[feature].name} - number of records', fontsize=20)
                else:
                    sns.barplot(ax=axes[column], x = x_plot, y=y_plot)
                    axes[column].tick_params(labelsize=20)
                    axes[column].set_title(f'{df[feature].name} - number of records', fontsize=20)

            num += 1
            return_df_arr[feature] = distr_info
        
        if return_df:
            return return_df_arr

        plt.show()