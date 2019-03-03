# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def forward_return(close_price_df, lag):
    ret_df = pd.DataFrame(columns = close_price_df.columns)
    ret_df['date'] = close_price_df['date'].values[:-lag]
    ret_df.iloc[:,1:] = close_price_df.iloc[:,1:].values[lag:]/close_price_df.iloc[:,1:].values[:-lag] - 1    
    return ret_df

# TODO: check this function
def daily_ret_alpha(close_price_df, alpha_df, lag = 1):
    daily_ret_df = pd.DataFrame()
    daily_ret_df['date'] = alpha_df['date']
    daily_ret_df['trade'] = [0 if alpha_df.iloc[x,1:].isnull().all() else 1 for x in alpha_df.index]
    alpha_df = alpha_df.fillna(0)
    ret_df = forward_return(close_price_df, lag)
    ret_df = ret_df[ret_df['date'] >= alpha_df['date'].values[0]]

    daily_ret_df['ret'] = pd.DataFrame(ret_df.values[:,1:]*alpha_df.values[:-lag,1:]).sum(axis=1)
    return daily_ret_df  

def daily_ret_to_db(daily_ret_df): 
    annual_ret_db = {}
    temp_year = [i.year for i in daily_ret_df['date']]
    year_list = list(sorted(set(temp_year)))
    old_lastday = 0
    for year in year_list:
        idx_lastday = max(i for i,e in enumerate(temp_year) if e==year)
        annual_ret_db[year] = daily_ret_df.loc[old_lastday:idx_lastday]
        old_lastday = idx_lastday+1
    return annual_ret_db

def annual_arithmetic_return(annual_ret_db):
    annual_ret = pd.DataFrame(columns = ['ret'])
    idx = 0
    keys = sorted(annual_ret_db.keys())
    for i in keys:
        temp = annual_ret_db[i].loc[:,'ret']
        trading_date = temp.count()
        temp = temp.fillna(0)
        annual_ret.loc[idx,'ret'] = temp.sum()*252/trading_date*100
        idx += 1
    return annual_ret

def annual_geometric_return(annual_ret_db):
    annual_ret = pd.DataFrame(columns = ['ret'])
    idx = 0
    keys = sorted(annual_ret_db.keys())
    for i in keys:
        temp = annual_ret_db[i].loc[:,'ret']
        trading_date = temp.count()
        temp = temp.fillna(0)
        temp = temp+1
        annual_ret.loc[idx,'ret'] = (temp.prod()**(252/trading_date)-1)*100
        idx += 1
    return annual_ret
    
def annual_arithmetic_sharpe_ratio(annual_ret_db, rf):
    annual_ret = annual_arithmetic_return(annual_ret_db)
    annual_ret_std = []
    keys = sorted(annual_ret_db.keys())
    for i in keys:
        temp = annual_ret_db[i].loc[:,'ret']
        temp = temp.fillna(0)
        annual_ret_std += [np.std(temp)]
    annual_sharpe_ratio = pd.DataFrame(
        pd.to_numeric((annual_ret['ret']/100 - rf),  errors='coerce')/(np.sqrt(252)*pd.Series(annual_ret_std)),
        columns = ['ret'])
    return annual_sharpe_ratio

def annual_geometric_sharpe_ratio(annual_ret_db, rf):
    annual_ret = annual_geometric_return(annual_ret_db)
    annual_ret_std = []
    keys = sorted(annual_ret_db.keys())
    for i in keys:
        temp = annual_ret_db[i].loc[:,'ret']
        temp = temp.fillna(0)
        annual_ret_std += [np.std(temp)]
    annual_sharpe_ratio = pd.DataFrame(
        pd.to_numeric((annual_ret['ret']/100 - rf),  errors='coerce')/(np.sqrt(252)*pd.Series(annual_ret_std)),
        columns = ['ret'])
    return annual_sharpe_ratio    
    
def winning_rate(annual_ret_db):
    flag = True
    wr = pd.DataFrame(columns = ['rate'])
    keys = sorted(annual_ret_db.keys())
    for i,v in enumerate(keys):
        temp = annual_ret_db[v].loc[:,'ret']
        trade_date = sum(annual_ret_db[v].loc[:,'trade'])
        wr.loc[i] = [len(temp[temp > 0])*1.0/trade_date]
    return flag, wr

def winning_rate_all(daily_ret_df):
    trade_date = sum(daily_ret_df['trade'])
    wr = len(daily_ret_df[daily_ret_df['ret'] > 0 ])*1.0/trade_date
    return wr
    
def arithmetic_max_drawdown(annual_ret_db):
    mdd = pd.DataFrame(columns = ['rate'])
    keys = sorted(annual_ret_db.keys())
    for ii,v in enumerate(keys):
        temp_mdd = [0.000001]
        temp = annual_ret_db[v].loc[:,'ret']
        temp = temp.fillna(0.000001)
        temp[temp == 0] = 0.000001
        #arithmetic
        cum_daily_ret = temp.cumsum()
        len_cum_daily_ret = len(cum_daily_ret)
        for i in range(1,len_cum_daily_ret):
            min_value = min(cum_daily_ret[1:i+1])
            min_idx = np.argmin(cum_daily_ret[1:i+1])+1
            max_value = max(cum_daily_ret[:min_idx])
            if max_value > min_value:
                temp_mdd += [(1- min_value/max_value)*100]
            else:
                temp_mdd += temp_mdd[-1:]
        mdd.loc[ii] = temp_mdd[-1:]
    return mdd
                
def geometric_max_drawdown(annual_ret_db):
    mdd = pd.DataFrame(columns = ['rate'])
    keys = sorted(annual_ret_db.keys())
    for ii,v in enumerate(keys):
        temp_mdd = [0.000001]
        temp = annual_ret_db[v].loc[:,'ret']
        temp = temp.fillna(0.000001)
        temp[temp == 0] = 0.000001
        #geometric
        cum_daily_ret = (temp+1).cumprod() - 1
        len_cum_daily_ret = len(cum_daily_ret)
        for i in range(1,len_cum_daily_ret):
            min_value = min(cum_daily_ret[1:i+1])
            min_idx = np.argmin(cum_daily_ret[1:i+1])+1
            max_value = max(cum_daily_ret[:min_idx])
            if max_value > min_value:
                temp_mdd += [(1- min_value/max_value)*100]
            else:
                temp_mdd += temp_mdd[-1:]
        mdd.loc[ii] = temp_mdd[-1:]
    return mdd
    
#    plt.figure(figsize=(15,8))
#    plt.plot(cum_daily_ret[160:])
#    plt.show()