# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:43:23 2018

@first-author: DucThinh
@second-author: Sivic (Thanh Nghia)

"""

import numpy as np
import pandas as pd
# from util import operation as opt
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import itertools
import time
# import os
import superalpha.config as cf
import multiprocessing as mp
# import pdb
from superalpha.oop_alpha import stocks_df,preprocess, Alpha, \
    report, momentum_with_market, momentum_long_short, momentum_standard, momentum_volatility

for ii in cf.df_report:
    ii['output'] = mp.Queue()


# for iter in range(len(cf.symbol)):
start_time_all = time.time()
# ------------------------------------------------ Main Process: Begin -------------------------
def main_process(iter, df_ret_alpha):
    cur_year = iter + min(list(cf.symbol))
    # start = datetime(cur_year, 2, 1)
    back_track = timedelta(days=cf.back_track_window)
    start = datetime(cur_year, 1, 1) - back_track
    end = datetime(cur_year, 12, 31)
    print("Data of year %d is processing..." % cur_year)
    # get data of each year year

    data_df_full = preprocess(cf.symbol[cur_year], start=start, end=end)
    data_full = data_df_full.data_df
    data = data_full.iloc[:, :-1]

    # Code to save data foreach years
    writer = pd.ExcelWriter('data/data' + str(cur_year) + '.xlsx', engine='xlsxwriter')
    data.to_excel(writer, 'Sheet1')
    writer.save()


    # ---------------------------------------------------------
    #parameter
    index = 0
    if cf.RUN[index]:

        strat_type = ['4.3','4.4']
        alpha_type = [1,2,3,4,5,6]
        option = ['week', 'month', 'quarter']
        market_return = [20, 60]
        mom_return = [20, 30, 50, 60, 90,100, 200]
        re_1 = [0.3, 0.5,1]
        re_2 = [0.3, 0.5,1]
        stop_loss = [True]
        N_day_sl = [20, 25,30]
        p_sl = [0.2, 0.3]
        q_sl = [0.2,0.3, 0.5]
        take_profit = [False]
        N_day_tp = [20]
        p_tp = [0.1]
        q_tp = [0.2]

        list_param = [strat_type,alpha_type,option, market_return,mom_return,re_1, re_2, stop_loss,
                      N_day_sl, p_sl, q_sl, take_profit, N_day_tp, p_tp, q_tp]

        df = report(data,list_param, cf.df_report[index]['list_param_name']).report_momentum_with_market([cf.rf[iter], cf.rf[iter+1]])
        df_sort = df.sort_index()
        report.save_report(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name'],
                           cf.df_report[index]['list_param_name'])
        report.save_report_sum(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name_sum'],
                               cf.df_report[index]['list_param_name'])

        cf.df_report[index]['output'].put(df)
    else:
        cf.df_report[index]['output'].put(None)
    # ---------------------------------------------------------
    index = 1
    if cf.RUN[index]:
        strat_type = ['1.1_positive_score', '1.1_negative_score']
        alpha_type = [1, 3, 5, 6, 8, 10]
        option = ['week', 'month', 'quarter']
        market_return = [30]
        mom_return = [20, 25, 30, 50, 60, 90, 100, 200]
        re = [0.3, 0.5, 1]
        stop_loss = [True]
        N_day_sl = [20, 25, 30]
        p_sl = [0.2, 0.3]
        q_sl = [0.2,0.3, 0.5]
        take_profit = [False]
        N_day_tp = [20]
        p_tp = [0.1]
        q_tp = [0.2]
        list_param = [strat_type,alpha_type,option, market_return,mom_return,re, stop_loss,
                      N_day_sl, p_sl, q_sl, take_profit, N_day_tp, p_tp, q_tp]
        df = report(data,list_param, cf.df_report[index]['list_param_name']).report_momentum_standard([cf.rf[iter], cf.rf[iter+1]])

        df_sort = df.sort_index()
        report.save_report(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name'],
                           cf.df_report[index]['list_param_name'])
        report.save_report_sum(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name_sum'],
                               cf.df_report[index]['list_param_name'])
        cf.df_report[index]['output'].put(df)
    else:
        cf.df_report[index]['output'].put(None)
    # ---------------------------------------------------------
    index = 2
    if cf.RUN[index]:
        strat_type = ['negative_short_positive_long', 'positive_short_positive_long']
        alpha_type = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        option = ['week', 'month', 'quarter']
        short = [5, 10, 15, 20, 25, 30, 50]
        long = [30, 50, 60, 100, 90, 120, 200]
        stop_loss = [True]
        N_day_sl = [20, 25, 30]
        p_sl = [0.2, 0.3]
        q_sl = [0.2,0.3, 0.5]
        take_profit = [False]
        N_day_tp = [20]
        p_tp = [0.1]
        q_tp = [0.2]
        list_param = [strat_type,alpha_type,option, short, long, stop_loss,
                      N_day_sl, p_sl, q_sl, take_profit, N_day_tp, p_tp, q_tp]

        df = report(data,list_param, cf.df_report[index]['list_param_name']).report_momentum_long_short([cf.rf[iter], cf.rf[iter+1]])
        df_sort = df.sort_index()
        report.save_report(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name'],
                           cf.df_report[index]['list_param_name'])
        report.save_report_sum(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name_sum'],
                               cf.df_report[index]['list_param_name'])
        cf.df_report[index]['output'].put(df)
    else:
        cf.df_report[index]['output'].put(None)
    # run momentum_volatility
    index = 3
    if cf.RUN[index]:

        strat_type = ['high_volatility', 'low_volatility']
        alpha_type = list(range(1, 6))
        option = ['week', 'month', 'quarter']
        option_vol = ['week', 'month', 'quarter']
        no_week = [4, 8, 12]
        mom_return = [5, 10, 15, 25, 30, 50, 60, 100, 120, 200]
        re_vol = [0.3, 0.5, 1]
        re_mom = [0.3, 0.5, 1]
        stop_loss = [True]
        N_day_sl = [20, 25, 30]
        p_sl = [0.2, 0.3]
        q_sl = [0.2, 0.3, 0.5]
        take_profit = [False]
        N_day_tp = [20]
        p_tp = [0.1]
        q_tp = [0.2]
        list_param = [strat_type, alpha_type, option, option_vol, no_week, mom_return, re_vol, re_mom, stop_loss,
                      N_day_sl, p_sl, q_sl, take_profit, N_day_tp, p_tp, q_tp]


        df = report(data,list_param, cf.df_report[index]['list_param_name']).report_momentum_volatility([cf.rf[iter], cf.rf[iter+1]])
        df_sort = df.sort_index()
        report.save_report(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name'],
                           cf.df_report[index]['list_param_name'])
        report.save_report_sum(df_sort, 'results/' + str(cur_year) + cf.df_report[index]['name_sum'],
                               cf.df_report[index]['list_param_name'])
        cf.df_report[index]['output'].put(df)
    else:
        cf.df_report[index]['output'].put(None)

    #parameter momentum_volatility_2

    index = 4
    if cf.RUN[index]:
        strat_type = ['high_volatility_high_momentum', 'low_volatility_high_momentum']
        alpha_type = [6, 8, 9, 10, 12, 13, 14]
        option = ['week', 'month', 'quarter']
        option_vol = ['week', 'month', 'quarter']
        no_week = [4, 6, 10, 12]
        mom_returns = [5, 20, 200]
        re_vol = [0.3, 0.5, 1]
        re_mom = [0.3, 0.5, 1]
        stop_loss = [True]
        N_day_sl = [20, 25, 30]
        p_sl = [0.2, 0.3]
        q_sl = [0.2, 0.3, 0.5]
        take_profit = [False]
        N_day_tp = [20]
        p_tp = [0.1]
        q_tp = [0.2]

        for mom_return in mom_returns:
            print("Mom return %d" %mom_return)

            list_param = [strat_type, alpha_type, option, option_vol, no_week, [mom_return], re_vol, re_mom, stop_loss,
                          N_day_sl, p_sl, q_sl, take_profit, N_day_tp, p_tp, q_tp]
            df = report(data,list_param, cf.df_report[index]['list_param_name']).report_momentum_volatility([cf.rf[iter], cf.rf[iter+1]])
            df_sort = df.sort_index()
            report.save_report(df_sort, 'results/' + str(cur_year) + ('_monret_%d_'%mom_return) + cf.df_report[index]['name'],
                               cf.df_report[index]['list_param_name'])
            report.save_report_sum(df_sort, 'results/' + str(cur_year) + ('_monret_%d_'%mom_return) + cf.df_report[index]['name_sum'],
                                   cf.df_report[index]['list_param_name'])
            cf.df_report[index]['output'].put(df)
    else:
        cf.df_report[index]['output'].put(None)

    ##Correlation of alpha
    index = 5
    if cf.RUN[index]:
        df = pd.DataFrame()
        #Alpha_1
        strat_type = 'low_volatility_low_momentum'
        alpha_type = 12
        option = 'week'
        option_vol = 'month'
        no_week = 8
        market_return = 25
        mom_return = 60
        mom_return = 10
        re_vol = 0.3
        re_mom = 0.3

        volatility_obj = momentum_volatility(data, strat_type, alpha_type, option, option_vol, no_week, mom_return, re_vol, re_mom)
        alpha_df = volatility_obj.get_alpha()
        ret_alpha_1 = volatility_obj.daily_ret_alpha(alpha_df, lag = 1)['ret_alpha']
        df['ret_alpha_1'] = ret_alpha_1

        #Alpha_2
        strat_type = '4.3'
        alpha_type = 5
        option = 'week'
        market_return = 120
        mom_return = 20
        re_1 = 0.3
        re_2 = 0.3

        momentum_obj = momentum_with_market(data,strat_type,alpha_type,option,market_return,mom_return,re_1,re_2)
        alpha_df = momentum_obj.get_alpha()
        ret_alpha_2 = momentum_obj.daily_ret_alpha(alpha_df, lag = 1)['ret_alpha']
        df['ret_alpha_2'] = ret_alpha_2
        #Alpha_3
        strat_type = '4.3'
        alpha_type = 5
        option = 'week'
        market_return = 60
        mom_return = 20
        re_1 = 0.3
        re_2 = 0.3

        momentum_obj = momentum_with_market(data,strat_type,alpha_type,option,market_return,mom_return,re_1,re_2)
        alpha_df = momentum_obj.get_alpha()
        ret_alpha_3 = momentum_obj.daily_ret_alpha(alpha_df, lag = 1)['ret_alpha']
        df['ret_alpha_3'] = ret_alpha_3
        df.corr()

        # cf.df_ret_alpha.put(df)
        # df_ret_alpha.put(df)

        print("Done %d" % len(df))

    print("year %d processed!" % cur_year)
# ------------------------------------------------ Main Process: End -------------------------

processes = [mp.Process(target=main_process, args=(x, cf.df_ret_alpha, )) for x in range(len(cf.symbol))]

# Run processes
for p in processes:
    p.start()


# append output
for ii in cf.df_report:
    # for p in processes:
    # df = [ii['output'].get() for p in processes]
    total_ii = ii['output'].qsize()
    if total_ii <= 0:
        print("No data for this step")
        continue

    ii['df'] = ii['output'].get()
    if ii['df'] is None:
        continue
    for x in range(total_ii-1):
        ret = ii['output'].get()
        if ret is None:
            continue
        ii['df'] = ii['df'].append(ret)


# Exit the completed processes
for p in processes:
    p.join()

print("process done!")
for ii in cf.df_report:
    if ii['df'] is not None:
        df_sort = ii['df'].sort_index()
        report.save_report(df_sort, 'results/' + ii['name'], ii['list_param_name'])
        report.save_report_sum(df_sort, 'results/' + ii['name_sum'], ii['list_param_name'])


running = True
while running:
    if any(process.is_alive() for process in processes):
        time.sleep(30)
    else:
        print('All processes done')
        running = False

print("saved!")
print('\n--- TOTAL TUN TIME %s seconds ---' % (time.time() - start_time_all))
