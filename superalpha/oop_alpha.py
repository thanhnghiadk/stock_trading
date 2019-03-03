
import numpy as np
import pandas as pd
from util import operation as opt
import matplotlib.pyplot as plt
import itertools
import os
import multiprocessing as mp
import config as cf

np.seterr(divide='ignore', invalid='ignore')

class stocks_df:
    def __init__(self, symbol_list):
        self.path = '../updated_data/trading_stock/report/'
        stock_dict = {}
        self.symbol_list = symbol_list
        self.stocks_df = pd.DataFrame()
        trading_date_df = pd.DataFrame(columns=['date'])
        for i in self.symbol_list:
            if not os.path.isfile(self.path + i + '.csv'):
                continue
            print(i + '.csv')
            stock_df = pd.read_csv(self.path + i + '.csv')
            stock_df = stock_df[['<Ticker>', '<DTYYYYMMDD>', '<CloseFixed>']].rename( \
                columns={'<Ticker>': 'symbol', '<DTYYYYMMDD>': 'date', '<CloseFixed>': 'close'})
            stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y%m%d')
            stock_df = stock_df.sort_values(by='date')
            trading_date_df = trading_date_df.merge(pd.DataFrame(stock_df['date']), how='outer')
            stock_dict[i] = stock_df[['date', 'close']].reset_index(drop=True)
        trading_date_df = trading_date_df.drop_duplicates().sort_values(by='date').reset_index(drop=True)
        for i in self.symbol_list:
            if i in stock_dict:
                stock_df_fill = trading_date_df.merge(stock_dict[i], on='date', how='left')
                stock_df_fill = stock_df_fill.fillna(method='ffill')
                stock_df_fill['symbol'] = i
                self.stocks_df = self.stocks_df.append(stock_df_fill, ignore_index=True)
            else:
                print('%s not exist!' % i)
        self.stocks_df = self.stocks_df.sort_values(by=['symbol', 'date'])


class preprocess(stocks_df):
    def __init__(self, symbol_list, start, end):
        super().__init__(symbol_list)
        self.close = pd.DataFrame(columns=['date'] + self.symbol_list)
        self.close['date'] = self.stocks_df['date'].drop_duplicates().reset_index(drop=True)
        for ticker in self.symbol_list:
            self.close.loc[:, ticker] = self.stocks_df[self.stocks_df['symbol'] == ticker]['close'].reset_index(
                drop=True)
        self.data_df = self.close.loc[(self.stocks_df['date'] >= np.datetime64(start))
                                      & (self.close['date'] <= np.datetime64(end))].reset_index(drop=True)

    @staticmethod
    def eval_ret_df(data_df, lag=1):
        ret_df = pd.DataFrame(columns=data_df.columns)
        ret_df['date'] = data_df['date'].values[:-lag]
        ret_df.iloc[:-1, 1:] = data_df.iloc[:-1, 1:] / data_df.iloc[:, 1:].shift(1) - 1
        return ret_df

    def plot_closed(self, symbol):
        msft = self.data_df.loc[:, symbol]
        plt.plot(self.data_df.loc[:, 'date'], msft, label='MSFT')


class Alpha(preprocess):
    def __init__(self, data):
        self.data = data
        self.date_vec = self.data['date'].drop_duplicates().reset_index(drop=True)

    def detect_first(self, option):
        if option == 'week':
            kq = [True]
            for i in range(len(self.date_vec.index) - 1):
                if self.date_vec[i].isocalendar()[1] != self.date_vec[i + 1].isocalendar()[1]:
                    kq += [True]
                else:
                    kq += [False]
        elif option == 'month':
            kq = [True]
            for i in range(len(self.date_vec.index) - 1):
                if self.date_vec[i].month != self.date_vec[i + 1].month:
                    kq += [True]
                else:
                    kq += [False]
        elif option == 'quarter':
            kq = [True]
            for i in range(len(self.date_vec.index) - 1):
                if self.date_vec[i].quarter != self.date_vec[i + 1].quarter:
                    kq += [True]
                else:
                    kq += [False]
        return kq

    def detect_last(self, option):
        if option == 'week':
            kq = []
            for i in range(len(self.date_vec.index) - 1):
                if self.date_vec[i].isocalendar()[1] != self.date_vec[i + 1].isocalendar()[1]:
                    kq += [True]
                else:
                    kq += [False]
        if kq[-1:] == [False]:
            kq += [True]
        else:
            kq += [False]
        return kq

    # At line 144, field cum_ret is calculated the same with the line 159, when calculate return of year (ret_year_df function)
    def daily_ret_alpha(self, alpha_df, lag=1):
        daily_ret_df = pd.DataFrame()
        daily_ret_df['date'] = alpha_df['date']
        daily_ret_df['trade'] = [0 if alpha_df.iloc[x, 1:].isnull().all() or alpha_df.iloc[x, 1:].sum() == 0 else 1 for
                                 x in alpha_df.index]
        alpha_df = alpha_df.fillna(0)
        ret_df = preprocess.eval_ret_df(self.data, lag)
        ret_df = ret_df[ret_df['date'] >= alpha_df['date'].values[0]]
        daily_ret_df['ret_alpha'] = pd.DataFrame(ret_df.values[:, 1:] * alpha_df.values[:-lag, 1:]).sum(axis=1)
        daily_ret_df['cum_ret'] = (daily_ret_df['ret_alpha'] + 1).cumprod() - 1
        return daily_ret_df

    @staticmethod
    # def ret_year_df(ret_alpha_df, ret_vnindex):
    #     ret_alpha_df['vnindex'] = ret_vnindex['^VNINDEX']
    def ret_year_df(ret_alpha_df):
        ret_year_df = {}
        temp_year = [i.year for i in ret_alpha_df['date']]
        year_list = list(sorted(set(temp_year)))
        old_lastday = 0
        for year in year_list:
            idx_lastday = max(i for i, e in enumerate(temp_year) if e == year)
            ret_year_df[year] = ret_alpha_df.loc[old_lastday:idx_lastday]
            old_lastday = idx_lastday + 1
            ret_year_df[year].loc[:, 'cum_ret'] = (ret_year_df[year].loc[:, 'ret_alpha'] + 1).cumprod() - 1
        return ret_year_df

    @staticmethod
    def max_dd(ret_alpha_df):
        cum_ret = ret_alpha_df['cum_ret']
        i = np.argmax(cum_ret.cummax() - cum_ret)  # end of the period
        j = np.argmax(cum_ret[:i])  # start of period
        if j == 0:
            max_dd = None
        else:
            max_dd = (cum_ret[j] - cum_ret[i]) / cum_ret[j]
        return max_dd

    def annual_geometric_return(self, ret_year_df, rf):
        annual_ret = pd.DataFrame(columns=['annual_ret', 'risk_free', 'std', 'sharpe'], \
                                  index=sorted(ret_year_df.keys()))
        if len(annual_ret) < len(rf):
            rf = rf[1:]
            print("Process data of one year!")
            print(ret_year_df.keys())

            if len(annual_ret) != len(rf):
                print("Risk free not match!")

        annual_ret.loc[:, 'risk_free'] = rf
        for year in annual_ret.index[1:]:
        # Change to calculate only for current year
        # for year in annual_ret.index:
            trading_date = ret_year_df[year]['cum_ret'].count()
            # annual_ret.loc[year,'annual_ret'] = (ret_year_df[year]['ret_alpha']+1).prod()**(max(252,trading_date)/trading_date)-1
            annual_ret.loc[year, 'annual_ret'] = (ret_year_df[year]['ret_alpha'] + 1).prod() - 1
            annual_ret.loc[year, 'std'] = ret_year_df[year].loc[:, 'ret_alpha'].std()
            annual_ret.loc[year, 'winning_rate'] = ret_year_df[year].loc[ret_year_df[year]['ret_alpha'] > 0].count()[
                                                       'ret_alpha'] / \
                                                   ret_year_df[year].loc[ret_year_df[year]['ret_alpha'] != 0].count()[
                                                       'ret_alpha']
            annual_ret.loc[year, 'sharpe'] = (
                                             annual_ret.loc[year, 'annual_ret'] - annual_ret.loc[year, 'risk_free']) / (
                                             np.sqrt(252) * annual_ret.loc[year, 'std'])
            annual_ret.loc[year, 'max_dd'] = self.max_dd(ret_year_df[year].reset_index(drop=True))
            # annual_ret.loc[year,'beta'] = ret_year_df[year].loc[:,'ret_alpha'].std()/ret_year_df[year].loc[:,'vnindex'].std()\
            # *ret_year_df[year].loc[:,'ret_alpha'].astype('float64').corr(ret_year_df[year].loc[:,'vnindex'].astype('float64'))
        return annual_ret

    @staticmethod
    def cum_twrr(cum_alpha):
        cum_alpha['cum_twrr'] = (cum_alpha['period_ret'] + 1).cumprod() - 1
        return cum_alpha

    @staticmethod
    def check_something(ret_alpha_df, alpha_df, cond, weight):
        checking_report = pd.DataFrame(columns=['total_trading', 'trade_dif_zero', 'num_tick_cond', 'count_weight'])
        total_trading = ret_alpha_df['trade'].count()
        trade_dif_zero = ret_alpha_df.loc[ret_alpha_df['trade'] != 0]['trade'].count()
        num_tick = len(alpha_df.iloc[:, 1:].columns) - (alpha_df.iloc[:, 1:] == 0).sum(axis=1)
        alpha_df.fillna(0)
        alpha_df.iloc[:, 1:] = alpha_df.iloc[:, 1:].apply(pd.to_numeric)
        num_tick_cond = num_tick[num_tick > cond].count()
        count_weight = (alpha_df.iloc[:, 1:] > weight).any(axis=1).sum()
        checking_report.append([total_trading, trade_dif_zero, num_tick_cond, count_weight])
        checking_report = checking_report.append(
            pd.DataFrame([[total_trading, trade_dif_zero, num_tick_cond, count_weight]], \
                         columns=checking_report.columns), ignore_index=True)
        return checking_report

    @staticmethod
    def plot(ret_alpha_df, money):
        cum_ret = ret_alpha_df['cum_ret'].reset_index(drop=True)
        cum_ret = (cum_ret + 1) * money
        i = np.argmax(cum_ret.cummax() - cum_ret)  # end of the period
        j = np.argmax(cum_ret[:i])
        plt.figure()
        plt.plot(ret_alpha_df['date'], cum_ret, label='cumulative time weighted rate of return')
        plt.plot([ret_alpha_df['date'][j], ret_alpha_df['date'][i]], [cum_ret[j], cum_ret[i]], 'o', color='Red',
                 markersize=10)
        plt.title('Cumulative time weighted rate of return and maximum drawdown')
        plt.legend()

    def cum_alpha(self, alpha_df):
        ret_norm = preprocess.eval_ret_df(self.data, lag=1).fillna(0)
        alpha_df = alpha_df.fillna(0)
        balance = []
        balance.append(((ret_norm.iloc[0, 1:] + 1) * alpha_df.iloc[:1, 1:]).sum(axis=1).sum())
        alpha_tb = {}
        for i in range(1, len(ret_norm['date'])):
            alpha_tb[0] = (ret_norm.iloc[0, 1:] + 1) * alpha_df.iloc[:1, 1:]
            alpha_tb[i] = alpha_tb[i - 1].append(alpha_df.iloc[i, 1:]) * (ret_norm.iloc[i, 1:] + 1)
            bal = alpha_tb[i].sum(axis=1).sum()
            balance.append(bal)
        balance_df = pd.DataFrame()
        balance_df['date'] = alpha_df['date'][:-1]
        balance_df['trade'] = alpha_df.sum(axis=1)[:-1]
        balance_df['money'] = balance_df['trade'].cumsum()
        balance_df['balance'] = balance
        return balance_df


class report(Alpha):
    def __init__(self, data, list_param, list_param_name):
        self.data = data
        self.list_param = list_param
        self.list_param_name = list_param_name

    def op_stop_loss(self, alpha_df, data, N_day, p, q):
        data_check = (1 - data.iloc[N_day:, 1:].values / data.iloc[:-N_day, 1:].values) > p
        alpha_check = alpha_df.iloc[N_day:, 1:].values > p
        mul_val = 1 - data_check * alpha_check * q

        # assign value
        date = (alpha_df.iloc[N_day:, 0]).reset_index(drop=True)
        values = pd.DataFrame(alpha_df.iloc[N_day:, 1:].values * mul_val)
        alpha_df_new = pd.concat([date, values], axis=1, ignore_index=True)
        alpha_df_new.columns = alpha_df.columns
        alpha_df_new = pd.concat([alpha_df.iloc[:N_day, :], alpha_df_new], axis=0)
        alpha_df_new.reset_index(drop=True)

        # normalize alpha
        alpha_df_new = opt.normalize(alpha_df_new)
        return alpha_df_new

    def op_take_profit(self, alpha_df, data, N_day, p, q):
        data_check = (data.iloc[N_day:, 1:].values / data.iloc[:-N_day, 1:].values - 1) > p
        alpha_check = alpha_df.iloc[N_day:, 1:].values > p
        mul_val = 1 - data_check * alpha_check * q

        # assign value
        # alpha_df.iloc[N_day:, 1:].values = alpha_df.iloc[N_day:, 1:].values * mul_val
        date = (alpha_df.iloc[N_day:, 0]).reset_index(drop=True)
        values = pd.DataFrame(alpha_df.iloc[N_day:, 1:].values * mul_val)
        alpha_df_new = pd.concat([date, values], axis=1, ignore_index=True)
        alpha_df_new.columns = alpha_df.columns
        alpha_df_new = pd.concat([alpha_df.iloc[:N_day, :], alpha_df_new], axis=0)
        alpha_df_new.reset_index(drop=True)

        # normalize alpha
        alpha_df_new = opt.normalize(alpha_df_new)
        return alpha_df_new


    def op_regulazation(self, alpha_df, data, config):
        alpha_ret = alpha_df
        if config[0]: # stop loss is used
            alpha_ret = self.op_stop_loss(alpha_ret, data,
                                          config[1], # N day of stop loss
                                          config[2], # p of stop loss
                                          config[3]) # q of stop loss

        if config[4]: # take profit is used
            alpha_ret = self.op_take_profit(alpha_ret, data,
                                          config[5], # Nday
                                          config[6], # p of take profit
                                          config[7]) # q of take profit

        return alpha_ret.reset_index(drop=True)

    def report_momentum_with_market(self, cur_rf):
        df = pd.DataFrame()
        for i in itertools.product(*self.list_param):
            try:
                print(list(i))
                annual_geometric_return = pd.DataFrame(columns=self.list_param_name[:-8])
                obj = momentum_with_market(self.data, *(i[:-8]))
                alpha_df = obj.get_alpha()
                alpha_df = self.op_regulazation(alpha_df=alpha_df, data=self.data, config=i[-8:])
                ret_alpha_df = obj.daily_ret_alpha(alpha_df, lag=1)
                ret_year_df = obj.ret_year_df(ret_alpha_df)
                annual_geometric_return = obj.annual_geometric_return(ret_year_df, cur_rf)

                # get annual_geometric return for only current year
                annual_geometric_return = annual_geometric_return[1:]
                annual_geometric_return = pd.concat([annual_geometric_return, pd.DataFrame(columns=self.list_param_name)])
                annual_geometric_return.loc[:, self.list_param_name] = list(i)

                df = pd.concat([df, annual_geometric_return])
            except:
                print("Error")
                continue
        return df

    def report_momentum_standard(self,cur_rf):
        df = pd.DataFrame()
        for i in itertools.product(*self.list_param):
            try:
                print(list(i))
                annual_geometric_return = pd.DataFrame(columns=self.list_param_name[:-8])
                obj = momentum_standard(self.data, *(i[:-8]))
                alpha_df = obj.get_alpha()
                alpha_df = self.op_regulazation(alpha_df=alpha_df, data=self.data, config=i[-8:])
                ret_alpha_df = obj.daily_ret_alpha(alpha_df, lag=1)
                ret_year_df = obj.ret_year_df(ret_alpha_df)
                annual_geometric_return = obj.annual_geometric_return(ret_year_df, cur_rf)
                # get annual_geometric return for only current year
                annual_geometric_return = annual_geometric_return[1:]
                annual_geometric_return = \
                    pd.concat([annual_geometric_return, pd.DataFrame(columns=self.list_param_name)])
                annual_geometric_return.loc[:, self.list_param_name] = list(i)
                df = pd.concat([df, annual_geometric_return])
            except:
                print("report_momentum_standard error!")
                continue
        return df

    def report_momentum_long_short(self, cur_rf):
        df = pd.DataFrame()
        for i in itertools.product(*self.list_param):
            try:
                print(list(i))
                annual_geometric_return = pd.DataFrame(columns=self.list_param_name[:-8])
                obj = momentum_long_short(self.data, *(i[:-8]))
                alpha_df = obj.get_alpha()
                alpha_df = self.op_regulazation(alpha_df=alpha_df, data=self.data, config=i[-8:])
                ret_alpha_df = obj.daily_ret_alpha(alpha_df, lag=1)
                ret_year_df = obj.ret_year_df(ret_alpha_df)
                annual_geometric_return = obj.annual_geometric_return(ret_year_df, cur_rf)
                # get annual_geometric return for only current year
                annual_geometric_return = annual_geometric_return[1:]
                annual_geometric_return = \
                    pd.concat([annual_geometric_return, pd.DataFrame(columns=self.list_param_name)])
                annual_geometric_return.loc[:, self.list_param_name] = list(i)
                df = pd.concat([df, annual_geometric_return])
            except:
                print("report_momentum_long_short error!")
                continue
        return df

    def report_momentum_volatility(self, cur_rf):
        df = pd.DataFrame()
        for i in itertools.product(*self.list_param):
            try:
                print(list(i))
                annual_geometric_return = pd.DataFrame(columns=self.list_param_name[:-8])
                obj = momentum_volatility(self.data, *(i[:-8]))
                alpha_df = obj.get_alpha()
                alpha_df = self.op_regulazation(alpha_df=alpha_df, data=self.data, config=i[-8:])
                ret_alpha_df = obj.daily_ret_alpha(alpha_df, lag=1)
                ret_year_df = obj.ret_year_df(ret_alpha_df)
                annual_geometric_return = obj.annual_geometric_return(ret_year_df, cur_rf)
                # get annual_geometric return for only current year
                annual_geometric_return = annual_geometric_return[1:]
                annual_geometric_return = \
                    pd.concat([annual_geometric_return, pd.DataFrame(columns=self.list_param_name)])
                annual_geometric_return.loc[:, self.list_param_name] = list(i)
                df = pd.concat([df, annual_geometric_return])
            except:
                print("report_momentum_volatility error!")
                continue
        return df

    @staticmethod
    def save_report(df, name, list_param_name):
        writer = pd.ExcelWriter(name + '.xlsx')
        c = list(set(df.columns) - set(list_param_name))
        df[c] = df[c].apply(pd.to_numeric, errors='coerce', axis=0)
        df.to_excel(writer, 'Sheet1')
        writer.save()

    @staticmethod
    def save_report_sum(df, name_sum, list_param_name):
        writer = pd.ExcelWriter(name_sum + '.xlsx', engine='xlsxwriter')
        c = list(set(df.columns) - set(list_param_name))
        df[c] = df[c].apply(pd.to_numeric, errors='coerce', axis=0)
        df_sum = df.groupby(list_param_name, as_index=False)[
            'annual_ret', 'max_dd', 'risk_free', 'sharpe', 'std', 'winning_rate'].mean()
        df_sum.to_excel(writer, 'Sheet1')
        writer.save()


class momentum_with_market(Alpha):
    def __init__(self, data, strat_type, alpha_type, option, market_return, mom_return, re_1, re_2):
        super().__init__(data)
        self.alpha_type = alpha_type
        self.option = option
        self.date_first = self.detect_first(self.option)
        self.market_return = market_return
        self.mom_return = mom_return
        self.re_1 = re_1
        self.re_2 = re_2
        self.alpha_df = self.data.copy()
        self.alpha_df.iloc[:, 1:] = float('nan')
        self.strat_type = strat_type

    def get_alpha(self):
        date_first_x_withdate = pd.concat([self.data, pd.Series(self.date_first)], axis=1, ignore_index=True)
        date_first_x_withdate.columns = list(self.data.columns) + ['bool']
        for i in range(self.market_return, len(date_first_x_withdate)):
            if date_first_x_withdate.loc[i, 'bool']:
                data = self.data.loc[self.data['date'] <= date_first_x_withdate.loc[i, 'date'], :].iloc[:, 1:]
                mar_ret = data.loc[i, :] / data.iloc[(i - self.market_return), :] - 1
                mar_ret = mar_ret.sum()
                mom_ret = pd.DataFrame()
                keep_stock = pd.DataFrame()
                if self.strat_type == '4.3':
                    if mar_ret > 0:
                        try:
                            mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                            mom_ret = mom_ret[mom_ret > 0]
                            num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re_1) + 1)
                            keep_stock = mom_ret.sort_values(ascending=False)[:num_pos_stock]
                        except:
                            print('4.3: mar_ret > 0 warning!')
                            continue
                    if mar_ret < 0:
                        try:
                            mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                            mom_ret = mom_ret[mom_ret < 0]
                            num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re_2) + 1)
                            keep_stock = mom_ret.sort_values(ascending=True)[:num_pos_stock]
                        except:
                            print('4.3: mar_ret < 0 warning!')
                            continue

                elif self.strat_type == '4.4':
                    if mar_ret > 0:
                        try:
                            mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                            mom_ret = mom_ret[mom_ret < 0]
                            num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re_1) + 1)
                            keep_stock = mom_ret.sort_values(ascending=True)[:num_pos_stock]
                        except:
                            print('4.4: mar_ret > 0 warning!')
                            continue
                    if mar_ret < 0:
                        try:
                            mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                            mom_ret = mom_ret[mom_ret > 0]
                            num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re_2) + 1)
                            keep_stock = mom_ret.sort_values(ascending=False)[:num_pos_stock]
                        except:
                            print('4.4: mar_ret < 0 warning!')
                            continue

                if self.alpha_type == 1:
                    self.alpha_df.loc[i, mom_ret.index] = 1
                elif self.alpha_type == 2:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret)
                elif self.alpha_type == 3:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret)
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 4:
                    self.alpha_df.loc[i, keep_stock.index] = 1
                elif self.alpha_type == 5:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock)
                elif self.alpha_type == 6:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock)
                    self.alpha_df = opt.rank(self.alpha_df)

        if self.date_first != 0:
            keep_alpha = self.alpha_df.iloc[0, 1:]
            for i in range(len(self.date_first)):
                if self.date_first[i]:
                    keep_alpha = self.alpha_df.iloc[i, 1:]
                else:
                    self.alpha_df.iloc[i, 1:] = keep_alpha
        self.alpha_df = opt.normalize(self.alpha_df)

        return self.alpha_df


class momentum_standard(Alpha):
    def __init__(self, data, strat_type, alpha_type, option, market_return, mom_return, re):
        super().__init__(data)
        self.alpha_type = alpha_type
        self.option = option
        self.date_first = self.detect_first(self.option)
        self.market_return = market_return
        self.mom_return = mom_return
        self.re = re
        self.alpha_df = self.data.copy()
        self.alpha_df.iloc[:, 1:] = float('nan')
        self.strat_type = strat_type

    def get_alpha(self):
        date_first_x_withdate = pd.concat([self.data['date'], pd.Series(self.date_first)], axis=1, ignore_index=True)
        date_first_x_withdate.columns = ['date', 'bool']
        for i in range(self.market_return, len(date_first_x_withdate)):
            if date_first_x_withdate.loc[i, 'bool']:
                data = self.data.loc[self.data['date'] <= date_first_x_withdate.loc[i, 'date'], :].iloc[:, 1:]
                mar_ret = data.loc[i, :] / data.iloc[(i - self.market_return), :] - 1
                mar_ret = mar_ret.sum()  # no need for this strategy
                mom_ret = pd.DataFrame()
                keep_stock = pd.DataFrame()
                if self.strat_type == '1.1_positive_score':
                    try:
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re) + 1)
                        keep_stock = mom_ret.sort_values(ascending=False)[:num_pos_stock]
                    except:
                        print('1.1_positive_score warning!')
                        continue
                elif self.strat_type == '1.1_positive_second_quintile':
                    try:
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.2) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.4) + 1)
                        keep_stock = mom_ret.sort_values(ascending=False)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.1_positive_second_quintile warning!')
                        continue
                elif self.strat_type == '1.1_positive_fourth_quintile':
                    try:
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.6) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.8) + 1)
                        keep_stock = mom_ret.sort_values(ascending=False)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.1_positive_fourth_quintile warning!')
                        continue
                elif self.strat_type == '1.1_negative_score':
                    try:
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret < 0]
                        num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re) + 1)
                        keep_stock = mom_ret.sort_values(ascending=True)[:num_pos_stock]
                    except:
                        print('1.1_negative_score warning!')
                        continue
                elif self.strat_type == '1.1_negative_second_quintile':
                    try:
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret < 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.2) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.4) + 1)
                        keep_stock = mom_ret.sort_values(ascending=True)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.1_negative_second_quintile warning!')
                        continue
                elif self.strat_type == '1.1_negative_fourth_quintile':
                    try:
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret < 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.6) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.8) + 1)
                        keep_stock = mom_ret.sort_values(ascending=True)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.1_negative_fourth_quintile warning!')
                        continue
                elif self.strat_type == '1.2_positive_score':
                    try:
                        mom_ret_1 = data.loc[i, :] / data.iloc[(i - self.mom_return), :]
                        mom_ret_2 = data.loc[i, :] / data.iloc[(i - self.mom_return - self.market_return), :]
                        mom_ret = mom_ret_1 - mom_ret_2
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re) + 1)
                        keep_stock = mom_ret.sort_values(ascending=False)[:num_pos_stock]
                    except:
                        print('1.2_positive_score warning!')
                        continue
                elif self.strat_type == '1.2_positive_second_quintile':
                    try:
                        mom_ret_1 = data.loc[i, :] / data.iloc[(i - self.mom_return), :]
                        mom_ret_2 = data.loc[i, :] / data.iloc[(i - self.mom_return - self.market_return), :]
                        mom_ret = mom_ret_1 - mom_ret_2
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.2) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.4) + 1)
                        keep_stock = mom_ret.sort_values(ascending=False)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.2_positive_second_quintile warning!')
                        continue
                elif self.strat_type == '1.2_positive_fourth_quintile':
                    try:
                        mom_ret_1 = data.loc[i, :] / data.iloc[(i - self.mom_return), :]
                        mom_ret_2 = data.loc[i, :] / data.iloc[(i - self.mom_return - self.market_return), :]
                        mom_ret = mom_ret_1 - mom_ret_2
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.6) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.8) + 1)
                        keep_stock = mom_ret.sort_values(ascending=False)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.2_positive_fourth_quintile warning!')
                        continue
                elif self.strat_type == '1.2_negative_score':
                    try:
                        mom_ret_1 = data.loc[i, :] / data.iloc[(i - self.mom_return), :]
                        mom_ret_2 = data.loc[i, :] / data.iloc[(i - self.mom_return - self.market_return), :]
                        mom_ret = mom_ret_1 - mom_ret_2
                        mom_ret = mom_ret[mom_ret < 0]
                        num_pos_stock = int(len(mom_ret) * 1.0 // (1.0 / self.re) + 1)
                        keep_stock = mom_ret.sort_values(ascending=True)[:num_pos_stock]
                    except:
                        print('1.2_negative_score warning!')
                        continue
                elif self.strat_type == '1.2_negative_second_quintile':
                    try:
                        mom_ret_1 = data.loc[i, :] / data.iloc[(i - self.mom_return), :]
                        mom_ret_2 = data.loc[i, :] / data.iloc[(i - self.mom_return - self.market_return), :]
                        mom_ret = mom_ret_1 - mom_ret_2
                        mom_ret = mom_ret[mom_ret < 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.2) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.4) + 1)
                        keep_stock = mom_ret.sort_values(ascending=True)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.2_negative_second_quintile warning!')
                        continue
                elif self.strat_type == '1.2_negative_fourth_quintile':
                    try:
                        mom_ret_1 = data.loc[i, :] / data.iloc[(i - self.mom_return), :]
                        mom_ret_2 = data.loc[i, :] / data.iloc[(i - self.mom_return - self.market_return), :]
                        mom_ret = mom_ret_1 - mom_ret_2
                        mom_ret = mom_ret[mom_ret < 0]
                        num_pos_stock_1 = int(len(mom_ret) * 1.0 // (1.0 / 0.6) + 1)
                        num_pos_stock_2 = int(len(mom_ret) * 1.0 // (1.0 / 0.8) + 1)
                        keep_stock = mom_ret.sort_values(ascending=True)[num_pos_stock_1:num_pos_stock_2]
                    except:
                        print('1.2_negative_fourth_quintile warning!')
                        continue
                if self.alpha_type == 1:
                    self.alpha_df.loc[i, mom_ret.index] = 1
                elif self.alpha_type == 2:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret)
                elif self.alpha_type == 3:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret)
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 4:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / np.abs(mom_ret)
                elif self.alpha_type == 5:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / np.abs(mom_ret)
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 6:
                    self.alpha_df.loc[i, keep_stock.index] = 1
                elif self.alpha_type == 7:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock)
                elif self.alpha_type == 8:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock)
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 9:
                    self.alpha_df.loc[i, keep_stock.index] = 1 / np.abs(keep_stock)
                elif self.alpha_type == 10:
                    self.alpha_df.loc[i, keep_stock.index] = 1 / np.abs(keep_stock)
                    self.alpha_df = opt.rank(self.alpha_df)

        if self.date_first != 0:
            keep_alpha = self.alpha_df.iloc[0, 1:]
            for i in range(len(self.date_first)):
                if self.date_first[i]:
                    keep_alpha = self.alpha_df.iloc[i, 1:]
                else:
                    self.alpha_df.iloc[i, 1:] = keep_alpha
        self.alpha_df = opt.normalize(self.alpha_df)

        return self.alpha_df


class momentum_long_short(Alpha):
    def __init__(self, data, strat_type, alpha_type, option, short, long):
        super().__init__(data)
        self.alpha_type = alpha_type
        self.option = option
        self.date_first = self.detect_first(self.option)
        self.long = long
        self.short = short
        self.alpha_df = self.data.copy()
        self.alpha_df.iloc[:, 1:] = float('nan')
        self.strat_type = strat_type

    def get_alpha(self):
        date_first_x_withdate = pd.concat([self.data['date'], pd.Series(self.date_first)], axis=1, ignore_index=True)
        date_first_x_withdate.columns = ['date', 'bool']
        for i in range(self.short, len(date_first_x_withdate)):
            if date_first_x_withdate.loc[i, 'bool']:
                data = self.data.loc[self.data['date'] <= date_first_x_withdate.loc[i, 'date'], :].iloc[:, 1:]
                mom_ret = pd.DataFrame(columns=['mom_ret_s', 'mom_ret_l'])

                if self.strat_type == 'positive_short_positive_long':
                    try:
                        mom_ret['mom_ret_s'] = data.loc[i, :] / data.iloc[(i - self.short), :] - 1
                        mom_ret['mom_ret_l'] = data.loc[i, :] / data.iloc[(i - self.long), :] - 1
                        mom_ret = mom_ret.loc[(mom_ret['mom_ret_s'] > 0) & (mom_ret['mom_ret_l'] > 0)]
                    except:
                        print('positive_short_positive_long warning!')
                        continue
                if self.strat_type == 'positive_short_negative_long':
                    try:
                        mom_ret['mom_ret_s'] = data.loc[i, :] / data.iloc[(i - self.short), :] - 1
                        mom_ret['mom_ret_l'] = data.loc[i, :] / data.iloc[(i - self.long), :] - 1
                        mom_ret = mom_ret.loc[(mom_ret['mom_ret_s'] > 0) & (mom_ret['mom_ret_l'] < 0)]
                    except:
                        print('positive_short_negative_long warning!')
                        continue
                if self.strat_type == 'negative_short_positive_long':
                    try:
                        mom_ret['mom_ret_s'] = data.loc[i, :] / data.iloc[(i - self.short), :] - 1
                        mom_ret['mom_ret_l'] = data.loc[i, :] / data.iloc[(i - self.long), :] - 1
                        mom_ret = mom_ret.loc[(mom_ret['mom_ret_s'] < 0) & (mom_ret['mom_ret_l'] > 0)]
                    except:
                        print('negative_short_positive_long warning!')
                        continue
                if self.strat_type == 'negative_short_negative_long':
                    try:
                        mom_ret['mom_ret_s'] = data.loc[i, :] / data.iloc[(i - self.short), :] - 1
                        mom_ret['mom_ret_l'] = data.loc[i, :] / data.iloc[(i - self.long), :] - 1
                        mom_ret = mom_ret.loc[(mom_ret['mom_ret_s'] < 0) & (mom_ret['mom_ret_l'] < 0)]
                    except:
                        print('negative_short_negative_long warning!')
                        continue

                if self.alpha_type == 1:
                    self.alpha_df.loc[i, mom_ret.index] = 1
                elif self.alpha_type == 2:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_s'])
                elif self.alpha_type == 3:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_s'])
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 4:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / np.abs(mom_ret['mom_ret_s'])
                elif self.alpha_type == 5:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / np.abs(mom_ret['mom_ret_s'])
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 6:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_l'])
                elif self.alpha_type == 7:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_l'])
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 8:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / np.abs(mom_ret['mom_ret_l'])
                elif self.alpha_type == 9:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / np.abs(mom_ret['mom_ret_l'])
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 10:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_l']) + np.abs(mom_ret['mom_ret_s'])
                elif self.alpha_type == 11:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_l']) + np.abs(mom_ret['mom_ret_s'])
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 12:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / (
                    np.abs(mom_ret['mom_ret_l']) + np.abs(mom_ret['mom_ret_s']))
                elif self.alpha_type == 13:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / (
                    np.abs(mom_ret['mom_ret_l']) + np.abs(mom_ret['mom_ret_s']))
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 14:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_l']) * np.abs(mom_ret['mom_ret_s'])
                elif self.alpha_type == 15:
                    self.alpha_df.loc[i, mom_ret.index] = np.abs(mom_ret['mom_ret_l']) * np.abs(mom_ret['mom_ret_s'])
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 16:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / (
                    np.abs(mom_ret['mom_ret_l']) * np.abs(mom_ret['mom_ret_s']))
                elif self.alpha_type == 17:
                    self.alpha_df.loc[i, mom_ret.index] = 1 / (
                    np.abs(mom_ret['mom_ret_l']) * np.abs(mom_ret['mom_ret_s']))
                    self.alpha_df = opt.rank(self.alpha_df)

        if self.date_first != 0:
            keep_alpha = self.alpha_df.iloc[0, 1:]
            for i in range(len(self.date_first)):
                if self.date_first[i]:
                    keep_alpha = self.alpha_df.iloc[i, 1:]
                else:
                    self.alpha_df.iloc[i, 1:] = keep_alpha
        self.alpha_df = opt.normalize(self.alpha_df)

        return self.alpha_df

        # parameter momentum_long_short


class momentum_volatility(Alpha):
    def __init__(self, data, strat_type, alpha_type, option, option_vol, no_week, mom_return, re_vol, re_mom):
        super().__init__(data)
        self.alpha_type = alpha_type
        self.option = option
        self.option_vol = option_vol
        self.date_first = self.detect_first(self.option)
        self.date_first_vol = self.detect_first(self.option_vol)
        self.no_week = no_week
        self.mom_return = mom_return
        self.re_mom = re_mom
        self.re_vol = re_vol
        self.alpha_df = self.data.copy()
        self.alpha_df.iloc[:, 1:] = float('nan')
        self.strat_type = strat_type

    def get_alpha(self):
        date_first_x_withdate = pd.concat([self.data, pd.Series(self.date_first)], axis=1, ignore_index=True)
        date_first_x_withdate.columns = list(self.data.columns) + ['bool']
        weekly_return = pd.DataFrame(columns=date_first_x_withdate.columns)
        weekly_return['date'] = date_first_x_withdate['date']
        weekly_return['bool'] = pd.Series(self.date_first_vol)
        weekly_return.iloc[:, 1:-1] = date_first_x_withdate.iloc[:, 1:-1] / date_first_x_withdate.iloc[:, 1:-1].shift(
            5) - 1
        # Rolling weekly standard deviation
        weekly_std = weekly_return.copy()
        weekly_std.iloc[:, 1:-1] = weekly_return.iloc[:, 1:-1].rolling(self.no_week).std()
        for i in range(1, len(weekly_std)):
            if weekly_std.loc[i, 'bool'] == False:
                weekly_std.iloc[i, 1:-1] = weekly_std.iloc[i - 1, 1:-1]
        weekly_std = weekly_std.iloc[:, 1:-1]

        for i in range(5, len(date_first_x_withdate)):
            if date_first_x_withdate.loc[i, 'bool']:
                data = self.data.loc[self.data['date'] <= date_first_x_withdate.loc[i, 'date'], :].iloc[:, 1:]
                keep_stock_vol = pd.DataFrame()
                keep_stock_mom = pd.DataFrame()
                mom_ret = pd.DataFrame()
                keep_stock = pd.DataFrame()
                if self.strat_type == 'high_volatility':
                    try:
                        num_pos_stock_vol = int(len(weekly_std.loc[i, :]) * 1.0 // (1.0 / self.re_vol) + 1)
                        keep_stock_vol = weekly_std.loc[i, :].sort_values(ascending=False)[:num_pos_stock_vol]
                    except:
                        print('high_volatility warning!')
                        continue
                if self.strat_type == 'low_volatility':
                    try:
                        num_pos_stock_vol = int(len(weekly_std.loc[i, :]) * 1.0 // (1.0 / self.re_vol) + 1)
                        keep_stock_vol = weekly_std.loc[i, :].sort_values(ascending=True)[:num_pos_stock_vol]
                    except:
                        print('low_volatility warning!')
                        continue
                if self.strat_type == 'high_volatility_high_momentum':
                    try:
                        num_pos_stock_vol = int(len(weekly_std.loc[i, :]) * 1.0 // (1.0 / self.re_vol) + 1)
                        keep_stock_vol = weekly_std.loc[i, :].sort_values(ascending=False)[:num_pos_stock_vol]
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_mom = int(len(mom_ret) * 1.0 // (1.0 / self.re_mom) + 1)
                        keep_stock_mom = mom_ret.sort_values(ascending=False)[:num_pos_stock_mom]
                        keep_stock = pd.concat([keep_stock_vol, keep_stock_mom], join='inner', axis=1)
                    except:
                        print('high_volatility_high_momentum warning!')
                        continue
                if self.strat_type == 'high_volatility_low_momentum':
                    try:
                        num_pos_stock_vol = int(len(weekly_std.loc[i, :]) * 1.0 // (1.0 / self.re_vol) + 1)
                        keep_stock_vol = weekly_std.loc[i, :].sort_values(ascending=False)[:num_pos_stock_vol]
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_mom = int(len(mom_ret) * 1.0 // (1.0 / self.re_mom) + 1)
                        keep_stock_mom = mom_ret.sort_values(ascending=True)[:num_pos_stock_mom]
                        keep_stock = pd.concat([keep_stock_vol, keep_stock_mom], join='inner', axis=1)
                    except:
                        print('high_volatility_low_momentum warning!')
                        continue
                if self.strat_type == 'low_volatility_high_momentum':
                    try:
                        num_pos_stock_vol = int(len(weekly_std.loc[i, :]) * 1.0 // (1.0 / self.re_vol) + 1)
                        keep_stock_vol = weekly_std.loc[i, :].sort_values(ascending=True)[:num_pos_stock_vol]
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_mom = int(len(mom_ret) * 1.0 // (1.0 / self.re_mom) + 1)
                        keep_stock_mom = mom_ret.sort_values(ascending=False)[:num_pos_stock_mom]
                        keep_stock = pd.concat([keep_stock_vol, keep_stock_mom], join='inner', axis=1)
                    except:
                        print('low_volatility_high_momentum warning!')
                        continue
                if self.strat_type == 'low_volatility_low_momentum':
                    try:
                        num_pos_stock_vol = int(len(weekly_std.loc[i, :]) * 1.0 // (1.0 / self.re_vol) + 1)
                        keep_stock_vol = weekly_std.loc[i, :].sort_values(ascending=True)[:num_pos_stock_vol]
                        mom_ret = data.loc[i, :] / data.iloc[(i - self.mom_return), :] - 1
                        mom_ret = mom_ret[mom_ret > 0]
                        num_pos_stock_mom = int(len(mom_ret) * 1.0 // (1.0 / self.re_mom) + 1)
                        keep_stock_mom = mom_ret.sort_values(ascending=True)[:num_pos_stock_mom]
                        keep_stock = pd.concat([keep_stock_vol, keep_stock_mom], join='inner', axis=1)
                    except:
                        print('low_volatility_low_momentum warning!')
                        continue
                if self.alpha_type == 1:
                    self.alpha_df.loc[i, keep_stock_vol.index] = 1
                elif self.alpha_type == 2:
                    self.alpha_df.loc[i, keep_stock_vol.index] = np.abs(keep_stock_vol)
                elif self.alpha_type == 3:
                    self.alpha_df.loc[i, keep_stock_vol.index] = np.abs(keep_stock_vol)
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 4:
                    self.alpha_df.loc[i, keep_stock_vol.index] = 1 / np.abs(keep_stock_vol)
                elif self.alpha_type == 5:
                    self.alpha_df.loc[i, keep_stock_vol.index] = 1 / np.abs(keep_stock_vol)
                    self.alpha_df = opt.rank(self.alpha_df)
                elif self.alpha_type == 6:
                    self.alpha_df.loc[i, keep_stock.index] = 1
                elif self.alpha_type == 7:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock.loc[keep_stock.index])
                elif self.alpha_type == 8:
                    self.alpha_df.loc[i, keep_stock.index] = 1 / np.abs(keep_stock_vol.loc[keep_stock.index])
                elif self.alpha_type == 9:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock_mom.loc[keep_stock.index])
                elif self.alpha_type == 10:
                    self.alpha_df.loc[i, keep_stock.index] = 1 / np.abs(keep_stock_mom.loc[keep_stock.index])
                elif self.alpha_type == 11:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock_mom.loc[keep_stock.index]) + \
                                                             np.abs(keep_stock_vol.loc[keep_stock.index])
                elif self.alpha_type == 12:
                    self.alpha_df.loc[i, keep_stock.index] = 1 / (np.abs(keep_stock_mom.loc[keep_stock.index]) \
                                                                  + np.abs(keep_stock_vol.loc[keep_stock.index]))
                elif self.alpha_type == 13:
                    self.alpha_df.loc[i, keep_stock.index] = np.abs(keep_stock_mom.loc[keep_stock.index]) \
                                                             * np.abs(keep_stock_vol.loc[keep_stock.index])
                elif self.alpha_type == 14:
                    self.alpha_df.loc[i, keep_stock.index] = 1 / (np.abs(keep_stock_mom.loc[keep_stock.index]) \
                                                                  * np.abs(keep_stock_vol.loc[keep_stock.index]))

        if self.date_first != 0:
            keep_alpha = self.alpha_df.iloc[0, 1:]
            for i in range(len(self.date_first)):
                if self.date_first[i]:
                    keep_alpha = self.alpha_df.iloc[i, 1:]
                else:
                    self.alpha_df.iloc[i, 1:] = keep_alpha
        self.alpha_df = opt.normalize(self.alpha_df)
        return self.alpha_df