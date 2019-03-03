import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
import time
import os
import ProjectX.superalpha.config as cf

if __name__ == "__main__":
    path = 'results/'
    alpha_df_ret = {}
    ii = 4

    mom_rets = [5, 25, 30, 50, 90, 100, 150]

    for jj in list(cf.symbol):
        alpha_df = None
        alpha_df_sum = None
        for mom_ret in mom_rets:
            file_name = path + str(jj) + '_monret_' + str(mom_ret) + '_' + cf.df_report[ii]['name'] + '.xlsx'
            file_name_sum = path + str(jj) + '_monret_' + str(mom_ret) + '_' + cf.df_report[ii]['name_sum'] + '.xlsx'
            if not os.path.isfile(file_name):
                print('File %s not exist!' % file_name)
                break

            alpha_df_jj = pd.read_excel(file_name)
            alpha_df_jj_sum = pd.read_excel(file_name_sum)

            if alpha_df is None:
                alpha_df = alpha_df_jj
                alpha_df_sum = alpha_df_jj_sum
            else:
                alpha_df = pd.concat([alpha_df, alpha_df_jj])
                alpha_df_sum = pd.concat([alpha_df_sum, alpha_df_jj_sum])

            print(file_name)

        if alpha_df is not None:
            # save to file
            alpha_df = alpha_df.sort_index()
            alpha_df_sum = alpha_df_sum.sort_index()

            writer = pd.ExcelWriter(path + str(jj) + cf.df_report[ii]['name'] + '.xlsx',
                                    engine='xlsxwriter')
            writer_sum = pd.ExcelWriter(path + str(jj) + cf.df_report[ii]['name_sum'] + '.xlsx',
                                    engine='xlsxwriter')
            alpha_df.to_excel(writer, 'Sheet1')
            alpha_df_sum.to_excel(writer_sum, 'Sheet1')
            writer.save()
            writer_sum.save()

