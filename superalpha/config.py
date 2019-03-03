symbol = \
    {2009 : ['STB', 'DPM', 'HPG', 'SSI', 'FPT', 'PVD', 'VPL', 'VNM', 'SJS', 'SAM', 'VIC', 'REE', 'FPC', 'PPC', 'ITA', 'VTO', 'TAC', 'DHG', 'GMD', 'DQC', 'VSH', 'TDH', 'BMC', 'KDC', 'VIP', 'DPR', 'PVT', 'PET', 'SGT', 'ANV'],
    2010 : ['BCI', 'CII', 'DPM', 'DRC', 'FPT', 'HAG', 'HCM', 'HPG', 'HSG', 'ITA', 'LCG', 'LSS', 'NBB', 'NTL', 'PET', 'PVD', 'PVF', 'PVT', 'SBT', 'SJS', 'SSI', 'STB', 'TDH', 'VCB', 'VIC', 'VIP', 'VIS', 'VNM', 'VSH', 'VTO'],
    2011 : ['GMD', 'CII', 'DIG', 'DPM', 'EIB', 'FPT', 'KDC', 'HAG', 'HCM', 'HPG', 'HSG', 'PNJ', 'ITA', 'ITC', 'KBC', 'NTL', 'PVD', 'PVF', 'REE', 'SBT', 'SJS', 'SSI', 'STB', 'OGC', 'VCB', 'VIC', 'CTG', 'VNM', 'VSH', 'BVH'],
    2012 : ['VCB', 'CII', 'CTG', 'DIG', 'DPM', 'EIB', 'VIC', 'FPT', 'GMD', 'HAG', 'HPG', 'VNM', 'HVG', 'IJC', 'ITA', 'KDC', 'KDH', 'MSN', 'OGC', 'PNJ', 'PVD', 'PVF', 'QCG', 'STB', 'REE', 'SBT', 'VSH', 'SJS', 'SSI', 'BVH'],
    2013 : ['VCB', 'CII', 'CSM', 'CTG', 'DIG', 'DPM', 'VIC', 'DRC', 'EIB', 'FPT', 'GMD', 'VNM', 'HAG', 'HPG', 'HSG', 'IJC', 'KDC', 'MBB', 'MSN', 'OGC', 'PGD', 'PNJ', 'PVD', 'STB', 'PVF', 'REE', 'VSH', 'SBT', 'SSI', 'BVH'],
    2014 : ['VCB', 'CII', 'CSM', 'CTG', 'DPM', 'DRC', 'VIC', 'EIB', 'FPT', 'GMD', 'HAG', 'VNM', 'HPG', 'HSG', 'IJC', 'ITA', 'KDC', 'MBB', 'MSN', 'OGC', 'PET', 'PGD', 'PPC', 'STB', 'PVD', 'PVT', 'VSH', 'REE', 'SSI', 'BVH'],
    2015 : ['BVH', 'CII', 'CSM', 'CTG', 'DPM', 'DRC', 'EIB', 'FLC', 'FPT', 'GMD', 'HAG', 'HCM', 'HPG', 'HSG', 'IJC', 'ITA', 'KDC', 'MBB', 'MSN', 'OGC', 'PPC', 'PVD', 'PVT', 'REE', 'SSI', 'STB', 'VCB', 'VIC', 'VNM', 'VSH'],
    2016 : ['BVH', 'CII', 'CSM', 'CTG', 'DPM', 'EIB', 'FLC', 'FPT', 'GMD', 'HAG', 'HCM', 'HHS', 'HPG', 'HSG', 'HVG', 'ITA', 'KBC', 'KDC', 'MBB', 'MSN', 'PPC', 'PVD', 'PVT', 'REE', 'SSI', 'STB', 'VCB', 'VIC', 'VNM', 'VSH'],
    2017 : ['BID', 'BVH', 'CII', 'CTG', 'DPM', 'FLC', 'FPT', 'GAS', 'GMD', 'HAG', 'HCM', 'HNG', 'HPG', 'HSG', 'ITA', 'KBC', 'KDC', 'MBB', 'MSN', 'MWG', 'NT2', 'PPC', 'PVD', 'REE', 'SBT', 'SSI', 'STB', 'VCB', 'VIC', 'VNM'],
    2018 : ['BID', 'BMP', 'BVH', 'CII', 'CTD', 'CTG', 'DHG', 'DPM', 'FPT', 'GAS', 'GMD', 'HPG', 'HSG', 'KBC', 'KDC', 'MBB', 'MSN', 'MWG', 'NT2', 'NVL', 'PVD', 'REE', 'ROS', 'SAB', 'SBT', 'SSI', 'STB', 'VCB', 'VIC', 'VNM']}


rf = [ 0.12, 0.12, 0.12, 0.12, 0.12, 0.0995, 0.0891, 0.07157, 0.07095, 0.06010, 0.0455] #risk-free rate

df_report = [
    {'name':'_momentum_with_market', 'df':None, 'name_sum':'_momentum_with_market_sum', 'list_param_name': ['strat_type','alpha_type','option', 'market_return','mom_return','re_1', 're_2', 'stop_loss', 'N_day_sl', 'p_sl', 'q_sl', 'take_profit', 'N_day_tp', 'p_tp', 'q_tp'], 'output': None},
    {'name':'_momentum_standard', 'df':None, 'name_sum':'_momentum_standard_sum', 'list_param_name': ['strat_type','alpha_type','option', 'market_return','mom_return','re', 'stop_loss', 'N_day_sl', 'p_sl', 'q_sl', 'take_profit', 'N_day_tp', 'p_tp', 'q_tp'], 'output': None},
    {'name':'_momentum_long_short', 'df':None, 'name_sum':'_momentum_long_short_sum', 'list_param_name': ['strat_type','alpha_type','option', 'short','long', 'stop_loss', 'N_day_sl', 'p_sl', 'q_sl', 'take_profit', 'N_day_tp', 'p_tp', 'q_tp'], 'output': None},
    {'name':'_momentum_volatility_1', 'df':None, 'name_sum':'_momentum_volatility_1_sum', 'list_param_name': ['strat_type','alpha_type','option','option_vol', 'no_week','mom_return','re_vol','re_mom', 'stop_loss', 'N_day_sl', 'p_sl', 'q_sl', 'take_profit', 'N_day_tp', 'p_tp', 'q_tp'], 'output': None},
    {'name':'_momentum_volatility_2', 'df':None, 'name_sum':'_momentum_volatility_2_sum', 'list_param_name': ['strat_type','alpha_type','option','option_vol', 'no_week','mom_return','re_vol','re_mom', 'stop_loss', 'N_day_sl', 'p_sl', 'q_sl', 'take_profit', 'N_day_tp', 'p_tp', 'q_tp'], 'output': None}]

df_ret_alpha = None

back_track_window = 250

RUN = [False, False , False, False, True, False]

regulazation = {
    'stop_loss' : {
        'N_day':[20],
        'p':[0.2, 0.3],
        'q':[0.2],
        'is_use': [True]
    },
    'take_profit' :{
        'N_day':[20],
        'p':[0.2],
        'q':[0.2],
        'is_use': [True]
    }
}

options = ['month', 'quater']
M_days = [10]





