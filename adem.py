from __future__ import division
from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from helper import pickle_store, pickle_restore

style.use('fivethirtyeight')

ins_ans_lst = pickle_restore("pickle/insider_list_file")
# start_d, end_d = '01-2-2010', '04-2-2010'
start_d, end_d = '06-1-2010', '12-31-2010'

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def explain_anomalies(baseline_y, y, window_size, sigma=3.0):
    avg = moving_average(y, window_size).tolist() #y.rolling(window_size).mean() #
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    ev = {'standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(y.index, y, avg) if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}
    return ev

    # return {'standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(count(), y, avg) if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def explain_anomalies_rolling_std(x, y, window_size, sigma=3.0):
    avg = moving_average(y, window_size) #y.rolling(window_size).mean()
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = residual.rolling(30).std()#pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    # rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.loc[window_size - 1]).round(3).iloc[:,0].tolist()
    rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.loc[testing_std_as_df.first_valid_index()][0]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    # ev = {'stationary standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i, rs_i in zip(count(), y, avg_list, rolling_std) if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}
    ev = {'stationary standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i, rs_i in zip(y.index, y, avg_list, rolling_std) if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}

    return ev

def plot_res (events, Y, baseline_type='UBP', feature_type='email_frequency', user_name='AAM0658'):
    sdf = pd.Series(events['anomalies_dict'], name=baseline_type+'anomaly')

    if len(sdf) == 0:
        return


    #plot the data
    plt.figure()
    Y.plot(title=user_name+' '+feature_type+' '+baseline_type, legend=True, figsize=(18,10), label=baseline_type+' normal')
    ax = sdf.plot(marker="*",  markersize=12, markerfacecolor="r", legend=True, label=baseline_type+' anomaly')
    ax.set_ylabel(feature_type)
    plt.legend(loc='best')
    plt.xticks(rotation=80)

    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    # plt.show()
    plt.savefig('plots/'+user_name+baseline_type+feature_type+'.png')
    plt.close()



    #########################################function for Cmnty write below

# def moving_average(data, window_size):
#     window = np.ones(int(window_size))/float(window_size)
#     return np.convolve(data, window, 'same')

def explain_anomalies_cmnty(baseline_y, y, window_size, sigma=3.0):
    avg = moving_average(baseline_y, window_size).tolist() #y.rolling(window_size).mean() #
    residual = baseline_y - avg

    # print (residual)

    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    # print (std)
    ev = {'standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(y.index, y, avg) if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}
    return ev

    # return {'standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(count(), y, avg) if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def explain_anomalies_rolling_std_cmnty(x, y, window_size, sigma=3.0):
    avg = moving_average(y, window_size) #y.rolling(window_size).mean()
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = residual.rolling(30).std()#pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    # rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.loc[window_size - 1]).round(3).iloc[:,0].tolist()
    rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.loc[testing_std_as_df.first_valid_index()][0]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    # ev = {'stationary standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i, rs_i in zip(count(), y, avg_list, rolling_std) if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}
    ev = {'stationary standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i, rs_i in zip(y.index, y, avg_list, rolling_std) if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}

    return ev

def plot_res_cmnty (events, baseline_X,  Y, baseline_type='UBP', feature_type='email_frequency', user_name='AAM0658'):
    sdf = pd.Series(events['anomalies_dict'], name=baseline_type+'anomaly')
    
    if len(sdf) == 0:
        return

    #plot the data
    plt.figure()
    Y.plot(title=user_name+' '+feature_type+' '+baseline_type, legend=True, figsize=(18,8), label=baseline_type+' usr normal')
    baseline_X.plot(title=user_name+' '+feature_type+' '+baseline_type, legend=True, figsize=(18,10), label=baseline_type+' cmnty normal')
    
    ax = sdf.plot(marker="*",  markersize=12, markerfacecolor="r", legend=True, label=baseline_type+' anomaly')
    ax.set_ylabel(feature_type)
    plt.legend(loc='best')
    plt.xticks(rotation=80)

    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    # plt.show()
    plt.savefig('plots/'+user_name+baseline_type+feature_type+'.png')
    plt.close()






    #########################################################################################################################################################################

    #UBP anom calculation and plot generation:
# 1. Take prepared feature dataframe and 
# 2. index by date
# 3. populate missing dates
# 4. get X and Y and call ADEM.

def anom_calc_ubp(df, feature_name, uname, ws=10, sig=3, fti='f1'):
    # index email data by date
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    #Fill the missing dates and corresponding feature values as 0.
    # df = df.resample('D').sum().fillna(0) # this doesnt work if only one date is present. Only fills date witin min and max date already present.

    rng = pd.date_range(start_d, end_d, freq='D')
    df = df.reindex(rng, fill_value=0)



    X = df.index.to_series() #debug_df['date'] # data_as_frame['Months']
    Y = df[feature_name] #data_as_frame['SunSpots']

    # # plot the results
    # plot_results(X, y=Y, window_size=20, text_xlabel="Months", sigma_value=3, text_ylabel="No. of Sun spots")

    events = explain_anomalies(X, Y, window_size=ws, sigma=sig)
    # Display the anomaly dict
    # print("Information about the anomalies model:{}".format(events))
    # print('num_anom=', len(events['anomalies_dict']))

    # events_rolling = explain_anomalies_rolling_std(X, Y, window_size=ws, sigma=sig)
    
    # Display the anomaly dict
    # print("Information about the anomalies model rolling:{}".format(events_rolling))
    # print()

    #####################################################################################################
    # if uname in ins_ans_lst:
    #     plot_res (events, Y, baseline_type='UBP', feature_type=feature_name+fti, user_name=uname)
        # plot_res (events_rolling, Y, baseline_type='UBP-rolling', feature_type=feature_name, user_name=uname)
    return events
    # return events_rolling

# email_freq_feature_dic.keys()
# Feature data format:  user: df [user, date, featurecolmn]
#This is for intrcmnty and intra cmnty
def anom_calc_cbp(df, df_cmnty, feature_name, uname, ws=10, sig=3, fti='f1'):
    df_cmnty['date'] = pd.to_datetime(df_cmnty['date'])
    df_cmnty.set_index('date', inplace=True)
    #Fill the missing dates and corresponding feature values as 0.
    # df_cmnty = df_cmnty.resample('D').sum().fillna(0)
    rng = pd.date_range(start_d, end_d, freq='D')
    df_cmnty = df_cmnty.reindex(rng, fill_value=0)



    baseline_df = df_cmnty
    
    # def anom_calc_cbp(df, baseline_df, feature_name, uname):
    # index email data by date
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    #Fill the missing dates and corresponding feature values as 0.
    # df = df.resample('D').sum().fillna(0)
    rng = pd.date_range(start_d, end_d, freq='D')
    df = df.reindex(rng, fill_value=0)


    #assuming baseline_x is date indexed and missed dates already filled
    X = baseline_df[feature_name] #debug_df['date'] # data_as_frame['Months']
    Y = df[feature_name] #data_as_frame['SunSpots']

    # # plot the results
    # plot_results(X, y=Y, window_size=20, text_xlabel="Months", sigma_value=3, text_ylabel="No. of Sun spots")

    events = explain_anomalies_cmnty(X, Y, window_size=ws, sigma=sig)
    # Display the anomaly dict
    # print("Information about the anomalies model:{}".format(events))
    # print('num_anom=', len(events['anomalies_dict']))


    # events_rolling = explain_anomalies_rolling_std_cmnty(X, Y, window_size=ws, sigma=sig)
    # # Display the anomaly dict
    # print("Information about the anomalies model rolling:{}".format(events_rolling))
    # print()

    #####################################################################################################
    # if uname in ins_ans_lst:
    #     plot_res_cmnty (events, X, Y, baseline_type='CBP', feature_type=feature_name+fti, user_name=uname)
        # plot_res (events_rolling, Y, baseline_type='UBP-rolling', feature_type=feature_name)
    return events
    # return events_rolling

# email_freq_feature_dic.keys()
# Feature data format:  user: df [user, date, featurecolmn]
def anom_calc_pbp(df, df_peer, feature_name, uname, ws=10, sig=3, fti='f1'):
    df_peer['date'] = pd.to_datetime(df_peer['date'])
    df_peer.set_index('date', inplace=True)
    #Fill the missing dates and corresponding feature values as 0.
    # df_peer = df_peer.resample('D').sum().fillna(0)
    rng = pd.date_range(start_d, end_d, freq='D')
    df_peer = df_peer.reindex(rng, fill_value=0)

    # anom_calc_cbp(df, df_cmnty, 'email_freq', 'AAM0658')

    baseline_df = df_peer

    # def anom_calc_cbp(df, baseline_df, feature_name, uname):
    # index email data by date
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    #Fill the missing dates and corresponding feature values as 0.
    # df = df.resample('D').sum().fillna(0)
    rng = pd.date_range(start_d, end_d, freq='D')
    df = df.reindex(rng, fill_value=0)

    #assuming baseline_x is date indexed and missed dates already filled
    X = baseline_df[feature_name] #debug_df['date'] # data_as_frame['Months']
    Y = df[feature_name] #data_as_frame['SunSpots']

    # # plot the results
    # plot_results(X, y=Y, window_size=20, text_xlabel="Months", sigma_value=3, text_ylabel="No. of Sun spots")

    events = explain_anomalies_cmnty(X, Y, window_size=ws, sigma=sig)
    # Display the anomaly dict
    # print("Information about the anomalies model:{}".format(events))
    # print()
    # print('num_anom=', len(events['anomalies_dict']))


    # events_rolling = explain_anomalies_rolling_std_cmnty(X, Y, window_size=ws, sigma=sig)
    # # Display the anomaly dict
    # print("Information about the anomalies model rolling:{}".format(events_rolling))
    # print()

    #####################################################################################################
    # if uname in ins_ans_lst:
    #     plot_res_cmnty (events, X, Y, baseline_type='PBP', feature_type=feature_name+fti, user_name=uname)
        # plot_res (events_rolling, Y, baseline_type='UBP-rolling', feature_type=feature_name)
    return events
    # return events_rolling
