# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 11:14
# @Author  : liqibao
# @File    : 20190226.py

import os
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from xgboost import plot_importance
from matplotlib.pylab import rcParams
warnings.filterwarnings('ignore')

def load_data(path):
    '''
    :param path: file path
    :return: dot_date
    '''
    # load data
    dot_num1 = pd.read_csv(path + 'cst1.csv', header = None, names = ['order_date', 'dot', 'company', 'order_num'])
    dot_num2 = pd.read_csv(path + 'cst2.csv', header = None, names = ['order_date', 'dot', 'company', 'order_num'])
    dot_num = pd.concat([dot_num1, dot_num2], ignore_index=True)
    dot_num['order_date'] = [str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:] for x in dot_num['order_date']]
    dot_num['order_date'] = pd.to_datetime(dot_num['order_date'])

    dot_info = pd.read_csv(path + 'dot_1.csv', header=None,names=['dot', 'check_date', 'dot_type', 'city_code', 'area', 'province', 'city', 'district','town', 'village'])

    # 过滤点部(过滤未运营点部）
    dot_data_tmp = dot_num[dot_num['dot'].isin(dot_info['dot'])]

    dot_data_tmp = dot_data_tmp.drop(['company'], axis=1)
    dot_groupby = dot_data_tmp.groupby(['dot', 'order_date']).sum()
    dot_groupby.reset_index(inplace=True)
    dot_groupby.sort_values(by='order_date', inplace=True)

    # 数据预处理
    dot_preprocessing = pd.DataFrame()
    for i, dot in enumerate(dot_groupby['dot'].unique()):
        one_dot = dot_groupby.loc[dot_groupby['dot'] == dot]
        check_time = dot_info.loc[dot_info['dot'] == dot, 'check_date']
        one_dot = one_dot[one_dot['order_date'] > check_time.iloc[0]]
        if len(one_dot) == 0:
            continue
        num_median = np.median(one_dot['order_num'])

        # 均值填充异常点
        lp = np.percentile(one_dot['order_num'], 25)
        up = np.percentile(one_dot['order_num'], 75)
        check_num = up + 1.5 * (up - lp)
        num_mean = np.mean(one_dot['order_num'])
        one_dot.loc[one_dot['order_num'] >= check_num, ['order_num']] = num_mean

        # 填充日期，中位数填充缺失值
        one_dot.index = one_dot['order_date']
        one_dot_resample = one_dot.resample('D').asfreq()
        one_dot_resample.drop(['order_date'], axis=1, inplace=True)
        one_dot_resample.reset_index(inplace=True)
        one_dot_resample['dot'].fillna(dot, inplace=True)
        one_dot_resample['order_num'].fillna(num_median, inplace=True)
        one_dot_resample['dot_id'] = i + 1

        dot_preprocessing = pd.concat([dot_preprocessing, one_dot_resample], axis=0, ignore_index=True)

    # 对数平滑
    dot_prep = dot_preprocessing.copy()
    dot_prep['order_num'] = np.log1p(dot_prep['order_num'])
    dot_prep.sort_values(by=['order_date'], inplace=True)

    # 合并点部信息
    dot_data = pd.merge(dot_info, dot_prep, how='left', on='dot')
    dot_data.loc[dot_data['city'].isnull(), ['city']] = dot_data.loc[dot_data['city'].isnull(), ['province']].values
    dot_data.loc[dot_data['district'].isnull(), ['district']] = dot_data.loc[dot_data['district'].isnull(), ['city']].values

    dot_data.dropna(axis=0, inplace=True)
    dot_data.sort_values(by='order_date', ascending=True, inplace=True)

    return dot_data

# 日期特征
def time_feature(data):
    '''
    :param data: 输入数据
    :return: dot_prep_1：时间特征
    '''
    dot_prep_1 = data.copy()

    dot_prep_1['year'] = dot_prep_1.order_date.dt.year
    dot_prep_1['month'] = dot_prep_1.order_date.dt.month
    dot_prep_1['day'] = dot_prep_1.order_date.dt.day
    dot_prep_1['dayofweek'] = dot_prep_1.order_date.dt.dayofweek
    dot_prep_1['dayofyear'] = dot_prep_1.order_date.dt.dayofyear
    dot_prep_1['weekofyear'] = dot_prep_1.order_date.dt.weekofyear

    # TODO，待修改
    weekofsun = pd.date_range('2017-01-01', '2019-02-08', freq='W-SUN')
    dot_prep_1['weekofsun'] = 0
    dot_prep_1.loc[dot_prep_1['order_date'].isin(weekofsun), ['weekofsun']] = 1

    #TODO 节假日
    dot_prep_1['holidays'] = 0
    dot_prep_1.loc[dot_prep_1['order_date'].isin(pd.to_datetime(
            ['2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2017-01-23', '2017-01-24', '2017-01-25',
             '2017-01-26', '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02',
             '2017-02-03', '2017-02-04', '2018-02-12', '2018-02-13', '2018-02-14', '2018-02-15', '2018-02-16', '2018-02-17',
             '2018-02-18', '2018-02-19', '2018-02-20', '2018-02-21', '2018-02-22', '2018-02-23', '2018-02-24', '2019-02-01',
             '2019-02-02', '2019-02-03', '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07', '2019-02-08', '2019-02-09',
             '2019-02-10', '2019-02-11', '2019-02-12', '2019-02-13', '2017-04-02', '2017-04-03', '2017-04-04', '2018-04-05',
             '2018-04-06', '2018-04-07', '2019-04-05', '2019-04-06', '2019-04-07', '2017-05-01', '2017-05-02', '2017-05-03',
             '2018-05-01', '2018-05-02', '2018-05-03', '2019-05-01', '2019-05-02', '2019-05-03', '2017-05-30', '2018-06-18',
             '2019-06-07', '2017-10-04', '2018-09-22', '2018-09-23', '2018-09-24', '2019-09-13', '2019-09-14', '2019-09-15',
             '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07', '2018-10-01',
             '2018-10-02', '2018-10-03', '2018-10-04', '2018-10-05', '2018-10-06', '2018-10-07', '2019-10-01', '2019-10-02',
             '2019-10-03', '2019-10-04', '2019-10-05', '2019-10-06', '2019-10-07'])), ['holidays']] = 1

    return dot_prep_1

# 滞后特征（训练数据）
def create_feature(data, all_dot):
    '''
    :param data: 输入数据
    :param all_dot: 点部名称
    :return: dot_prep_result：训练数据
    '''
    dot_prep_1 = time_feature(data)
    dot_prep_ts = data.copy()
    dot_prep_ts.index = pd.to_datetime(dot_prep_ts['order_date'])

    dot_prep_2 = pd.DataFrame()
    dot_prep_3 = pd.DataFrame()
    for dot in all_dot:
        dot_ts = dot_prep_ts.loc[dot_prep_ts['dot'] == dot, ['order_num']]
        dot_ts.sort_index(ascending=True, inplace=True)
        # lagging
        lagging = pd.concat([dot_ts.shift(1), dot_ts.shift(2), dot_ts.shift(3), dot_ts.shift(4), dot_ts.shift(5), dot_ts.shift(6), dot_ts.shift(7)], axis=1)
        lagging.columns = ['lagging1', 'lagging2', 'lagging3', 'lagging4', 'lagging5', 'lagging6', 'lagging7']
        lagging['dot'] = dot
        lagging['order_date'] = dot_ts.index
        lagging.dropna(axis=0, inplace=True)
        dot_prep_2 = pd.concat([dot_prep_2, lagging], axis=0)

        # rolling
        rolling = pd.concat([dot_ts.rolling(window=3).mean(), dot_ts.rolling(window=5).mean(), dot_ts.rolling(window=7).mean()], axis=1)
        rolling.index = rolling.index + pd.DateOffset(days=1)
        rolling.columns = ['rolling3', 'rolling5', 'rolling7']
        rolling['dot'] = dot
        rolling['order_date'] = rolling.index
        rolling.dropna(axis=0, inplace=True)
        dot_prep_3 = pd.concat([dot_prep_3, rolling], axis=0)
    dot_prep_2.reset_index(drop=True, inplace=True)
    dot_prep_3.reset_index(drop=True, inplace=True)

    # merge
    dot_prep_result_tmp = pd.merge(dot_prep_1, dot_prep_2, how='right', on=['dot', 'order_date'])
    dot_prep_result = pd.merge(dot_prep_result_tmp, dot_prep_3, how='left', on=['dot', 'order_date'])
    dot_prep_result.sort_values(by='order_date', ascending=True, inplace=True)
    #     dot_prep_result.drop(['dot', 'order_date'], axis = 1, inplace = True)

    return dot_prep_result

# 滞后特征（预测数据）
def pre_feature(data, dot_name):
    x_test_tmp = pd.DataFrame()
    rolling_tmp = pd.DataFrame()
    for dot in dot_name:
        # lagging
        dot_test = data.loc[data['dot'] == dot]
        dot_test_ts = dot_test.copy()
        dot_test['lagging7'] = dot_test['lagging6']
        dot_test['lagging6'] = dot_test['lagging5']
        dot_test['lagging5'] = dot_test['lagging4']
        dot_test['lagging4'] = dot_test['lagging3']
        dot_test['lagging3'] = dot_test['lagging2']
        dot_test['lagging2'] = dot_test['lagging1']
        dot_test['lagging1'] = dot_test['order_num']

        x_test_tmp = pd.concat([x_test_tmp, dot_test])

        # rolling
        dot_test_ts.index = pd.to_datetime(dot_test_ts['order_date'])
        num_ts = dot_test_ts.loc[dot_test_ts['dot'] == dot, ['order_num']]
        rolling = pd.concat([num_ts.rolling(window=3).mean(), num_ts.rolling(window=5).mean(), num_ts.rolling(window=7).mean()], axis=1)
        rolling.columns = ['rolling3', 'rolling5', 'rolling7']
        rolling['dot'] = dot
        rolling['order_date'] = rolling.index
        rolling_tmp = pd.concat([rolling_tmp, rolling])

    x_test = pd.merge(x_test_tmp.drop(['rolling3', 'rolling5', 'rolling7'], axis=1), rolling_tmp, how='left', on=['dot', 'order_date'])
    x_test['order_date'] = x_test['order_date'] + pd.DateOffset(days=1)
    x_test.dropna(axis=0, inplace=True)

    return x_test

# training model
def xgboost_model(x_train, y_train):
    params = {
        "learning_rate": 0.01,
        "n_estimators": 3300,
        "max_depth": 5,
        "min_child_weight": 3,
        "gamma": 0.5,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "silent": 1,
        "seed": 21
    }
    print("train_start : ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    gbm = xgb.XGBRegressor(**params)
    gbm.fit(x_train.drop(['dot', 'order_date'], axis=1), y_train)
    print("train_end : ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for column, importance in zip(x_train.drop(['dot', 'order_date'], axis=1).columns, gbm.feature_importances_):
        print(column, '\t', importance)
    # plot_importance(gbm)
    # plt.savefig(path + "feature_importance.png")

    return gbm

def RMSE(y_test, y_pred):
    return np.sqrt(np.sum((y_test - y_pred) ** 2)/len(y_test))

def MAE(y_test, y_pred):
    return np.sum(np.abs(y_test - y_pred))/len(y_test)

# mdoel forecast
# def xgb_predict(gbm, x_test, y_test, path):
#     # 验证
#     print("predict_start : ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#     y_pre = gbm.predict(x_test.drop(['dot', 'order_date'], axis=1))
#     print("predict_end : ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#     result = pd.DataFrame({'dot': x_test['dot'], 'order_date': x_test['order_date'], 'y_test': np.expm1(y_test), 'y_pre': np.expm1(y_pre)})
#     result['RMSE'] = RMSE(np.expm1(y_test), np.expm1(y_pre))
#     result['MAE'] = MAE(np.expm1(y_test), np.expm1(y_pre))
#     print("MAE : ", MAE(np.expm1(y_test), np.expm1(y_pre)), " RMSE : ", RMSE(np.expm1(y_test), np.expm1(y_pre)))
#
#     result.to_csv(path + "dot_result.csv", encoding='GBK', index=False)
#     return result

def forecast(gbm, dot_res):
    print("PREDICT BEGAN : ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_to_csv = pd.DataFrame()
    pre_date = dot_res['order_date'].max() + pd.DateOffset(months=-1)
    org_test = dot_res[dot_res['order_date'] > pre_date]
    test_dot = org_test['dot'].unique()

    feature_date = dot_res.columns.drop(['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'weekofsun', 'holidays'])
    feature_num = dot_res.columns.drop(['order_num'])
    feature = dot_res.columns

    # 预测天数
    for i in range(15):
        print(i)
        x_test_all = pre_feature(org_test, test_dot)
        last_train_date = dot_res['order_date'].max() + pd.DateOffset(days=i + 1)
        x_test_day = x_test_all.loc[x_test_all['order_date'] == last_train_date, feature_date]

        x_test = time_feature(x_test_day)
        x_test = x_test[feature_num]  # 排除order_num字段

        y_pre = gbm.predict(x_test.drop(['dot', 'order_date'], axis=1))
        result = pd.DataFrame({'dot': x_test['dot'], 'order_date': x_test['order_date'], 'order_num': y_pre})
        result_to_csv = pd.concat([result_to_csv, result])
        x_test_merge = pd.merge(x_test, result, on=['dot', 'order_date'], how='left')
        x_test_merge = x_test_merge[feature]

        org_test = pd.concat([org_test, x_test_merge])
    print("PREDICT END : ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return result_to_csv

def main():
    path = "D:/program/program1/Cargo_forecast/dot_data_all/"

    # 数据预处理
    dot_data = load_data(path)

    # 拆分数据集
    train_date = dot_data['order_date'].max() + pd.DateOffset(months=-1)
    dot_train = dot_data[dot_data['order_date'] <= train_date]
    dot_test = dot_data[dot_data['order_date'] > train_date]

    # 时间、滞后、移动平均特征
    all_dot = dot_train['dot'].unique()
    dot_prep_feature = create_feature(dot_train, all_dot)
    dot_res_tmp = dot_prep_feature.drop(['check_date', 'city', 'district', 'town', 'village'], axis=1)

    # 统计特征
    dot_prep_sta = dot_res_tmp.copy()
    dot_prep_dayofweek = dot_prep_sta.groupby(['dot', 'dayofweek'])['order_num'].agg([('week_mean', np.mean), ('week_median', np.median), ('week_min', np.min), ('week_max', np.max), ('week_std', np.std)]).reset_index()
    dot_prep_day = dot_prep_sta.groupby(['dot', 'day'])['order_num'].agg([('day_mean', np.mean), ('day_median', np.median), ('day_min', np.min), ('day_max', np.max), ('day_std', np.std)]).reset_index()

    dot_prep_tmp = pd.merge(dot_prep_sta, dot_prep_dayofweek, on=['dot', 'dayofweek'], how='left')
    dot_prep_stas = pd.merge(dot_prep_tmp, dot_prep_day, on=['dot', 'day'], how='left')

    # 哑变量
    dot_res = pd.concat([dot_prep_stas, pd.get_dummies(dot_prep_stas['dot_type']), pd.get_dummies(dot_prep_stas['area']), pd.get_dummies(dot_prep_stas['province'])], axis=1)
    dot_res.drop(['dot_type', 'area', 'province'], axis=1, inplace=True)
    dot_res = dot_res[~dot_res['dot'].isin(dot_res[dot_res['week_std'].isnull()]['dot'].unique())]

    # 存储数据
    # dot_res.to_csv(path + "dot_res_20190226.csv", encoding='GBK')

    # 模型训练输入数据x,y
    dot_xtrain = dot_res.drop(['order_num'], axis=1)
    dot_ytrain = dot_res['order_num']

    # trainning model
    gbm = xgboost_model(dot_xtrain, dot_ytrain)

    # dump model
    fw = open(path + "gbm.txt", 'wb')
    pickle.dump(gbm, fw)

    # model forecast
    result_to_csv = forecast(gbm, dot_res)

    # 评价函数，回写csv
    result_to_csv['order_num'] = np.expm1(result_to_csv['order_num'])
    result_to_csv.columns = ['dot', 'order_date', 'y_pre']
    dot_test['order_num'] = np.expm1(dot_test['order_num'])
    dot_test = dot_test[['dot', 'order_date', 'order_num']]
    pre_result = pd.merge(result_to_csv, dot_test, how = 'left', on = ['dot', 'order_date'])
    pre_result = pre_result[['dot', 'order_date', 'order_num', 'y_pre']]

    print("MAE : ", MAE(pre_result.order_num, pre_result.y_pre))
    print("RMSE : ", RMSE(pre_result.order_num, pre_result.y_pre))

    pre_result.to_csv(path + "pre_result.csv", encoding = 'GBK')

if __name__ == '__main__':
    main()




