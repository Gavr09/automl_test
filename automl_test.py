# Standard python libraries
import logging
import os
import time
import requests
logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.INFO)

# Installed libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import torch

# Imports from our package
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

N_THREADS = 4 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 0.5*60*60 # Time in seconds for automl run - UPDATED VALUE FOR UTILIZATION
TARGET_NAME = 'final_price' # Target column name

train_data = pd.read_csv('train_data.csv')
train_data.head()

test_data = pd.read_csv('test_data.csv')
test_data.head()

submission = pd.read_csv('sample_submission.csv')
submission.head()

import vininfo
from vininfo import Vin


def vin_to_dic(vin_no):
    """Преобразует VIN-номер в словарь параметров автомобиля"""
    vin_dic = {}
    # print(type(vin_no))

    try:
        vin_info = Vin(vin_no)

        vin_dic['checksum_is_ok'] = vin_info.verify_checksum()
        vin_dic['country'] = vin_info.country
        vin_dic['manufacturer'] = vin_info.manufacturer
        vin_dic['region'] = vin_info.region
        vin_dic['produce_year'] = vin_info.years[0]
        vin_dic['model_year'] = vin_info.years[1]
        vin_dic['wmi'] = vin_info.wmi  # всемирный индекс изготовителя
        vin_dic['vds'] = vin_info.vds[:-1]  # технические характеристики автомобиля
        vin_dic['vis'] = vin_info.vis  # идентификационный номер автомобиля

        details = vin_info.details
        if details:
            vin_dic['details'] = True
            vin_dic['body'] = str(details.body)
            vin_dic['engine'] = str(details.engine)
            vin_dic['model'] = str(details.model)
            vin_dic['plant'] = str(details.plant)
            vin_dic['serial'] = str(details.serial)
            vin_dic['transmission'] = str(details.transmission)
        else:
            vin_dic['details'] = False

            for field in ['body', 'engine', 'model', 'plant', 'serial', 'transmission']:
                vin_dic[field] = None
        # print(vin_dic)
    except:
        vin_dic['checksum_is_ok'] = False
    return vin_dic

def manufacturer_category_dic(manufacturer_name):
    category_dic = {}
    luxury = ['FERRARI', 'LAMBORGHINI']
    premium = ['BENTLEY', 'ROLLS-ROYCE', 'MASERATI', 'CHRYSLER', 'TESLA', 'PORSCHE', 'MERCEDES-BENZ', 'ASTON MARTIN']
    business = ['JAGUAR', 'DODGE', 'TOYOTA', 'LEXUS', 'VOLKSWAGEN', 'AUDI', 'BMW',
              'MITSUBISHI', 'SUBARU', 'INFINITI', 'LAND ROVER', 'JEEP', 'MINI',
              'CADILLAC', 'ALFA ROMEO', 'VOLVO', 'HUMMER', 'LINCOLN', 'BUICK', 'MG', 'LANCIA']
    middle = ['SKODA', 'HYUNDAI', 'CHEVROLET', 'NISSAN', 'FORD', 'HONDA', 'SSANGYONG',
            'MAZDA', 'OPEL', 'KIA', 'RENAULT', 'SUZUKI', 'ROVER', 'FIAT', 'UAZ',
            'CITROEN', 'PEUGEOT', 'ACURA', 'CHERY', 'ISUZU', 'GMC', 'MERCURY', 'SEAT', 'PONTIAC', 'SAAB']
    trash = ['DAEWOO', 'VAZ', 'GAZ', 'ZAZ', 'MOSKVICH', 'DAIHATSU', 'HAVAL', 'TATA', 'SCION', 'SATURN']
    chinese = ['JAC', 'GREATWALL', 'FOTON']
    if manufacturer_name in luxury:
        category_dic['category'] = 'luxury'
    elif manufacturer_name in premium:
        category_dic['category'] = 'premium'
#     elif manufacturer_name in business:
#         category_dic['category'] = 'business'
#     elif manufacturer_name in chinese:
#         category_dic['category'] = 'chinese'
#     elif manufacturer_name in middle:
#         category_dic['category'] = 'middle'
#     elif manufacturer_name in trash:
#         category_dic['category'] = 'trash'
    else:
        category_dic['category'] = 'other'
    return category_dic

def additional_info_from_vin(data):
    vin_data = data['car_vin'].apply(vin_to_dic).apply(pd.Series)
    vin_data.head()
    # print(vin_data[vin_data['checksum_is_ok']==True])
    print(vin_data.columns)
    data = data.join(vin_data)
    data.head()
    print(data.columns)
    return data


def add_manufacturer_category_info(data):
    manufacturer_category_data = data['vehicle_manufacturer'].apply(manufacturer_category_dic).apply(pd.Series)
    manufacturer_category_data.head()
    data = data.join(manufacturer_category_data)
    data.head()
    print(data.columns)
    return data

def mean_price_by_manufacturer_dic(data):
    dic = {}
    keys = data['vehicle_manufacturer'].unique()
    for key in keys:
        dic[key] = {}
        data_for_key = data[data['vehicle_manufacturer']==key]
        for deal_type in ['For Sale', 'For Rent']:
            data_for_deal_type = data_for_key[data_for_key['deal_type'] == deal_type]
            dic[key]['mean_price_' + deal_type] = data_for_deal_type['final_price'].mean()
    return dic


def create_extra_features(data):
    data['NANs_cnt'] = data.isnull().sum(axis=1)
    data['age'] = data['vehicle_year'].apply(lambda x: 2021 - x)
    data = additional_info_from_vin(data)
    # data = add_mean_price_by_manufacturer(data)
    return data


def create_col_with_min_freq(data, col, min_freq=10):
    # replace rare values (less than min_freq rows) in feature by RARE_VALUE
    data[col + '_fixed'] = data[col].astype(str)
    data.loc[data[col + '_fixed'].value_counts()[data[col + '_fixed']].values < min_freq, col + '_fixed'] = "RARE_VALUE"
    data.replace({'nan': np.nan}, inplace=True)


def create_gr_feats(data):
    # create aggregation feats for numeric features based on categorical ones
    dic = mean_price_by_manufacturer_dic(data)
    mean_price_data = data['vehicle_manufacturer'].apply(lambda x: dic[x]).apply(pd.Series)
    data = data.join(mean_price_data)

    for cat_col in ['vehicle_manufacturer', 'vehicle_model', 'vehicle_category',
                    'vehicle_gearbox_type', 'doors_cnt', 'wheels', 'vehicle_color',
                    'vehicle_interior_color', 'deal_type']:
        create_col_with_min_freq(data, cat_col, 15)
        for num_col in ['current_mileage', 'vehicle_year', 'car_leather_interior']:
            for n, f in [('mean', np.mean), ('min', np.nanmin), ('max', np.nanmax)]:
                data['FIXED_' + n + '_' + num_col + '_by_' + cat_col] = data.groupby(cat_col + '_fixed')[
                    num_col].transform(f)

    # create features with counts
    for col in ['vehicle_manufacturer', 'vehicle_model', 'vehicle_category',
                'current_mileage', 'vehicle_year', 'vehicle_gearbox_type', 'doors_cnt',
                'wheels', 'vehicle_color', 'vehicle_interior_color', 'car_vin', 'deal_type']:
        data[col + '_cnt'] = data[col].map(data[col].value_counts(dropna=False))

    return data

train_data=create_extra_features(train_data)
test_data=create_extra_features(test_data)

all_df = pd.concat([train_data, test_data]).reset_index(drop=True)
all_df = create_gr_feats(all_df)
train_data, test_data = all_df[:len(train_data)], all_df[len(train_data):]
print(train_data.shape, test_data.shape)

train_data_for_sale = train_data[train_data.deal_type == 'For Sale']
train_data_for_rent = train_data[train_data.deal_type != 'For Sale']

test_data_for_sale  = test_data[test_data.deal_type == 'For Sale']
test_data_for_rent  = test_data[test_data.deal_type != 'For Sale']

print(train_data_for_sale.shape, train_data_for_rent.shape)
print(test_data_for_sale.shape, test_data_for_rent.shape)

# COMPETITION METRIC IS MAE SO WE SET IT FOR OUR TASK
task = Task('reg', loss='mae', metric='mae')

roles = {'target': TARGET_NAME,
         'drop': ['row_ID']
         }

def studying_and_prediction(train_data, test_data, task, roles, TIMEOUT, N_THREADS, N_FOLDS, RANDOM_STATE):
    # CHANGED TabularAutoML to TabularUtilizedAutoML for timeout utilization
    automl = TabularUtilizedAutoML(task=task,
                                    timeout=TIMEOUT,
                                    cpu_limit=N_THREADS,
                                    general_params={'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]},
                                    reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS,
                                                   'random_state': RANDOM_STATE},
                                    )
    oof_pred = automl.fit_predict(train_data, roles=roles)
    logging.info('oof_pred:\n{}\nShape = {}'.format(oof_pred, oof_pred.shape))

    # Fast feature importances calculation
    try:
        fast_fi = automl.get_feature_scores('fast')
        fast_fi.set_index('Feature')['Importance'].plot.bar(figsize=(20, 10), grid=True)
    except:
        pass

    test_pred = automl.predict(test_data)
    logging.info('Prediction for test data:\n{}\nShape = {}'.format(test_pred, test_pred.shape))

    logging.info('Check scores...')
    try:
        logging.info('OOF score: {}'.format(
            mean_absolute_error(train_data[TARGET_NAME].values, oof_pred.data[:, 0])))
    except:
        pass
    return test_pred

# # CHANGED TabularAutoML to TabularUtilizedAutoML for timeout utilization
# automl_for_sale = TabularUtilizedAutoML(task = task,
#                        timeout = TIMEOUT,
#                        cpu_limit = N_THREADS,
#                        general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]},
#                        reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
#                       )
# oof_pred_for_sale = automl_for_sale.fit_predict(train_data_for_sale, roles = roles)
# logging.info('oof_pred:\n{}\nShape = {}'.format(oof_pred_for_sale, oof_pred_for_sale.shape))
#
# # CHANGED TabularAutoML to TabularUtilizedAutoML for timeout utilization
# automl_for_rent = TabularUtilizedAutoML(task = task,
#                        timeout = TIMEOUT,
#                        cpu_limit = N_THREADS,
#                        general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]},
#                        reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
#                       )
# oof_pred_for_rent = automl_for_rent.fit_predict(train_data_for_rent, roles = roles)
# logging.info('oof_pred:\n{}\nShape = {}'.format(oof_pred_for_rent, oof_pred_for_rent.shape))
#
# # Fast feature importances calculation
# try:
#     fast_fi = automl_for_sale.get_feature_scores('fast')
#     fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (20, 10), grid = True)
# except:
#     pass
#
# try:
#     fast_fi = automl_for_rent.get_feature_scores('fast')
#     fast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (20, 10), grid = True)
# except:
#     pass
#
# test_pred_for_sale = automl_for_sale.predict(test_data_for_sale)
# logging.info('Prediction for test data:\n{}\nShape = {}'.format(test_pred_for_sale, test_pred_for_sale.shape))
#
# logging.info('Check scores...')
# try:
#     logging.info('OOF score: {}'.format(mean_absolute_error(train_data_for_sale[TARGET_NAME].values, oof_pred_for_sale.data[:, 0])))
# except:
#     pass

# test_pred_for_rent = automl_for_rent.predict(test_data_for_rent)
# logging.info('Prediction for test data:\n{}\nShape = {}'
#               .format(test_pred_for_rent, test_pred_for_rent.shape))
#
# logging.info('Check scores...')
# try:
#     logging.info('OOF score: {}'.format(mean_absolute_error(train_data_for_rent[TARGET_NAME].values, oof_pred_for_rent.data[:, 0])))
# except:
#     pass
# submission[TARGET_NAME] = test_pred.data[:, 0]

FLAG_TWO_MODELS = False

if FLAG_TWO_MODELS:
    test_pred_for_sale = studying_and_prediction(train_data_for_sale, test_data_for_sale, task, roles, TIMEOUT, N_THREADS, N_FOLDS, RANDOM_STATE)
    test_pred_for_rent = studying_and_prediction(train_data_for_rent, test_data_for_rent, task, roles, TIMEOUT, N_THREADS, N_FOLDS, RANDOM_STATE)

    submission_for_sale = pd.DataFrame(index=test_data_for_sale.index.values)
    submission_for_sale[TARGET_NAME] = test_pred_for_sale.data[:, 0]

    submission_for_rent = pd.DataFrame(index=test_data_for_rent.index.values)
    submission_for_rent[TARGET_NAME] = test_pred_for_rent.data[:, 0]
    submission = pd.concat([submission_for_sale, submission_for_rent])
    submission=submission.sort_index()

else:
    test_pred = studying_and_prediction(train_data, test_data, task, roles, TIMEOUT, N_THREADS, N_FOLDS, RANDOM_STATE)
    submission[TARGET_NAME] = test_pred.data[:, 0]
    submission.head()

submission.to_csv('lightautoml_baseline_custom_fe.csv', index = True)