import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from src.preprocess.data_validation import *
from src.preprocess.data_preprocessing import data_preprocess_pipelipne

def dummy_col_names(model_param):
    dummy_col = []        
    for i in model_param['unit_price_labels']:
        col_name = 'Unit_Price_cat' + '_' + i
        dummy_col.append(col_name)
    for j in model_param['quantity_labels']:
        col_name = 'Total_Qty_cat' + '_' + j
        dummy_col.append(col_name)
    return dummy_col

def xgb_model_colum_creation(model_param, num_class):
    table_col = []
    if num_class > 2:
        for i in range(len(model_param['column_list'])):
            for j in range(num_class):
                col_name = model_param['column_list'][i].split('_')[0] + '_cat_' + str(j)
                table_col.append(col_name)
        for i in range(num_class):
            col_name = 'Prob_of_FA_' + str(i) 
            table_col.append(col_name)
    else:
        for i in range(len(model_param['column_list'])):
            col_name = model_param['column_list'][i].split('_')[0] + '_score'
            table_col.append(col_name)
        table_col.append('Prob_of_FA')      
    dummy_col = dummy_col_names(model_param)
    table_col = table_col + dummy_col
    return table_col

def wrapper_multiclass(df, FA_cutoff):
    df['flag'] = 1
    df['FA_ind'] = np.where(df['predict_class'] == 1, 1, 0)
    req_group = df.groupby(['ReqNum']).agg({'FA_ind': sum, 'flag': sum, 'Unit_Price': sum}).reset_index()
    req_group['FA_req_prob'] = (req_group['FA_ind'])/ req_group['flag']
    req_group = req_group.rename(columns={"Unit_Price": "Unit_Price_sum"})
    df = df.merge(req_group.loc[:,['ReqNum', 'FA_req_prob', 'Unit_Price_sum']], on = 'ReqNum')
    
    df['predict_class_revised'] = np.where(((df['Unit_Price_sum'] >= FA_cutoff) & (df['FA_req_prob'] >= 0.5)), 1,
                                       np.where(df['event_ind'] == 1, 0, 
                                            np.where(((df['predict_class'] == 0) & (df['line_total_ind'] == 1)), 0, 
                                                np.where(((df['predict_class'] == 0) & (df['milestone_ind'] == 0)), 0, 2)))) 
    
    df['ML_Prediction'] = np.where(df['predict_class'] == 0, 'Prepaid', 
                                           np.where(df['predict_class'] == 1, 'Fixed Asset', 'Expense'))
    
    df['ML_Prediction_revised'] = np.where(df['predict_class_revised'] == 0, 'Prepaid', 
                                           np.where(df['predict_class_revised'] == 1, 'Fixed Asset', 'Expense'))
    
    return df

def wrapper_two_class(df, FA_cutoff):
    df['flag'] = 1
    req_group = df.groupby(['ReqNum']).agg({'predict_class': sum, 'flag': sum}).reset_index()
    req_group['FA_req_prob'] = (req_group['flag'] - req_group['predict_class'])/ req_group['flag']
    df = df.merge(req_group.loc[:,['ReqNum', 'FA_req_prob']], on = 'ReqNum')
    df['predict_class_revised'] = np.where(((df['FA_req_prob'] >= 0.5) & (df['Unit_Price'] >= FA_cutoff)) , 0, 1)
    df['ML_Prediction'] = np.where(df['predict_class'] == 1, 'Prepaid', 'Fixed_Asset')
    df['ML_Prediction_revised'] = np.where(df['predict_class_revised'] == 1, 'Prepaid', 'Fixed_Asset')
    return df


def apply_xgb_classifier(df, model_param, num_class):
    df_small = df.filter(items = ['Unit_Price_cat','Total_Qty_cat'])
    df_small = pd.get_dummies(df_small, prefix=['Unit_Price_cat','Total_Qty_cat'])
    df = pd.concat([df, df_small], axis = 1)
    table_col = xgb_model_colum_creation(model_param, num_class)
    exclusion_col = table_col + model_param['column_list']
    df_model = df.filter(items = table_col)
    
    if num_class > 2:
        model_path = os.path.join(model_param['xgb_weight_folder'], model_param['xgb_weights'][1])
        loaded_model = pickle.load(open(model_path, "rb"))
        y_pred = loaded_model.predict(df_model.values)
        y_pred_prob = loaded_model.predict_proba(df_model.values)
        filter_col = [i for i in df.columns if i not in exclusion_col]
        df = df.filter(items = filter_col)
        df['predict_class'] = y_pred
        df = pd.concat([df, pd.DataFrame(y_pred_prob, columns = ['prob_class_PP', 'prob_class_FA', 'prob_class_Exp'])], axis = 1)
        df = wrapper_multiclass(df, model_param['FA_cutoff_unit_price'])
    else:
        model_path = os.path.join(model_param['xgb_weight_folder'], model_param['xgb_weights'][0])
        loaded_model = pickle.load(open(model_path, "rb"))
        y_pred = loaded_model.predict(df_model.values)
        y_pred_prob = loaded_model.predict_proba(df_model.values)
        filter_col = [i for i in df.columns if i not in exclusion_col]
        df = df.filter(items = filter_col)
        df['predict_class'] = y_pred
        df['prob_class_FA'] = y_pred_prob.T[0]
        df['prob_class_PP'] = y_pred_prob.T[1]
        df = wrapper_two_class(df, model_param['FA_cutoff_unit_price'])
        
    return df

def make_evaluation(config):
    file_input = config['file_input']
    model_param = config['model_input']
    if file_input['use_DB'] == 0:
        df = pd.read_csv(os.path.join(os.getcwd(), file_input['input_file_path'], file_input['input_file_name']))
        
    if model_param['num_class'] > 2:
        supplier_prob = pd.read_csv(os.path.join(os.getcwd(), file_input['helper_file_path'], file_input['supplier_filename'][1]))
    else:
        supplier_prob = pd.read_csv(os.path.join(os.getcwd(), file_input['helper_file_path'], file_input['supplier_filename'][0]))
    
    if column_validation(df, model_param['essential_col']) == 1:
        print('------------Required input not provided. Please check the input---------')
    else:
        df = data_filteration(df)
        df = data_preprocess_pipelipne(df, supplier_prob, model_param)
        df = apply_xgb_classifier(df, model_param, model_param['num_class'])
        return df