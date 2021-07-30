import os
import pandas as pd
import numpy as np
from src.preprocess.utils import *
from src.models.ml_evaluation import test_on_pretrained_model

def softMax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 0)
    
def datacol_null_handling(df, sup, sup_default):
    df['Supplier_sort'] = df.apply(lambda x: str(x['SupplierName']).split('-')[0], axis = 1)
    df['Justification_mod'] =  df.apply(lambda x: 'na' if pd.isna(x['Justification']) else x['Justification'], axis = 1)
    df['Item_mod'] =  df.apply(lambda x: 'na' if pd.isna(x['Item']) else x['Item'], axis = 1)
    df['Commodity_mod'] =  df.apply(lambda x: 'na' if pd.isna(x['Commodity']) else x['Commodity'], axis = 1)
    df = pd.merge(left= df, right= sup, on = 'Supplier_sort', how = 'left')
    col_list = []
    for i in df.columns:
        if i.find('Prob_of_FA') >= 0:
            col_list.append(i)
    if len(col_list) == 1:
        df['Prob_of_FA'] = df.apply(lambda x: sup_default if pd.isna(x['Prob_of_FA']) else x['Prob_of_FA'], axis = 1)
    else:
        for c in col_list:
            df[c] = df.apply(lambda x: 0.33 if pd.isna(x[c]) else x[c], axis = 1)
    return (df)

def price_column_modifications(df, unit_bins, unit_labels, qty_bins, qty_labels, curr_list, conv_list, conv_default):
    df['LineTotal_Mod'] = df.apply(lambda x: float(x['LineTotal'].replace(',', '').split(' ')[0]), axis = 1)
    df['Unit_Price'] = df.apply(lambda x: float(str(x['Price']).replace(',', '')) if pd.notna(x['Price']) else 0 , axis = 1)
    df['Total_Qty'] = df.apply(lambda x: 1.0 if x['Unit_Price'] == 0 else float(x['LineTotal_Mod']/x['Unit_Price']), axis = 1)
    for i in list(set(df['Currency'])):
        if i not in curr_list:
            print('Currency converter for {0} is not provided. Currently {1} taken'.format(i, conv_default))
            curr_list.append(i)
            conv_list.append(conv_default)
    curr_conv = pd.DataFrame({'Currency': curr_list, 'Converter': conv_list})
    df = df.merge(curr_conv, on = 'Currency', how = 'left')
    df['LineTotal_Mod'] = df['LineTotal_Mod'] * df['Converter']
    df['Unit_Price'] = df['Unit_Price'] * df['Converter']
	#print(unit_bins)
    df['Unit_Price_cat'] = pd.cut(df['Unit_Price'], bins=unit_bins, labels=unit_labels)
    df['Total_Qty_cat'] = pd.cut(df['Total_Qty'], bins=qty_bins, labels=qty_labels)
    return (df)

def create_indicator_columns(df, key_word_list, line_total_ind_amount):
	df['line_total_ind'] = df.apply(lambda x: 1 if x['LineTotal_Mod'] >= line_total_ind_amount else 0, axis = 1)
	df['milestone_ind'] = df.apply(lambda x: 1 if pd.isna(x['SowText']) else Check_milestone(x['SowText']), axis = 1)
	df['event_ind'] = df.apply(lambda x: 0 if pd.isna(x['SowText']) else Check_event(x['SowText'], key_word_list), axis = 1)
	return (df)

def create_bert_weights(df, num_class, column_list, weight_dir, bert_weight_path, max_len, device, base_dir, use_saved, batch_size = 32):
    for i in range(len(column_list)):
        model_location =  os.path.join(bert_weight_path, weight_dir[i])
        base_location = os.path.join(os.getcwd(), bert_weight_path, base_dir)
        score = test_on_pretrained_model(df, column_list[i], model_location, base_location, use_saved, \
                                         num_class, device, batch_size, max_len[i])    
        if num_class > 2:
            score = softMax(score.T)
            table_col = []
            for j in range(num_class):
                col_name = column_list[i].split('_')[0] + '_cat_' + str(j)
                table_col.append(col_name)
            print(table_col)
            tmp = pd.DataFrame(score.T, columns = table_col)
            df = pd.concat([df,tmp], axis =1)
        else:
            col_name = column_list[i].split('_')[0] + '_score'
            df[col_name] = score
    return (df)

def data_preprocess_pipelipne(df, sup, model_param):
    df = datacol_null_handling(df, sup, model_param['sup_default_prob_FA'])
    df = price_column_modifications(df, model_param['unit_price_bins'], \
                                    model_param['unit_price_labels'], model_param['quantity_bins'], \
                                    model_param['quantity_labels'], model_param['currency_list'], \
                                    model_param['converter_list'], model_param['conv_default'])
    df = create_indicator_columns(df, model_param['key_word_list'], model_param['line_total_ind_amount'])
    if model_param['num_class'] > 2:
        weight_dir = model_param['weight_dir_3']
    else:
        weight_dir = model_param['weight_dir_2']
    df = create_bert_weights(df, model_param['num_class'], model_param['column_list'], \
                             weight_dir, model_param['bert_weight_path'], \
                             model_param['max_len'], model_param['device'], model_param['base_dir'],\
                             model_param['use_saved'], model_param['batch_size'])
    return (df)

