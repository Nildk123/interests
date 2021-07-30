import pandas as pd

def column_validation(df, essential_columns):
    data_col = df.columns
    check = [i for i in essential_columns if i not in data_col]
    if len(check) > 0:
        return 1
    else:
        return 0

def data_filteration(df):
    ini_records = df.shape[0]
    missing_account = df.loc[pd.isna(df['Account'])].shape[0]
    df = df.loc[pd.notna(df['Account'])]
    missing_amount = df.loc[pd.isna(df['LineTotal'])].shape[0]
    df = df.loc[pd.notna(df['LineTotal'])]
    print('The # of records removed due to mussing account: {0} '.format(missing_account))
    print('The # of records removed due to mussing line total : {0} '.format(missing_amount))
    print('The # of records considered for final analysis: {0} '.format(df.shape[0]))
    return df