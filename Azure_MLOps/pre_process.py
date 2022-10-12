# Azure
from azureml.core import Workspace, Experiment, Run, Dataset, Model

#python
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import argparse
# lifetimes package
from lifetimes.utils import summary_data_from_transaction_data

# defining fuctions
def prepare_data(df_RAW):
    '''
    1. Segregating Required Data using observating end date
    2. Renaming columns
    3. Aggregating SALES using BILL_DATE
    4. preparing data for prediction
    '''

    # taking required data 
    curDate     = datetime.now()
    curYear     = str(curDate.year)
    curMonth    = str(curDate.month)
    cutoff_Date = pd.to_datetime(f'{curYear}-{curMonth}-01')
    print(f'CutOff Date: {cutoff_Date}')
    
    # 1. Segregating Data (in case if we are not using LAST BILL DATE as OBS_END_DATE)
    df_main = df_RAW[df_RAW['BILL_DATE_YYYY_MM_DD']<cutoff_Date]
    obs_end_date = df_main['BILL_DATE_YYYY_MM_DD'].max()
    print(f"Observation End Date: {obs_end_date}")
    
        
    # 2.Renaming column names
    df_main.rename(columns={'BILL_DATE_YYYY_MM_DD':'BILL_DATE',
                            'SAP_NET_SALES_AMOUNT_USD':'NET_SALES_AMOUNT',
                            'SAP_NET_SALES_C_UNITS':'NET_SALES_UNITS'}, inplace=True)
    
    # 3. Agrregating Sales at Invoice Level
    df_main = df_main.sort_values(by=['BILL_NUMBER','BILL_DATE'])
    df_main = df_main.groupby(['COUNTRY','CUSTOMER_ID','BILL_NUMBER']).agg({'BILL_DATE':lambda x: x.unique()[-1],
                                                                            'NET_SALES_AMOUNT':['sum'],
                                                                            'NET_SALES_UNITS':['sum'],
                                                                          }).reset_index()
    # Note: If one bill number has more than one Bill date,  
    # picking up the latest bill date as suggest by Business(MARIANA)
    df_main.columns = ['COUNTRY','CUSTOMER_ID','BILL_NUMBER','BILL_DATE','NET_SALES_AMOUNT','NET_SALES_UNITS']
    
    # as per the instructions given by business
    # droppping '-Ve' and '0' Sale amount after aggregating Sales at Invoice(Bill Number) Level
    print('dropping Negative and Zero Sales after aggregating Sales Amount at Invoice(BILL_NUMBER) level')
    df_main_filtered = df_main[df_main['NET_SALES_AMOUNT']>0]
    
    # grouping by at Bill Date level
    df_main_filtered2 = df_main_filtered.groupby(['COUNTRY','CUSTOMER_ID','BILL_DATE']).agg({'NET_SALES_AMOUNT':['sum'],
                                                                                          'NET_SALES_UNITS':['sum'],
                                                                                          }).reset_index()
    
    df_main_filtered2.columns = ['COUNTRY','CUSTOMER_ID','BILL_DATE','NET_SALES_AMOUNT','NET_SALES_UNITS'] 

    print(f'RAW Dataframe Shape Before dropping Negative and Zero Sales:{df_RAW.shape}')
    print(f'Dataframe Shape After dropping Negative and Zero Sales:{df_main_filtered.shape}')  
    print(f'Total Invoices(BillNumbers) dropped with Negative and Zero Sale Amount: {df_main.shape[0]-df_main_filtered.shape[0]}')
    
    print(f"Dataframe Shape aggregating Sales at BILL_DATE level:{df_main_filtered2.shape}")
    print(f"Min Bill Date: {df_main_filtered2['BILL_DATE'].min()}")
    print(f"Max Bill Date: {df_main_filtered2['BILL_DATE'].max()}")  
    
    # calling function
    df_train = prepare_train_test_data(df_main_filtered2)

    # Aggregating RAW DATA
    df_RAW  = df_RAW.groupby(['COUNTRY','CUSTOMER_ID']).agg({'SAP_SOURCE_CUSTOMER_ID':lambda x:x.unique()}).reset_index()
          
    return df_RAW, df_main_filtered2, df_train 

def prepare_train_test_data(df_main):

    # PREPARING DATA FOR TRAINING & PREDICTION
    print(f"First BillDate Available in dataset: {df_main['BILL_DATE'].min()}")
    print(f"Last BillDate Available in dataset : {df_main['BILL_DATE'].max()}")
    obs_period   = df_main['BILL_DATE'].max()
    df_prepare   = summary_data_from_transaction_data(df_main, 
                                                     'CUSTOMER_ID',
                                                     'BILL_DATE', 
                                                      monetary_value_col= 'NET_SALES_AMOUNT',
                                                      observation_period_end= obs_period,
                                                      freq='W').reset_index()
    
    # data should have at-least One Repeat Purchase
    df_train = df_prepare[df_prepare['frequency']>0].reset_index(drop=True)
    
    print('prepared data for prediction (Filtered Data to have atleast 1 Repeat Purchase)')
    print(f'Shape before filtering: {df_prepare.shape}')
    print(f'Shape after filtering:{df_train.shape}')

    return df_train


if __name__=='__main__':
    
    #Initializing run object
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    #input
    parser.add_argument("--input_data_eu"  , type=str, help="EU input data")
    parser.add_argument("--input_data_apac", type=str, help="APAC input data")
    parser.add_argument("--input_data_la"  , type=str, help="LA input data")
    # countries Tenant wise
    parser.add_argument("--eu_tenant",  type=str,  help= 'String of EU Tenant Countries')
    parser.add_argument('--la_tenant',  type=str,  help= 'String of LA Tenant Countries')
    parser.add_argument('--apac_tenant',type=str,  help= 'String of APAC Tenant Countries')
    #output
    parser.add_argument("--output", type=str, help="output directory")
    args = parser.parse_args()

    # print("Argument 1: %s" % args.input_data)
    # print("Argument 2: %s" % args.output)
    # print("Argument 1: %s" %  arg.input_data_eu.to_pandas_dataframe().head())
    
    # converting to dataframes
    df_eu   = run.input_datasets['raw_data_eu'].to_pandas_dataframe()
    df_la   = run.input_datasets['raw_data_la'].to_pandas_dataframe()
    df_apac = run.input_datasets['raw_data_apac'].to_pandas_dataframe()

    # sometimes LA TENANT COUNTRIES DATA may present in EU TENANT
    # making sure picking EU TENATNT COUNTRIES FROM EU TENANT, LA TENANT COUNTRIES FROM LA AND APAC TENANT COUNTRIES FROM APAC
    # Converting String to List
    def String_to_List(string):
        l = list(string.split(" "))
        return l

    # TENANT COUNTRIES LIST
    EU_COUNTRIES   = String_to_List(args.eu_tenant)
    LA_COUNTRIES   = String_to_List(args.la_tenant)
    APAC_COUNTRIES = String_to_List(args.apac_tenant)

    df_eu   = df_eu[df_eu['COUNTRY'].isin(EU_COUNTRIES)]
    df_la   = df_la[df_la['COUNTRY'].isin(LA_COUNTRIES)]
    df_apac = df_apac[df_apac['COUNTRY'].isin(APAC_COUNTRIES)]
    
    # EU
    print(f"Countries in EU: {df_eu['COUNTRY'].unique()}")
    print(f"EU DF SHAPE  : {df_eu.shape}")
    print(f"MIN BILL DATE: {df_eu['BILL_DATE_YYYY_MM_DD'].min()}")
    print(f"MAX BILL DATE: {df_eu['BILL_DATE_YYYY_MM_DD'].max()}")

    # LA
    print(f"\nCountries in LA: {df_la['COUNTRY'].unique()}")
    print(f"LA DF SHAPE  : {df_la.shape}")
    print(f"MIN BILL DATE: {df_la['BILL_DATE_YYYY_MM_DD'].min()}")
    print(f"MAX BILL DATE: {df_la['BILL_DATE_YYYY_MM_DD'].max()}")

    # APAC
    print(f"\nCountries in APAC: {df_apac['COUNTRY'].unique()}")
    print(f"APAC DF SHAPE: {df_apac.shape}")
    print(f"MIN BILL DATE: {df_apac['BILL_DATE_YYYY_MM_DD'].min()}")
    print(f"MAX BILL DATE: {df_apac['BILL_DATE_YYYY_MM_DD'].max()}")

    # concatenating
    df_RAW = pd.concat([df_eu, df_apac,df_la], axis=0) # 
    
    # unique countries
    country_list =  df_RAW['COUNTRY'].unique().tolist()
    print(f'Country List: {country_list}')

    # calling function
    df_raw, df_main, df_train = prepare_data(df_RAW)
    
    # saving data
    if not (args.output  is None):
        os.makedirs(args.output, exist_ok=True)
        print(f"created output folder: {args.output}")

    save_path1  = os.path.join(args.output , f'raw.csv')
    save_path2  = os.path.join(args.output , f'main.csv')
    save_path3  = os.path.join(args.output , f'processed_data.csv')

    df_raw.to_csv (save_path1,  header=True, index=False)
    df_main.to_csv(save_path2,  header=True, index=False)
    df_train.to_csv(save_path3, header=True, index=False)

    run.complete()
    
    

        
        
        
