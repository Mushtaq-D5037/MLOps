# import libraries
#Azure
from azureml.core import Run
#python
import os
import pandas as pd
import numpy as np
import argparse
import shutil
from datetime import datetime, timedelta 
#sqlalchemy
from sqlalchemy import create_engine 
import urllib


# defining functions
def create_columns(df_main):
    '''
    1. creating LAST PURCHASE DATE Column and 
    2. LAST SEEN SINCE Column
    '''
          
    # 1. LAST PURCHASE DATE OR LAST BILL DATE
    df_lastBD = df_main.groupby('CUSTOMER_ID').agg({'BILL_DATE':lambda x: x.max() }).reset_index()
    df_lastBD.columns = ['CUSTOMER_ID','LAST_PURCHASE_DATE']
   
    # 2. LAST SEEN SINCE
    OBS_END_DATE = df_main['BILL_DATE'].max()
    
    # dividiing by 7 to get result in weeks
    df_lastBD['LAST_SEEN_SINCE_IN_WEEKS'] =  (OBS_END_DATE - df_lastBD['LAST_PURCHASE_DATE'])/timedelta(days=7)
                 
    print('columns LAST SEEN SINCE IN WEEKS & LAST PURCHASE DATE Created')
    
    return df_lastBD


def avg_purchase_pattern(df_main):
    # CREATING LAG BILL DATE
    df2 = df_main.copy()
    df2['LAG_BILL_DATE'] = df2.sort_values('BILL_DATE',ascending=True).groupby(['CUSTOMER_ID'])['BILL_DATE'].shift(1)

    # INTER PURCHASE DAYS
    df2['INTER_PURCHASE_DAYS'] = df2['BILL_DATE'] - df2['LAG_BILL_DATE']
    df2['INTER_PURCHASE_DAYS'] = df2['INTER_PURCHASE_DAYS'].dt.days
    
    # Calculating AVG PURCAHSE PATTERN IN WEEKS
    df2['INTERPURCHASE_IN_WEEKS'] = np.round((df2['BILL_DATE'] - df2['LAG_BILL_DATE'])/timedelta(days=7),3)
    df_interpurchase_count = df2.groupby('CUSTOMER_ID').agg({'INTERPURCHASE_IN_WEEKS': [lambda x: x.count(),
                                                                                        lambda x: x.sum()],}).reset_index()

    df_interpurchase_count.columns =['CUSTOMER_ID','INTERPURCHASE_IN_WEEKS_COUNT', 'INTERPURCHASE_IN_WEEKS_SUM']
    df2 = df2.merge(df_interpurchase_count, on='CUSTOMER_ID', how='left')
    df2['AVG_PURCHASE_PATTERN_IN_WEEKS'] = np.round(df2['INTERPURCHASE_IN_WEEKS_SUM']/df2['INTERPURCHASE_IN_WEEKS_COUNT'], 2)
    
    # grouping
    df3 = df2.groupby('CUSTOMER_ID').agg({'AVG_PURCHASE_PATTERN_IN_WEEKS': lambda x : x.unique()}).reset_index()
    df3 = df3[['CUSTOMER_ID','AVG_PURCHASE_PATTERN_IN_WEEKS']]

    print('column AVG_PURCHASE_PATTERN_IN_WEEKS created ')
    
    return df3

def calculate_monthly_avg_sales(df_main):
    ''' 
    AVG_MONTHLY_SALE = TOTAL SALE AMOUNT / NO_OF_MONTHS TRANSACTED
    Ex: IF trxn months are 4JAN 2019, 10JULY 2021, 11AUG 2021 then NO_OF_MONTHS TRANSACTED = 3
    '''
    
    df4 = df_main.copy()

    df4['BILL_MONTH'] = df4['BILL_DATE'].dt.month
    df4['BILL_YEAR']  = df4['BILL_DATE'].dt.year

    df4 = df4.groupby(['CUSTOMER_ID','BILL_MONTH','BILL_YEAR']) \
             .agg({'NET_SALES_AMOUNT':['sum'],'NET_SALES_UNITS':['sum']}).reset_index()

    df4.columns =['CUSTOMER_ID','BILL_MONTH','BILL_YEAR','NET_SALES_AMOUNT','NET_SALES_UNITS']
    df4 = df4.groupby('CUSTOMER_ID') \
             .agg({'BILL_MONTH': lambda x : x.count(),
                   'NET_SALES_AMOUNT':['sum'],
                   'NET_SALES_UNITS':['sum']}).reset_index()
    #renaming columns
    df4.columns =['CUSTOMER_ID','NO_OF_MONTHS','TOTAL_AMOUNT','TOTAL_UNITS']
    
    # calculating avg monthly sale
    df4['AVG_MONTHLY_SALE'] = df4['TOTAL_AMOUNT']/df4['NO_OF_MONTHS']
    
    print('AVG MONTHLY SALES column Created')
    
    return df4


    
def monthly_sale_breakup(df_main):
    
    # MONTHLY AMOUNT BREAKUP
    df_raw2 = df_main.copy()
    
    # creating MONTH and YEAR Column
    df_raw2['MONTH'] = df_raw2['BILL_DATE'].dt.month
    df_raw2['YEAR']  = df_raw2['BILL_DATE'].dt.year
    
    # observation End date
    obs_end_date = df_main['BILL_DATE'].max()
    print(f'Last Bill Date Available in Dataset: {obs_end_date}')
    
    # getting total days of a month
    cur_year      = obs_end_date.year
    cur_month     = obs_end_date.month
    cur_day       = obs_end_date.day

    cur_period    = pd.Period(str(obs_end_date))
    curYear_month_totaldays = cur_period.daysinmonth

    obs_end_date2 = pd.to_datetime(f'{cur_year}-{cur_month}-{curYear_month_totaldays}')
    print(f'Observation End date:{obs_end_date2}')

    # last 12 months from observation End date
    last_12month_from_obs = obs_end_date2 + timedelta(days=-365)
    
    # below code is to take care FEBRUARY MONTH Cacluation
    prev_year         = last_12month_from_obs.year
    prev_obs_end_date = pd.to_datetime(f'{prev_year}-{cur_month}-{cur_day}')                         
    prev_period       = pd.Period(str(prev_obs_end_date))
    prevYear_month_totaldays = prev_period.daysinmonth

    last_12month_from_obs2 = pd.to_datetime(f'{prev_year}-{cur_month}-{prevYear_month_totaldays}')
    print(f'Last 12 Months Date from Observation End Date:{last_12month_from_obs2}')

    # taking last one year data from observation date and calculating last 12month total sale
    df_raw2 = df_raw2[df_raw2['BILL_DATE']>last_12month_from_obs2]
    print(f"Min Date: {df_raw2['BILL_DATE'].min()}")
    print(f"Max Date: {df_raw2['BILL_DATE'].max()}")

    df_sales =  df_raw2.groupby('CUSTOMER_ID').agg({'NET_SALES_AMOUNT':'sum'}).reset_index()
    df_sales.columns = ['CUSTOMER_ID','LAST_12_MONTH_TOTAL_SALES']
    print('LAST 12 MONTH TOTAL SALES Column Created')
    
    # creating monthly sales from last one year data
    df_raw3 = df_raw2.groupby(['CUSTOMER_ID','MONTH','YEAR']).agg({'NET_SALES_AMOUNT':'sum'}).reset_index()
    
    # calculating last 12 months sale individually
    c = 12
    for i in range(0,12):
        
        # for labelling column name
        j = i + 1 
        
        # calculating last month sales from observation end month, 
        # Note: observation end month is last 1st month
        # example: feb is observation end month, then last_month1 = feb, last_month2 = jan, last_month3 = dec
        m = obs_end_date.month - i
        
        # to avoid 0 & Negative Numbers
        # example: for feb, 2-2=0, which means 12months, so taking 12th month and decreasing from 12th month
        if m <= 0:
            m = c
            c -= 1
        
        df_last_month_j_sale = df_raw3[df_raw3['MONTH']==m]
        df_last_month_j_sale.columns = ['CUSTOMER_ID','MONTH','YEAR',f'LAST_MONTH_{j}_SALES']
        
        # merging 
        df_sales = df_sales.merge(df_last_month_j_sale[['CUSTOMER_ID',f'LAST_MONTH_{j}_SALES']], on=['CUSTOMER_ID'], how='left')
    
        print(f'Calculated LAST_MONTH_{j}_SALES')
    
    return df_sales


def create_final_result(df_result, df_raw, df4, df_lastBD, df_purchase_pattern, df_sale, model_version):
        ''' 
            Creating Final data  
            1. INNER MERGE RESULTS WITH RAW DATA 
            2. LEFT MERGE WITH AVG_MONTHLY_$
            3. LEFT MERGE WITH LAST_SEEN_SINCE 
            4. LEFT MERGE WITH AVERAGE PURCHASE PATTERN
            5. LEFT MERGE WITH MONTHLY SALE BREAKUP
            6. CREATE RATIO COLUMN [LAST SEEN / AVG PURCHASE ]
            7. Filtering DORMANT ACCOUNT (Not ACTIVE in last 52Weeks = 364Days) Using LAST_SEEN_SINCE 
         '''
        
        # INNER MERGE RAW DATA
        df_result_final = df_result.merge(df_raw, on ='CUSTOMER_ID', how='inner')   
        print('Merged: SAP ID')
        
        # AVG MONTHLY SALE
        df4       = df4[['CUSTOMER_ID','AVG_MONTHLY_SALE']]
        df_result_final = df_result_final.merge(df4, on ='CUSTOMER_ID', how='left')  
        print('Merged: AVG_MONTHLY_SALE')
        
        # LAST SEEN SINCE
        df_lastBD = df_lastBD[['CUSTOMER_ID','LAST_PURCHASE_DATE','LAST_SEEN_SINCE_IN_WEEKS']]
        df_result_final = df_result_final.merge(df_lastBD, on='CUSTOMER_ID', how='left') 
        print('Merged: LAST_SEEN_SINCE, LAST_PURCHASE_DATE Columns')
        
        # AVG PURCHASE PATTER
        df_result_final = df_result_final.merge(df_purchase_pattern, on='CUSTOMER_ID',how='left') 
        print('Merged: AVERAGE PURCHASE PATTERN')
        
        # MONTHLY SALE BREAKUP
        df_result_final = df_result_final.merge(df_sale, on='CUSTOMER_ID', how='left') 
        print('Merged: MONTHLY SALE BREAKUP')
        
        # RATIO COLUM
        df_result_final['RATIO'] = df_result_final['LAST_SEEN_SINCE_IN_WEEKS'] / df_result_final['AVG_PURCHASE_PATTERN_IN_WEEKS']
        print('Created RATIO Column')

        # adding other columns
        df_result_final['VALUE_PRIORITY']  = np.nan
        df_result_final['RISK_PRIORITY']   = np.nan
        df_result_final['PRIORITY_BUCKET'] = np.nan
        df_result_final['MODEL_VERSION']   = model_version 
        df_result_final['EXECUTION_DATE']  = datetime.now()
        
        print(f'MODEL VERSION: {model_version}')

        # re-arranging columns
        print('Re-Arranging columns')
        df_result_final = df_result_final[[ 'COUNTRY', 'CUSTOMER_ID', 'SAP_SOURCE_CUSTOMER_ID',  
                                            'FREQUENCY_IN_WEEKS','RECENCY_IN_WEEKS', 'TENURE_IN_WEEKS', 'LAST_PURCHASE_DATE',
                                            'LAST_SEEN_SINCE_IN_WEEKS', 'AVG_PURCHASE_PATTERN_IN_WEEKS', 'RATIO',
                                            'PROBABILITY_OF_CHURN','EXPECTED_NUMBER_OF_PUCHASES_NEXT_12WEEKS',
                                            'PREDICTED_PURCHASE_IN_NEXT_12WEEKS', 'VALUE_PRIORITY', 'RISK_PRIORITY',
                                            'PRIORITY_BUCKET', 'AVG_MONTHLY_SALE', 'LAST_MONTH_12_SALES',
                                            'LAST_MONTH_11_SALES', 'LAST_MONTH_10_SALES', 'LAST_MONTH_9_SALES',
                                            'LAST_MONTH_8_SALES', 'LAST_MONTH_7_SALES', 'LAST_MONTH_6_SALES',
                                            'LAST_MONTH_5_SALES', 'LAST_MONTH_4_SALES', 'LAST_MONTH_3_SALES',
                                            'LAST_MONTH_2_SALES', 'LAST_MONTH_1_SALES', 'LAST_12_MONTH_TOTAL_SALES',
                                            'MODEL_VERSION', 'EXECUTION_DATE'
                                          ]]

        print('Filtering data to have accounts which are active in last 12 months')
        print(f'Dataframe Shape Before Filter last 12Months: {df_result_final.shape}')
        
        df_result_final = df_result_final[df_result_final['LAST_SEEN_SINCE_IN_WEEKS']<=52]
        
        print(f'Dataframe Shape after filter: {df_result_final.shape}')
        
        return df_result_final



if __name__=='__main__':

    #Initializing run object
    run = Run.get_context()
 
    # establishing sql server connection: 
    # EU
    server_eu   = 
    database_eu = 
    username_eu = 
    password_eu = 
    env_eu      = 

    # LA
    server_la   = 
    database_la = 
    username_la = 
    password_la = 
    env_la      = 

    # APAC
    server_apac   = 
    database_apac = 
    username_apac = 
    password_apac =  
    env_apac      = 


    # sqlalchemy
    # EU
    params_eu = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};'+ 'SERVER='+server_eu+';DATABASE='+database_eu+';UID='+username_eu+';PWD='+ password_eu) 
    engine_eu = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_eu,fast_executemany=True) 

    # #APAC
    params_apac = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};'+ 'SERVER='+server_apac+';DATABASE='+database_apac+';UID='+username_apac+';PWD='+ password_apac) 
    engine_apac = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_apac,fast_executemany=True) 

    # # LA
    params_la = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};'+ 'SERVER='+server_la+';DATABASE='+database_la+';UID='+username_la+';PWD='+ password_la) 
    engine_la = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_la,fast_executemany=True) 


    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data",   type=str,  help= 'output folder from pre_process step')
    parser.add_argument('--main_data',  type=str,  help= 'output folder from pre_process step')
    parser.add_argument('--result_data',type=str,  help= 'output folder from prediction step')
    parser.add_argument("--eu_tenant",  type=str,  help= 'String of EU Tenant Countries')
    parser.add_argument('--la_tenant',  type=str,  help= 'String of LA Tenant Countries')
    parser.add_argument('--apac_tenant',type=str,  help= 'String of APAC Tenant Countries')
    args = parser.parse_args()

    
    folder_path1 = os.path.join(args.raw_data)
    folder_path2 = os.path.join(args.result_data)

    # file path
    # getting filepath of raw data and main data
    for f in os.listdir(folder_path1):
        if 'raw' in f:
            filepath_raw  = os.path.join(folder_path1, f)
        elif 'main' in f:
            filepath_main = os.path.join(folder_path1, f)
    
    # getting filepath of result data
    for f in os.listdir(folder_path2):
        filepath_result = os.path.join(folder_path2, f)

    # loading data
    df_raw    = pd.read_csv(filepath_raw)
    df_main   = pd.read_csv(filepath_main)
    df_result = pd.read_csv(filepath_result)

    # For naming model version
    month_dict = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}

    # Changin data type
    df_main['BILL_DATE'] = pd.to_datetime(df_main['BILL_DATE'])
    obs_end_date = df_main['BILL_DATE'].max()
    year         = obs_end_date.year
    month        = month_dict[obs_end_date.month]
    day          = obs_end_date.day
    pred_period  = 'PP_12Weeks'
    
    # getting total days in that particular month of that year
    cur_period = pd.Period(str(obs_end_date))
    totaldays  = cur_period.daysinmonth

    # calling functions
    df_lastSeen = create_columns(df_main)
    df_sale     = monthly_sale_breakup(df_main)
    df_avgMonthlySale   = calculate_monthly_avg_sales(df_main)
    df_purchase_pattern = avg_purchase_pattern(df_main)
    
    
    # Converting String to List
    def String_to_List(string):
        l = list(string.split(" "))
        return l

    # TENANT COUNTRIES LIST
    EU_COUNTRIES   = String_to_List(args.eu_tenant)
    LA_COUNTRIES   = String_to_List(args.la_tenant)
    APAC_COUNTRIES = String_to_List(args.apac_tenant)
    
    print(f"EU TENANT:{EU_COUNTRIES}")
    print(f"LA TENANT:{LA_COUNTRIES}")
    print(f"APAC TENANT: {APAC_COUNTRIES}")

    unique_countries = df_main['COUNTRY'].unique().tolist()
    print(f"Unique Countries: {unique_countries}")
    
    for c in unique_countries:
        print('\n')
        print(c)
        model_version = f'ML_{c}_v2.0_01JAN2019_{day}{month}{year}_{pred_period}'
        df_raw2    = df_raw.copy()
        df_raw2    = df_raw2[df_raw2['COUNTRY']==c]
        print(f"{c} RAW DF SHAPE: {df_raw2.shape}")
        exec(f"df_result_{c} = create_final_result(df_result, df_raw2, df_avgMonthlySale, df_lastSeen, df_purchase_pattern, df_sale,f'{model_version}')")
        
        # storing results to db
        if c in EU_COUNTRIES:
            exec(f"df_result_{c}.to_sql('ML_OUTPUT_CHURN_PREDICTION', con=engine_eu, schema='C360', if_exists='append', index=False, chunksize=1000, method=None)")
            print(f'{c} results updated in output table')
        
        elif c in APAC_COUNTRIES:
            exec(f"df_result_{c}.to_sql('ML_OUTPUT_CHURN_PREDICTION', con=engine_apac, schema='C360', if_exists='append', index=False, chunksize=1000, method=None)")
            print(f'{c} results updated in output table')
        
        elif c in LA_COUNTRIES:
            exec(f"df_result_{c}.to_sql('ML_OUTPUT_CHURN_PREDICTION', con=engine_la, schema='C360', if_exists='append', index=False, chunksize=1000, method=None)")
            print(f'{c} results updated in output table')

    # close sql_alchemy engine
    engine_eu.dispose()
    engine_apac.dispose()
    engine_la.dispose()


    run.complete()
   
    
    
