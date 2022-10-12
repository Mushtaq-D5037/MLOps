# Training Model each time we run the pipeline
# Reason: when predicting using Trained Model, its giving 1-10% probability difference on new data

# Azure
from azureml.core import Workspace, Experiment, Run, Dataset, Model
#python
import os
import pandas as pd
import numpy as np
import joblib
import pyodbc
import argparse
from datetime import datetime, timedelta

# lifetimes package
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter


def model_training(df, weeks):

    print(f"dataframe shape, before dropping frequency>2 & recency>=12: {df.shape}")
    # Training data should have atleat frequency>2 and recency>=12 Weeks(suggested by business)
    df = df[(df['frequency']>2)&(df['recency']>=12)]
    print(f"dataframe shape, after dropping frequency>2 & recency>=12: {df.shape}")

    # Training the Model
    model = BetaGeoFitter(penalizer_coef=0.01)
    model.fit(df['frequency'],
              df['recency'],
              df['T'])
    
    print('Model Training Completed')
    # Note:
    # Every time we run the pipeline we are re-training the model
    # Reason: on Anlaysis, we found there is significant amount of variation/change(upto 10%) in probabilities with respect to customer id 
    #         when predicted using trained model, because of which customer importance is changing
    # hence re-training model 
    
    # calling prediction fuction
    df_result = model_prediction(df,model,weeks)

    return df_result

    
def model_prediction(df, model,weeks):
    
    # copying filtered train data 
    df_result = df.copy()

    #Fetching the Output for absolute churn
    df_result['PROB_ALIVE']=model.conditional_probability_alive(df_result['frequency'], 
                                                                df_result['recency'], 
                                                                df_result['T']
                                                                 )

    df_result[f'PROBABILITY_OF_CHURN'] = 1 - df_result['PROB_ALIVE']

    # Fetching the output for expected number of purchases in next 90 days
    df_result[f'EXPECTED_NUMBER_OF_PUCHASES_NEXT_{weeks}WEEKS']= model.conditional_expected_number_of_purchases_up_to_time(weeks, 
                                                                                                                           df_result['frequency'], 
                                                                                                                           df_result['recency'], 
                                                                                                                           df_result['T']
                                                                                                                      )
    # Prediction Logic: ExpectedNumberPurchase > 1 -->1 ELSE 0 
    df_result[f'PREDICTED_PURCHASE_IN_NEXT_{weeks}WEEKS'] = df_result[f'EXPECTED_NUMBER_OF_PUCHASES_NEXT_{weeks}WEEKS'].apply(lambda x: 1 if x>1 else 0)
    
    # Renaming columns
    df_result.rename(columns={'frequency':'FREQUENCY_IN_WEEKS',
                              'recency':'RECENCY_IN_WEEKS',
                              'T':'TENURE_IN_WEEKS'}, inplace=True)
    
    # taking only required columns
    df_result = df_result[['CUSTOMER_ID','FREQUENCY_IN_WEEKS','RECENCY_IN_WEEKS','TENURE_IN_WEEKS', 'PROBABILITY_OF_CHURN', 
                           f'EXPECTED_NUMBER_OF_PUCHASES_NEXT_{weeks}WEEKS',f'PREDICTED_PURCHASE_IN_NEXT_{weeks}WEEKS']]
    
    print('Model Predictions Completed')
    
    return df_result


if __name__=='__main__':

    # initializing run
    run = Run.get_context()


    # # load the model
    # global model
    # model_path = Model.get_model_path('CHURN_PREDICTION_BGNBD')
    # model = joblib.load(model_path)

    # arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="output data from previous step is used as input")
    parser.add_argument("--output",     type=str, help="output directory")

    args = parser.parse_args()

    # getting list of dir
    output_files = os.listdir(args.input_data)
    print(f'Pre-process Step Output files: {output_files}')

     # creating output folder
    if not (args.output is None):
        os.makedirs(args.output, exist_ok=True)

    # input file path
    input_folder_path = os.path.join(args.input_data)
    print(f"Training & Prediction input folder path:{input_folder_path}")

    for f in os.listdir(input_folder_path):
        if 'processed' in f:
            input_file_path = os.path.join(input_folder_path, f)
            df = pd.read_csv(input_file_path)
            print(df.head())
            
            # calling function
            df_result = model_training(df, weeks=12)
            print(df_result.head())

            # Saving
            save_path  = os.path.join(args.output, 'result_data.csv')
            print(save_path)
            df_result.to_csv(save_path, header=True, index=False)
        
    run.complete()
    
    
       

        
    
