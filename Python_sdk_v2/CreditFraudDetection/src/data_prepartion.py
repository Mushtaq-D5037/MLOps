# Importing Libraries
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

if __name__ == "__main__":

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--processed_data", type=str, help="path to processed data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)

    # loading data
    df = pd.read_csv(args.data)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features",df.shape[1] - 1)

    # dropping un-necessary columns
    drop_col = ['Time']
    df.drop(drop_col, axis=1, inplace=True)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df.to_csv(os.path.join(args.processed_data, "processed_data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()

    

    


    
