# Importing Libraries
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import mlflow
import logging


if __name__ == "__main__":

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to processed data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # getting list of dir
    files_list = os.listdir(args.input_data)
    print(f'Previous Step Output: {files_list}')

    # creating output folder
    # if not (args.output is None):
    #     os.makedirs(args.output, exist_ok=True)

    # input file path
    input_folder_path = os.path.join(args.input_data)
    print(f"Training Data folder path:{input_folder_path}")
    
    # Loading processed data
    for f in os.listdir(input_folder_path):
        if 'processed_data' in f:
            data_file = os.path.join(input_folder_path, f)
            df = pd.read_csv(data_file)

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = args.test_train_ratio, 
                                                        random_state = 42,  
                                                        stratify = y
                                                       )

    # Training Model
    model = RandomForestClassifier(class_weight='balanced',
                                    bootstrap=True,
                                    max_depth=100,
                                    max_features=2,
                                    min_samples_leaf=5,
                                    min_samples_split=10,
                                    n_estimators=1000,
                                    random_state = 42
                                    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


    # Registering Model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model( sk_model=model,
                              registered_model_name=args.registered_model_name,
                              artifact_path=args.registered_model_name,
                            )

    # Saving the model to a file
    mlflow.sklearn.save_model( sk_model=model,
                               path=os.path.join(args.model, "trained_model"),
                             )

    # Stop Logging
    mlflow.end_run()


