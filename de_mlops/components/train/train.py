# Import libraries

import argparse
import glob
import os

import mlflow


import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



os.makedirs("./outputs", exist_ok=True)
mlflow.start_run()
# enable autologging
mlflow.sklearn.autolog()

# define functions
def main(args):
    # TO DO: enable autologging
    # Start Logging
    # run_name = args.registered_model_name 

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args, X_train, X_test, y_train, y_test)

    # Stop Logging
    # mlflow.end_run()


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    # csv_files = glob.glob(f"{path}/*.csv")
    # if not csv_files:
    #     raise RuntimeError(f"No CSV files found in provided data path: {path}")
    # return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)
    return pd.read_csv(path)


# TO DO: add function to split data
def split_data(df):
    y = df['Diabetic']
    X = df.iloc[:, :-1]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(args, X_train, X_test, y_train, y_test):
    # Train model
    model = LogisticRegression(C=1/args.reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    # Evaluate the model and log metrics
    accuracy = model.score(X_test, y_test)
    print(accuracy)
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
        # artifact_path = os.path.join(args.model, "trained_model_" + args.registered_model_name),
    )
    mlflow.log_param('param', args.reg_rate)
    mlflow.log_metric('accuracy', accuracy)
    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.model, "trained_model_" + args.registered_model_name),
    )



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)
    
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")