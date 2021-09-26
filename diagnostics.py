import os
import subprocess
import json
import pickle
import pandas as pd
import numpy as np
import timeit


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']

# Function to get model predictions


def model_predictions(model_folder, data):
    """
    read the deployed model and a test dataset, calculate predictions
    Args:
      model_folder: the deployed model path
      data: (dataframe) the dataset for prediction

    Return:
      y_pred: (list) a list containing all predictions
    """

    y_data = data.pop('exited')
    y_data = y_data.values
    x_data = data.values
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    y_pred = list(model.predict(x_data))
    return y_pred  # return value should be a list containing all predictions

# Function to get summary statistics


def dataframe_summary(data_folder):
    """
    calculate summary statistics here

    Returns:
      summary_stat: (dict) a dict containing all summary statistic lists of
        the data such as num_sample, mean, medium, std, and na_percent

    """
    data = pd.read_csv(
        os.path.join(
            data_folder,
            "finaldata.csv")).set_index('corporation')
    summary_stat = data.describe()
    mean = data.mean()
    median = data.median()
    std = data.std()
    na_percent = (data.isna().sum() / len(data))
    num_sample = len(data)
    summary_stat =  {
        'num_sample': num_sample,
        'mean': mean.to_list(),
        "median": median.to_list(),
        "std": std.to_list(),
        "na_percent": na_percent.to_list()}
    return summary_stat

# Function to get timings


def execution_time():
    """
    calculate timing of training.py and ingestion.py

    Returns:
      timing_values: (list) a list of 2 timing values in seconds

    """
    timing_values = []
    for file_name in ["training.py", "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system(f'python {file_name}')
        timing = timeit.default_timer() - starttime
        timing_values.append(timing)
    return timing_values

# Function to check dependencies


def outdated_packages_list():
    """
    get a list of outdated_packages
    """
    outdated = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode("utf-8")
    return outdated


if __name__ == '__main__':

    # read data
    test_data = pd.read_csv(
        os.path.join(
            test_data_path,
            "testdata.csv")).set_index('corporation')

    # run functions
    model_predictions(model_folder=prod_deployment_path, data=test_data)
    dataframe_summary(data_folder=output_folder_path)
    execution_time()
    outdated_packages_list()
