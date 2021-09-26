from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


# function for deployment
def deploy_model_into_production(model_folder, prod_folder):
    """
    copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """

    # Prepare file paths
    file_paths = [
        (os.path.join(
            dataset_csv_path, "ingestedfiles.txt"), os.path.join(
            prod_folder, "ingestedfiles.txt")), (os.path.join(
                model_path, "trainedmodel.pkl"), os.path.join(
                    prod_folder, "trainedmodel.pkl")), (os.path.join(
                        model_path, "latestscore.txt"), os.path.join(
                            prod_folder, "latestscore.txt"))]

    # copy files to production
    if not os.path.exists(prod_folder):
        os.mkdir(prod_folder)
    for file_path in file_paths:
        source, destination = file_path
        os.system(f'cp {source} {destination}')


if __name__ == '__main__':
    deploy_model_into_production(
        model_folder=model_path,
        prod_folder=prod_deployment_path)
