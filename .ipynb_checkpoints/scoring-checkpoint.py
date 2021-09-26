"""
Model Evaluation
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


# Function for model scoring
def score_model(data_folder, model_path, filename="testdata.csv"):
    """
    Take a trained model, load test data, and calculate an F1 score for the model relative to the test data and write the result to the latestscore.txt file
    """

    # read the model and the test data
    test_data = pd.read_csv(
        os.path.join(
            data_folder,
            filename)).set_index('corporation')
    y_data = test_data.pop('exited')
    y_data = y_data.values
    x_data = test_data.values
    with open(os.path.join(model_path, "trainedmodel.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    y_pred = model.predict(x_data)

    # Evaluation
    f1_score = metrics.f1_score(y_data, y_pred)
    with open(os.path.join(output_model_path, "latestscore.txt"), "w") as score_file:
        score_file.write(str(f1_score))
    return f1_score



if __name__ == '__main__':
    score_model(data_folder=test_data_path, model_path=output_model_path)
