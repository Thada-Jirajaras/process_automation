import os
import json
import pickle
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
#import create_prediction_model
#import diagnosis
#import predict_exited_from_saved_model
from diagnostics import model_predictions, dataframe_summary
from diagnostics import  execution_time, outdated_packages_list
from scoring import score_model

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)
dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    call the prediction function you created in Step 3

    Return:
        predicted_results: value for prediction outputs
    """

    data = pd.read_csv(request.json.get('dataset_path')
                       ).set_index('corporation')
    predicted_results = model_predictions(
        model_folder=prod_deployment_path, data=data)
    return str(predicted_results)

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    check the score of the deployed model
    """
    f1_score = score_model(
        data_folder=test_data_path,
        model_path=prod_deployment_path)
    return str(f1_score)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    check means, medians, and modes for each column
    
    Returns:
        summary_stat: (dict) a dict of lists of all calculated summary statistics
    """
    
    summary_stat = dataframe_summary(data_folder=dataset_csv_path)
    return  json.dumps(summary_stat)

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    """ 
    check timing and percent NA values
    """
    na_percent = dataframe_summary(data_folder=dataset_csv_path)["na_percent"]
    outdated = outdated_packages_list()
    timing = execution_time()
    return   json.dumps({"execution_time": timing, "na_percent": na_percent, "outdated_packages_list": outdated})  


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
