"""
Process Automation

to clean this file try
pylint --errors-only fullprocess.py
autopep8  --in-place --aggressive --aggressive fullprocess.py
"""
import os
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get input and output paths
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Check and read new data
# first, read ingestedfiles.txt
with open(os.path.join(output_folder_path, "ingestedfiles.txt"), 'r') as file:
    ingestedfiles = file.read().splitlines()

# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
filenames = os.listdir(input_folder_path)
input_files = {os.path.join(input_folder_path, each_filename)
               for each_filename in filenames if each_filename.endswith('.csv')}
newfiles = set(input_files) - set(ingestedfiles)

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(newfiles) > 0:
    logger.info("New input files detected")
    ingestion.merge_multiple_dataframe(
        input_folder=input_folder_path,
        output_folder=output_folder_path)
    #new_data = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    #train_data, test_data = train_test_split(
    #    new_data, test_size=0.2, random_state=42, stratify=new_data['exited'])
    #train_data.to_csv(
    #    os.path.join(
    #        output_folder_path,
    #        "finaldata.csv"),
    #    index=False)
    #test_data.to_csv(os.path.join(test_data_path, "testdata.csv"), index=False)

    # Checking for model drift
    # check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    with open(os.path.join(prod_deployment_path, "latestscore.txt"), 'r') as f:
        latest_f1 = float(f.read())
    new_f1 = scoring.score_model(
        data_folder=output_folder_path,
        model_path=prod_deployment_path,
        filename="finaldata.csv")
    logger.info(f"new_f1={new_f1}, latest_f1={latest_f1}")
    if new_f1 < latest_f1:
        IS_MODELDRIFT = 1
    else:
        IS_MODELDRIFT = 0

    # Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the
    # process here
    if IS_MODELDRIFT:
        logger.info("Model drift detected")
        training.train_model(
            input_folder=dataset_csv_path,
            model_folder=output_model_path)
        scoring.score_model(
            data_folder=test_data_path,
            model_path=output_model_path)

        # Re-deployment
        # if you found evidence for model drift, re-run the deployment.py
        # script
        deployment.deploy_model_into_production(
            model_folder=output_model_path,
            prod_folder=prod_deployment_path)

        ##################Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model
        reporting.report_model(
            data_folder=test_data_path,
            model_folder=output_model_path,
            confusionmatrix_name="confusionmatrix2.png")
        
        # Warining: api need to be started before running this process
        apicalls.call(apireturn_file="apireturns2.txt")
