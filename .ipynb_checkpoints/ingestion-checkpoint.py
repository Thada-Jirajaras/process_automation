'''
Ingest the data
to clean this file try
autopep8  --in-place --aggressive --aggressive ingestion.py
pylint --errors-only ingestion.py
'''

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


# Load config.json and get input and output paths
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe(input_folder, output_folder):
    """
    check for datasets, compile them together, and write to an output file
    """

    # Read and merge files
    final_dataframe = pd.DataFrame()
    input_files = []
    filenames = os.listdir(input_folder)
    for each_filename in filenames:
        if each_filename.endswith('.csv'):
            input_file = os.path.join(input_folder, each_filename)
            currentdf = pd.read_csv(input_file)
            input_files.append(input_file)
            final_dataframe = final_dataframe.append(
                currentdf).reset_index(drop=True)
    final_dataframe.drop_duplicates(inplace=True)

    # Save file
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    final_dataframe.to_csv(
        os.path.join(
            output_folder,
            "finaldata.csv"),
        index=False)
    with open(os.path.join(output_folder, "ingestedfiles.txt"), "w") as list_file:
        list_file.write("\n".join(input_files))


if __name__ == '__main__':
    merge_multiple_dataframe(
        input_folder=input_folder_path,
        output_folder=output_folder_path)
