import os
import json
import requests


def call(apireturn_file="apireturns.txt"):
    # Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1/"

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_model_path = os.path.join(config['output_model_path'])
    test_data_path = os.path.join(config['test_data_path'])

    # Call each API endpoint and store the responses
    response1 = requests.post(
        "http://0.0.0.0:8000/prediction",
        json={
            "dataset_path": os.path.join(
                test_data_path,
                "testdata.csv")}).text
    response2 = requests.get("http://0.0.0.0:8000/scoring").text
    response3 = requests.get("http://0.0.0.0:8000/summarystats").text
    response4 = requests.get("http://0.0.0.0:8000/diagnostics").text

    # combine all API responses
    responses = [response1, response2, response3, response4]

    # write the responses to your workspace
    with open(os.path.join(output_model_path, apireturn_file), "w") as apireturns_file:
        apireturns_file.write("\n".join(responses))

if __name__ == '__main__':
    call()