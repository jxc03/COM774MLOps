import os
import json
import requests
import pandas as pd

def main():
    # Config
    scoring_uri = "https://<your-endpoint-name>.<region>.inference.ml.azure.com/score"
    api_key = "<YOUR_ENDPOINT_KEY>" # better: read from env var

    if api_key.startswith("<"):
        raise ValueError("Please set your endpoint URL and key in call_endpoint.py")

    # Build sample input load
    # Load one row from X_test CSV
    X_test_path = "data_processed_split/X_test_classification.csv" 
    X_test = pd.read_csv(X_test_path)

    # Take a single row to send to the endpoint
    sample_row = X_test.iloc[[0]] # [[0]] keeps it as DataFrame with one row

    # Convert to list-of-dicts (one dict per row). This is easy to inspect
    input_data = sample_row.to_dict(orient="records")

    payload = {
        "input_data": input_data
    }

    print("Sample payload being sent:")
    print(json.dumps(payload, indent=2)[:800], "\n")

    # Send request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(scoring_uri, headers=headers, json=payload)

    print("Status code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception:
        print("Raw response text:", response.text)


if __name__ == "__main__":
    main()
