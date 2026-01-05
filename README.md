# COM774MLOps

<i>Logging into Azure</i>
1. Open Windows Powershell
2. az login

<i>Running training job with MLflow through Azure ML</i>
1. Create a job.yaml file for Azure ML based on:
https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2
2. Then submit:
az ml job create --file job.yaml --resource-group <your-resouce-group> --workspace-name <your-workspace>  #Make sure to run this from inside the "production" directory
az ml job create --file ml/job.yaml --resource-group cw2-mlops --workspace-name cw2-mlops #'ml/' since job.yaml is in ml folder

<i>Deploying the model</i>
# Register the model from your MLflow run
az ml model create --name incident-priority-model --version 1 --path azureml://jobs/<job-id>/outputs/model --resource-group cw2-mlops --workspace-name <your-workspace>

<i>For testing</i>
1. Go to file and run file 
2. Or 'cd test' then 
python -m pytest test_unit_data.py test_unit_model.py -v --tb=short > test_data_model_results.txt #To run unit tests and capture output 
python -m pytest test_integration_pipeline.py -v --tb=short >> test_inte_pipe_results.txt #To run integration test