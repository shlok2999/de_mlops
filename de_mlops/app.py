from flask import Flask, render_template, request
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
from azure.ai.ml import load_component
from azure.ai.ml import dsl, Input, Output
# import requests
import json
import ast
import os

app = Flask(__name__)


custom_env = 'mlflow-env'
dependencies_dir = './dependencies'
cpu_cluster_name = 'cpu-cluster-00'
workspace_name = 'de-mlops'
subscription_id = "Your Subscription ID"
resource_group = "MLOPS"
'''
route to land on the index page
'''
@app.route("/")
def hello_world():
    return render_template('index.html')

'''
route which collect the values, wraps them as a json, sends to endpoint api to predict the result.
'''
@app.route('/predict', methods=['POST'])
def predict():
    # collecting values
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['PlasmaGlucose'])
    dbp = float(request.form['DiastolicBloodPressure'])
    triceps_thickness = float(request.form['TricepsThickness'])
    serum_insulin = float(request.form['SerumInsulin'])
    bmi = float(request.form['BMI'])
    diabetes_pedigree = float(request.form['DiabetesPedigree'])
    age = float(request.form['Age'])
    id = float(request.form['ID'])


    # wrapping it to send it to api
    data_arr=[id,pregnancies,glucose,dbp,triceps_thickness,serum_insulin,bmi,diabetes_pedigree,age]
    np_arr=np.array([data_arr])
    input_value={
        "columns": [0,1,2,3,4,5,6,7,8],
        "index": [0],
        "data":np_arr.tolist()
    }
    req_file = {
        "input_data":input_value,
        "params": {}
    }
    with open('text.json', 'w') as json_file:
        json.dump(req_file, json_file)


    credential = AzureCliCredential()


    ml_client = MLClient(
        credential=credential,
        subscription_id= subscription_id,
        resource_group_name= resource_group,
        workspace_name= workspace_name,
    )
    # Getting the result from endpoint
    result = ml_client.online_endpoints.invoke(
    endpoint_name='de-mlops-xzqta',
    request_file="./text.json",
    deployment_name="diabetes-model-8",)

    result = ast.literal_eval(result)

    if(result[0]==1):
        return "<h1>Patient has diabetes</h1>"
    return "<h1>Patient do not has diabetes</h1>"

@app.route('/parameter', methods=['POST'])
def parameter():
    lr = float(request.form['LearningRate'])
    credential = AzureCliCredential()


    ml_client = MLClient(
        credential=credential,
        subscription_id= subscription_id,
        resource_group_name= resource_group,
        workspace_name= workspace_name,
    )
    train_src_dir = "./components/train"
    os.makedirs(train_src_dir, exist_ok=True)

    train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))

    # Now we register the component to the workspace
    train_component = ml_client.create_or_update(train_component)

    # Create (register) the component in your workspace
    print(
        f"Component {train_component.name} with Version {train_component.version} is registered"
    )

    # the dsl decorator tells the sdk that we are defining an Azure ML pipeline

    @dsl.pipeline(
        compute="serverless",
        description="E2E data_perp-train pipeline",
    )
    def diabetes_pipeline(
        pipeline_job_data_input,
        pipeline_job_learning_rate,
        pipeline_job_registered_model_name,
    ):

        # using train_func like a python call with its own inputs
        train_job = train_component(
            training_data=pipeline_job_data_input,  # note: using outputs from previous step
            reg_rate=pipeline_job_learning_rate,  # note: using a pipeline input as parameter
            registered_model_name=pipeline_job_registered_model_name,
        )

        # a pipeline returns a dictionary of outputs
        # keys will code for the pipeline output identifier
        return {
            "pipeline_job_model": train_job.outputs.model,
        }
    

    registered_model_name = "diabetes_model"
    uri_path = 'URI Path To CSV File'
    # Let's instantiate the pipeline with the parameters of our choice
    pipeline = diabetes_pipeline(
        pipeline_job_data_input=Input(type="uri_file", path=uri_path),
        pipeline_job_learning_rate= lr,
        pipeline_job_registered_model_name=registered_model_name,
    )

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        # Project's name
        experiment_name="e2e_registered_components",
    )

    ml_client.jobs.stream(pipeline_job.name)
    return "<h1>Job Submitted</h1>"


if __name__ == '__main__':
    app.run(debug=True,port=8080)