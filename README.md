# Documentation: Deploying a Diabetic Classification Model with Azure ML


## Table of Contents

1. Introduction
2. Use Case
3. Resource Group and Workspace Creation
4. Compute Cluster Setup
5. Environment Configuration
6. Training the Diabetic Classification Model
   - 6.1 Training Data
   - 6.2 Training Script (`train.py`)
   - 6.3 Training Component Registration
7. Creating Azure ML Pipeline
8. Building the Web Application (`app.py`)
   - 8.1 Front-End for End Clients (`index.html`)
   - 8.2 Parameter Input for Data Engineers (`index2.html`)
9. Running the Web Application
10. Deployment and Serving Models
11. Conclusion

## 1. Introduction

This documentation outlines the steps to deploy a Diabetic Classification Model using Azure ML. This includes setting up the environment, creating and managing Azure ML pipelines, and version control for the machine learning model. The web application aspect will be mentioned as part of the project, but the primary emphasis will be on the MLOps tasks and Azure ML pipeline creation.




## 2. Use Case

The medical domain company wants to develop a web application that helps healthcare professionals quickly classify patients as diabetic or non-diabetic based on their medical information. The Data Science team within the medical domain company has successfully developed a sophisticated Diabetic Classification Model. This model serves as a vital tool in assisting healthcare professionals and clinicians in the accurate identification of patients with diabetes. The primary objective of this model is to enable swift and reliable classification of patients into two categories: those with diabetes (diabetic) and those without diabetes (non-diabetic). Now as an MLOP’s engineer you are required to productionize the model and build a web application that classifies a patient as diabetic or not based on his information. 

1. Create an environment where you can train, retrain, and save the best model.
2. Maintain a user interface (UI) for parameter input.
3. Manage a model registry with all versions of the model that are saved.
4. Maintain tags for each model, such as overfit, low accuracy, and best accuracy.
5. Serve the best model as an endpoint through a UI to the customer.

The following sections will provide a detailed overview of each component and the associated code used in this project.

## 3. Resource Group and Workspace Creation

1. Create a resource group with the desired location (e.g., "eastus").
2. Create a Machine Learning Workspace within the resource group by specify the location and other relevant details.
3. We have used the Azure Machine Learning Python SDKv2 for creating the resource group and workspace.

## 4. Compute Cluster Setup

Set up a compute cluster to facilitate model training and other computational tasks. The cluster, in this case, is named cpu-cluster-00 and is configured to meet the requirements. The cluster comprises maximum of 2 nodes. 


## 5. Environment Configuration

Configure a custom environment for the machine learning operations. This environment, named mlflow-env, includes the necessary dependencies and specifications for training the diabetic classification model. It ensures consistency in the machine learning workflow.

## 6. Training the Diabetic Classification Model
### 6.1 Training Data:

- The training data used for this project is stored in a CSV file, which includes features related to diabetes diagnosis and a binary label for diabetic or non-diabetic. It is stored as data asset in azure cloud.

### 6.2 Training Script (`train.py`):
- A Python script, `train.py`, is used to train the Diabetic Classification Model. This script employs the Scikit-Learn library and a Logistic Regression model.

- The script performs the following tasks:
    - Reads the training data from the provided file path.
    - Splits the data into training and testing sets.
    - Trains the model using logistic regression.
    - Logs the model metrics and saves the model using Azure MLflow.

- We can pass the training data, learning rate, and the name of the registered model as command-line arguments.

### 6.3 Training Component Registration:
- The trained model is registered as a component in the Azure Machine Learning workspace. This component can be reused and incorporated into Azure ML pipelines.

# 7. Creating Azure ML Pipeline

- An Azure ML pipeline is created to automate the end-to-end process of data preparation and model training.
- The pipeline consists of two components: data preparation and model training. The model training component uses the previously registered training component.
- The pipeline takes input data, a learning rate, and the registered model name as argument.
- The entire pipeline is defined using the Azure Machine Learning Python SDKv2.


## 8. Building the Web Application (`app.py`)

### 8.1 Front-End for End Clients (`index.html`):
- The application includes a landing page (`index.html`) with a user interface for collecting patient information.
- End clients can input patient data, including parameters like pregnancies, glucose levels, blood pressure, and more.
- Upon form submission, the data is collected and packaged as a JSON request.
- The application communicates with the Azure Machine Learning service to make predictions using the registered model.
- The predicted result, whether the patient is diabetic or not, is displayed to the end clients.

### 8.2 Parameter Input for Data Scientists (`index2.html`):
- The application provides a separate page (`index2.html`) to input the learning rate for model training.
- Data scientists can input the learning rate and initiate the model retraining process.
- The training pipeline is triggered in response to the input, creating a new version of the model.

## 9. Running the Web Application

- The Flask application is run using the `app.py` script.
- The application can be accessed locally by navigating to `http://localhost:8080` in a web browser.

## 10. Deployment and Serving Models

- All model versions are stored in the Azure Machine Learning service.
- Data scientists and relevant authorities assess the models and manually switch the endpoint to refer to the desired model version using the Azure Machine Learning Studio UI.

## 11. Conclusion

This documentation provides a comprehensive overview of the steps involved in building a medical domain diabetic classification web application. It outlines the use of Azure Machine Learning, to create a seamless workflow from model training to serving predictions in a user-friendly interface. The resulting application allows healthcare professionals to classify patients as diabetic or non-diabetic based on their medical information, aiding in the accurate identification of patients with diabetes. End clients can use the application for predictions, and data scientists can retrain and create new model versions as needed.
