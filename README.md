# Data Scientist Nanodegree

## Data Engineering

## Project: Disaster Response Pipeline

## Table of Contents

- [Project Overview](#project-overview)
- [Project Components](#project-components)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Prerequisite](#prereq)
- [Running](#project-running)
  - [ETL pipeline(Data Cleaning and Storing stage)](#cleaning)
  - [ML Pipeline(Training and Classifying stage)](#train-classify)
  - [Running the Flask Web App](#run-flask)
- [Conclusion](#conclusion)
- [Files](#files)

***

<a id='project-overview'></a>

## 1. Project Overview
In this project using data engineering, I will analyze disaster data from <a href="https://appen.com/datasets/combined-disaster-response-data/" target="_blank">Figure Eight</a> to classify into relevant disaster messages.


<a id='project-components'></a>

## 2. Project Components

There are three components of this project:

<a id='etl_pipeline'></a>

### 2.1. ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages` and `categories` dataset and merge them into a single dataframe
- Cleans the merged data
- Stores it in a **SQLite database**

Running [this command](#clean-cmd) **in the parent directory** will start the process of ETL(Extract, Transform, Load).

<a id='ml_pipeline'></a>

### 2.2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:

- Using pandas' _read_sql_table_, load the data from the **SQLite database**
- Splits this loaded data into training and testing sets using _train_test_split_ from _sklearn.model_selection_
- Builds a text processing which includes normalizing, tokenizing and lemmatizing and, machine learning pipeline. 
- Trains on train dataset and tunes a model using _GridSearchCV_
- Outputs result on the test dataset
- Exports the final model-pipeline into a _pickle_ file

<a id='flask'></a>

### 2.3. Flask Web App

Running [this command](#web-cmd) **inside the app directory** will start the web app where users can enter their query, i.e., a request message sent during a natural disaster, e.g. _"Please, we need tents and water. We are in Silo, Thank you!"_.

<a id="prereq"></a>
## 3. Prerequisite:
To run the code, you need to have the following:

- Python (More Details: https://www.python.org/downloads/)
- Jupyter (More Details: https://test-jupyter.readthedocs.io/en/latest/install.html)

<a id='project-running'></a>

## 4. Running

The application requires to be started from ETL pipeline stage

<a id='cleaning'></a>

### 4.1. Stage 1: ETL pipeline(Data Cleaning and Storing stage)

**Go to the project directory** and the run the following command:

<a id='clean-cmd'></a>

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

![Stage1](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Stage1.JPG?raw=true)

<a id='train-classify'></a>
### 4.2. Stage 2: ML Pipeline(Training and Classifying stage)

After Stage 1 is complete

**Go to the project directory** and the run the following command:

<a id='train-cmd'></a>

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
![Stage2-Initialize](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Stage2_init.JPG?raw=true)


<a id='stage2-results'></a>
![Stage2-Results1](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Stage2_results1.JPG?raw=true)

![Stage2-Results2](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Stage2_results2.JPG?raw=true)


<a id='run-flask'></a>
### 4.3. Stage 3: Running the Flask Web App

After Stage 2 is complete (the model is saved as pickle file)

**Go the app directory** and run the following command:

<a id='web-cmd'></a>

```bat
python run.py
```

![Stage3](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Stage3.jpg?raw=true)


This will start the web application in your browser.

**_Screenshot 1_**

![Landing Page](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Landing_page.JPG?raw=true)

**_Screenshot 2_**

![Overview Figure1](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Overview_Figure1.JPG?raw=true)

**_Screenshot 3_**

![Overview Figure2](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Overview_Figure2.JPG?raw=true)

**_Screenshot 4_**

![Alert Results](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Alert_Results.JPG?raw=true)

The relevant results have been highlighted in _green_ and other categories are in _disabled_ state.
On selecting any of the relevant results, we get the following popup.

**_Screenshot 5_**

![Emergency Alert Message](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Emergency_Alert_Message.JPG?raw=true)

**_Screenshot 6_**

![Emergency_Popup_Location](https://github.com/prabhatdutt95/Disaster-Response-Pipeline/blob/main/Screenshots/Emergency_Popup_Location.JPG?raw=true)

Note: "Location Found" doesnot actually takes your current location. This is only for demonstration purpose.


## 5. Conclusion

From [Stage-2 Results](#stage2-results), We can see that:
- Accuracy is ~0.95 (High)
- Recall is 0.64 (Moderate)
- F1-score is 0.67 (Moderate)

So, take appropriate measures when using this model for decision-making process at a larger scale or in a production environment.


## 6. Files

<pre>
.
├── app
│   ├── templates
│   │   ├── assets
│   │   │   ├── favicon.ico-----------# FAVICON FOR THE WEB APP
│   │   │   └── logo.png--------------# LOGO FOR THE WEB APP
│   │   ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│   │   └── master.html---------------# MAIN PAGE OF WEB APP
│   └── run.py------------------------# FLASK FILE THAT RUNS APP
│
├── data
│   ├── disaster_categories.csv-----------# CSV DATA CONTAINING CATEGORIES DATA
│   ├── disaster_messages.csv-------------# CSV DATA CONTAINING MESSAGES DATA
│   ├── DisasterResponse.db---------------# DATABASE IN WHICH WE SAVE THE CLEANED DATA
│   └── process_data.py-------------------# PERFORMS ETL PROCESS
│   └── ETL Pipeline Preparation.ipynb----# JUPYTER NOTEBOOK CONTAINING ETL PIPELINE
│
├── models
│   ├── classifier.pkl -------------------# PICKLE FILE TO SAVE THE EXPORT THE FINAL MODEL-PIPELINE
│   ├── ML Pipeline Preparation.ipynb-----# JUPYTER NOTEBOOK CONTAINING ML PIPELINE
│   └── train_classifier.py---------------# PERFORMS CLASSIFICATION TASK
|
└── Screenshots ----------------------# CONTAINS SCREENSHOTS FOR VARIOUS STEPS IN APPLICATION

</pre>
