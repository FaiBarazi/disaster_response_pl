# Disaster Response Pipeline Project
- Overview
- Repo content
- Setup
- Pipeline:
	- ETL Pipeline
	- ML Pipeline
- Web App
- Conclusion
- Acknowledgment

## Overview
-------------
In this project we build a model for an API that classifies disaster messages. The classification is done over 36 categories based on the content of these messages. The dataset is provided by [Figure Eight](https://appen.com) through Udacity.

## Repo Content
-------------
- app: Flask app with static files to build a web app. 
- data:
	- csv files: Containg the messages and categories matched on ID column.
	- process_data: ETL pipeline that takes the CSVs, loads them and saves them to a SQLite DB. 
	
- model:
	- train_classifier: Training model code that loads the data from the DB trains multiclassification model
		and saves the output in a joblib pickled format.
		

## Setup
-------------
Clone the repo, create a virtual environment and run:
`pip install -r requirements.txt`
This will install all the requirements need to run the app.

## Pipeline
-------------
### ETL
To run ETL pipeline that cleans data and stores in database
`$python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

This will run the ETL process, doing the following:
- Extract the information of the two CSV files of the _messages_ and _categories_
- Clean, combine and transform the data.
- Load the transformed data into one table in an SQLite DB.

### Model
To run ML pipeline that trains classifier and saves the model
`python models/train_classifier.py data/DisasterResponse.db models/classifier.joblib`
The classifier uses a random forest as a multioutput classifier. The parameters are saved in a joblib pickled file. 
Predictions are made using the pickled model after being loaded.

### App
To run the web app cd to the app folder, then run:
`python run.py`
This will run  locally at: http://0.0.0.0:3001/

Type in the message and the results of the predicted category will be highlighted.

## Conclusion
-------------
The dataset is skewed with category 'related' having a much higher occurance and the messages are skewed to disaster related in the original dataset. The model doesn't generalize properly when it comes to categorizing unrelated messages. 
Try typing: 'Help, I really love burgers and fries' and you will see that that would be categorized as 'related'. 
One way to solve this could be adding more unrelated messages to the training dataset and train on that. 


## Acknolwedgement
-------------
Thanks for Figure Eight (appen) and Udacity for providing the data. 
