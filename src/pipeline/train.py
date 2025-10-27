#============================#
#---------- Imports ---------#
#============================#

import pandas as pd

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


import pickle
from datetime import datetime
from src.logger import logging


#============================#
#----- Data Preparation -----#
#============================#

def load_data(path: str) -> pd.DataFrame: 
    """Function to load data as dataframe and preprocess for model training.
    Params:
        path (str): path to file. Could be string or url
    Returns:
        df: pd.DataFrame
    """
    logging.info("Loading data...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    
    logging.info(f"df shape: {df.shape}")

    return df




#============================#
#------ Model Training ------#
#============================#

def train_model(df:pd.DataFrame) -> sklearn.pipeline.Pipeline:
    """Function to load data as dataframe and preprocess for model training.
    Params:
        df (pd.Dataframe): preprocessed dataframe to be used to train the model
    Returns:
        pipeline: sklearn.pipeline.Pipeline
    """
    logging.info("Beginning training model on data...")

    numerical =  ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ] 

    y_train = df.churn
    train_dict = df[categorical + numerical].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )

    pipeline.fit(train_dict, y_train)

    logging.info(f"Training complete.")

    return pipeline


#============================#
#------- Model Saving -------#
#============================#

def save_model(pipeline):
    logging.info(f"Saving Model...")
    timestamp = datetime.now().strftime('%d_%b_%Y_%H_%M')

    output_file_name = f'./models/model_{timestamp}.bin'

    with open(output_file_name, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

    logging.info(f"Model saved to: ./../models/{output_file_name}")

#============================#
#---------- Main() ----------#
#============================#

def main():
    logging.info("Loading data...")
    data_path = "./data/raw/Customer-Churn-Data.csv"
    df = load_data(data_path)
    pipeline = train_model(df)
    save_model(pipeline)
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()