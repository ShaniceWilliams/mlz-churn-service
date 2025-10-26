#============================#
#---------- Imports ---------#
#============================#

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import pickle
from datetime import datetime

from src.logger import logging

#============================#
#-------- Parameters --------#
#============================#

C = 1.0
n_splits = 5
logging.info(f"Start training. Model Params: C={C}, n_splits={n_splits}")


#============================#
#----- Data Preparation -----#
#============================#

logging.info("Reading CSV")
df = pd.read_csv("./data/raw/Customer-Churn-Data.csv")
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
logging.info(f"df_full_train shape: {df_full_train.shape}, df_test shape: {df_test.shape}")


#============================#
#------ Model Training ------#
#============================#

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

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


#============================#
#----- Model Validation -----#
#============================#

logging.info(f"Start Model validation. Model Params: C={C}, n_splits={n_splits}")

scores = []

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):

    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

logging.info(f'C={C} mean +- std: {np.mean(scores):.3f} +- {np.std(scores):.3f}') 


#============================#
#--- Final Model Training ---#
#============================#

logging.info(f"Start Final model training. Model Params: C={C}")

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
logging.info(f" Final Model AUC ROC Score: {auc}") 


#============================#
#------- Model Saving -------#
#============================#

logging.info(f"Saving Model...")
timestamp = datetime.now().strftime('%d_%b_%Y_%H_%M')

output_file_name = f'./models/model_C={C}_{timestamp}.bin'

with open(output_file_name, 'wb') as f:
    pickle.dump((dv, model), f)

logging.info(f"Model saved to: ./../models/{output_file_name}")