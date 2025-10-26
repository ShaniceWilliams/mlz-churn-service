#============================#
#---------- Imports ---------#
#============================#

import pickle
from src.logger import logging

#============================#
#-------- Parameters --------#
#============================#

input_file_path = './models/model_C=1.0_25_Oct_2025_22_32.bin'

logging.info(f"Loading model {input_file_path}")


#============================#
#------ Load the model ------#
#============================#

with open(input_file_path, 'rb') as f:
    dv, model = pickle.load(f)


# dv, model

#============================#
#---- Make New Prediction ---#
#============================#

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


logging.info(f"Customer input: {customer}")

x = dv.transform([customer])

y_pred = model.predict_proba(x)[0, 1]

logging.info(f"Churn prediction probability: {y_pred}")