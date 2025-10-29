#============================#
#---------- Imports ---------#
#============================#

import pickle
from src.logger import logging

#============================#
#-------- Parameters --------#
#============================#

input_file_path = './models/model_C=1.0_27_Oct_2025_10_18.bin'

logging.info(f"Loading model {input_file_path}")


#============================#
#------ Load the model ------#
#============================#

with open(input_file_path, 'rb') as f_in:
    pipeline = pickle.load(f_in)


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

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

y_pred = predict_single(customer)

print(f'Churn Probability: {y_pred:.3f}')

if y_pred >= 0.5:
    print('Customer predicted to churn. Send email with promo')
else:
    print('Customer is not predicted to churn, no action required.')

logging.info(f"Churn prediction probability: {y_pred}")