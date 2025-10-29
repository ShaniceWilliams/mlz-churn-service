"""
This file represents the front end of our application that would be used for the marketing team to request predictions for new customers.
    We will use this to test our endpoint.
"""

import requests
# from src.data import customer

url = 'http://localhost:9696/predict'

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
  }

response = requests.post(url, json=customer)

predictions = response.json()

print(predictions)

if predictions['churn']:
    print('customer is likely to churn, send promo')
else:
    print('customer is not likely to churn')