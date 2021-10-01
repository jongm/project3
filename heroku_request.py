"""
Script for trying POST request on Heroku deployed APP
"""

import requests

body = {
    'age': 38,
    'workclass': 'Private',
    'fnlwgt': 215646,
    'education': 'HS-grad',
    'education_num': 9,
    'marital_status': 'Divorced',
    'occupation': 'Handlers-cleaners',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 40,
    'native_country': 'United-States'
}

resp = requests.post("https://project3-inference.herokuapp.com/inference", json=body)

print("Status code:", resp.status_code)
print("Response:", resp.json())
