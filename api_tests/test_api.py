# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:31:20 2021

@author: jongm
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
    'native_country': 'United-States',
}


resp = requests.post("http://127.0.0.1:8000/inference", json=body)

resp.json()

