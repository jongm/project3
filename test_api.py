"""
Script for testing the inference API
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() ==  "Welcome to the inference API!"


def test_post_correct():

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

    r = client.post("/inference", json=body)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], list)


def test_post_wrong():

    body = {
        'wrong_field':10,
        'another_wrong_field':'foo'
    }

    r = client.post("/inference", data=body)
    assert r.status_code != 200