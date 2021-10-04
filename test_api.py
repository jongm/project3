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


def test_post_more_than_50k():

    body = {
        'age': 35,
        'workclass': 'Private',
        'fnlwgt': 215646,
        'education': 'Masters',
        'education-num': 12,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 50000,
        'capital-loss': 3000,
        'hours-per-week': 50,
        'native-country': 'United-States'
    }

    r = client.post("/inference", json=body)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], list)
    assert r.json()["prediction"][0] == ">50K"


def test_post_less_than_50k():

    body = {
        'age': 19,
        'workclass': 'Private',
        'fnlwgt': 215646,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family',
        'race': 'Asian-Pac-Islander',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 30,
        'native-country': 'Peru'
    }

    r = client.post("/inference", json=body)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], list)
    assert r.json()["prediction"][0] == "<=50K"


def test_post_wrong():

    body = {
        'wrong_field':10,
        'another_wrong_field':'foo'
    }

    r = client.post("/inference", data=body)
    assert r.status_code != 200