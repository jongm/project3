"""
Code for testing model training
"""

import pandas as pd
import numpy as np
import pytest

from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def fake_data():

    """
    Fake data for model training
    """
    fake_data = pd.DataFrame({
        "var1":[1,2,-3,-1,2,3],
        "var2":[0,0,0,1,1,1],
        "var3":[2.7,1.5,-0.8,0.2,-2,0.3],
        "label":[1,1,1,1,0,0]
    })

    return fake_data


def test_train_model(fake_data):
    """
    Tests if a model can be correctly trained
    """

    X_fake = fake_data.copy()
    y_fake = X_fake.pop("label")

    model = train_model(X_fake, y_fake)

    assert model.n_classes_ == 2


def test_compute_model_metrics():
    """
    Compute metrics with fake arrays
    """
    fake_y = np.array([1,1,1,0,0,1,1,1])
    fake_preds = np.array([1,0,1,0,1,0,1,0])

    prec, rec, fb, FPR =  compute_model_metrics(fake_y, fake_preds)

    assert all([prec == 0.75, rec == 0.5, fb == 0.6, FPR == 0.5])


def test_inference():

    X_fake = fake_data.copy()
    y_fake = X_fake.pop("label")
    model = train_model(X_fake, y_fake)

    assert all(inference(model, X_fake) == [1, 1, 1, 1, 0, 0])



