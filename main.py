"""
API for salary classification
"""

import pickle
import os
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.train_model import cat_features


# Loading model and encoders:
with open("./models/inference_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("./models/onehot_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("./models/label_encoder.pkl", "rb") as file:
    lb = pickle.load(file)

with open("./models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Creating API and Json body:
app = FastAPI()

class PostBody(BaseModel):

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }


# Main route:
@app.get("/")
async def return_greetings():

    return "Welcome to the inference API!"


# Inference route:
@app.post("/inference")
async def model_inference(data: PostBody):

    data = data.dict()

    data, y, encoder_ret, lb_ret, scaler_ret = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
        scaler=scaler
    )

    pred = model.predict(data)
    pred = lb.inverse_transform(pred)

    return {"prediction":pred.tolist()}
