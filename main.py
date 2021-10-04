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
from pydantic import BaseModel, Field

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
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                'age': 38,
                'workclass': 'Private',
                'fnlwgt': 215646,
                'education': 'HS-grad',
                'education-num': 9,
                'marital-status': 'Divorced',
                'occupation': 'Handlers-cleaners',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
                }
        }


# Main route:
@app.get("/")
async def return_greetings():

    return "Welcome to the inference API!"


# Inference route:
@app.post("/inference")
async def model_inference(data: PostBody):

    data = data.dict(by_alias=True)

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
