"""
API for salary classification
"""

import pickle

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


# Main route:
@app.get("/")
async def return_greetings():

    return "Welcome to the API"


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

    return pred.tolist()
