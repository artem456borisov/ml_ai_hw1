from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.encoders import jsonable_encoder
import pandas as pd
import joblib
import pickle

app = FastAPI()

# reg = joblib.load('model_final.joblib')

with open('model.pkl', 'rb') as f:
    reg = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: float
    max_torque_rpm: float


class Items(BaseModel):
    objects: List[Item]

class ItemResponse(Item):
    prediction: float


def pydantic_model_to_df(model_instance):
    if type(model_instance) == list:
        frame = pd.DataFrame(jsonable_encoder(model_instance))
    else:
        frame = pd.DataFrame([jsonable_encoder(model_instance)])
    frame = frame.loc[:, ["year", "km_driven", "mileage", "engine", "max_power", "max_torque_rpm"]]
    return frame

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_instance = pydantic_model_to_df(item)
    result = reg.predict(df_instance)
    return result


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df_instance = pydantic_model_to_df(items)
    result = reg.predict(df_instance)
    return result