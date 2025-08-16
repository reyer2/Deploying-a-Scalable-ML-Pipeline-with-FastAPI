import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Load the saved encoder and model only once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(BASE_DIR, "model", "encoder.pkl")
model_path = os.path.join(BASE_DIR, "model", "model.pkl")

encoder, _, _ = load_model(encoder_path)
model, _, _ = load_model(model_path)

# Create FastAPI instance
app = FastAPI()

# GET request on the root
@app.get("/")
async def get_root():
    """ Say hello! """
    return {"message": "Hello from the API!"}

# POST request for model inference
@app.post("/data/")
async def post_inference(data: Data):
    # Convert Pydantic model to dict and adjust column names
    data_dict = data.dict()
    data_df = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data_df = pd.DataFrame.from_dict(data_df)

    # Process data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder
    )

    # Predict
    preds = inference(model, data_processed)
    return {"result": apply_label(preds)}
