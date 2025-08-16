import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model
import joblib

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(BASE_DIR, "model", "encoder.pkl")
model_path = os.path.join(BASE_DIR, "model", "model.pkl")

# Debug: confirm files exist
if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print("Encoder path:", encoder_path)
print("Model path:", model_path)

# Load models using absolute paths
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

# Create FastAPI instance
app = FastAPI()

# GET request on the root
@app.get("/")
async def get_root():
    return {"message": "Hello from the API!"}

# POST request for model inference
@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.dict()
    data_df = pd.DataFrame([{k.replace("_", "-"): v for k, v in data_dict.items()}])

    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    data_processed, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder
    )

    preds = inference(model, data_processed)
    return {"result": apply_label(preds)}

