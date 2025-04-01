from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

model_path = "./laptop_price_model.pkl"
model = joblib.load(model_path)

@app.get("/")
def index():
    return {"text": "It is works"}

@app.post("/predict/")
async def predict(file: UploadFile):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
