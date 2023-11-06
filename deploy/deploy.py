from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd

class History(BaseModel):
    t_1: int
    t_2: int
    t_3: int
    t_4: int
    t_5: int
    t_6: int
    t_7: int 

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(history: History):
    # Load model
    model = pickle.load(open("linear_regressor.pickle", "rb"))

    history_df = pd.DataFrame({
        "t-1": [history.t_1],
        "t-2": [history.t_2],
        "t-3": [history.t_3],
        "t-4": [history.t_4],
        "t-5": [history.t_5],
        "t-6": [history.t_6],
        "t-7": [history.t_7],
    })

    # Predict
    prediction = model.predict(history_df)

    # Return prediction
    return {"prediction": round(prediction[0])}