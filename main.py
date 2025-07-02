from fastapi import FastAPI
from stock_logic import get_prediction_for_random_stock

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Predictor API!"}

@app.get("/predict")
def predict_stock():
    try:
        prediction = get_prediction_for_random_stock()
        return prediction
    except ValueError:
        return {"error": "Prediction failed, try another stock."}