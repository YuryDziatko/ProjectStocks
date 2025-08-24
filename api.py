# from fastapi import FastAPI
# from stock_logic import get_prediction_for_random_stock
#
# app = FastAPI()
#
# @app.get("/")
# def root():
#     return {"message": "Welcome to the Stock Predictor API!"}
#
# @app.get("/predict")
# def predict_stock():
#     try:
#         prediction = get_prediction_for_random_stock()
#         return prediction
#     except ValueError:
#         return {"error": "Prediction failed, try another stock."}

import os
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Form

from MLpPrediction import get_or_create_model
from Stocks import get_stock_from_yesterday

# Define absolute path base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STOCKS_PATH = os.path.join(BASE_DIR, "Stocks_name.json")

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATE_DIR)


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    try:
        df_stocks = pd.read_json(STOCKS_PATH)
        tickers = df_stocks["symbol"].tolist()
    except Exception as e:
        tickers = []
        print(f"Error loading stocks list: {e}")
    return templates.TemplateResponse("index.html", {"request": request, "tickers": tickers})

@app.post("/api/predict")
def predict_stock(ticker: str = Form(...)):
    print(f"Received request for ticker: {ticker}")
    # return JSONResponse(content={"message": f"Success for {ticker}"})
    model = get_or_create_model(ticker)
    if not model:
        print(f"Model not found or failed for {ticker}")
        return JSONResponse(
            content={"success": False, "message": f"Model load/train failed for {ticker}."}
        )
    # Get input data
    data_from_yesterday = get_stock_from_yesterday(ticker)
    print(f"data_from_yesterday is {data_from_yesterday.shape}")
    if data_from_yesterday is None or (
            hasattr(data_from_yesterday, "empty") and data_from_yesterday.empty
    ) or (
            hasattr(data_from_yesterday, "size") and data_from_yesterday.size == 0
    ):
        return JSONResponse(
            content={"success": False, "message": f"No recent stock data for {ticker}."}
        )

    print("start prediction")

    prediction = model.predict(data_from_yesterday)
    print("finished prediction")
    result = prediction[-1].argmax()

    output_msg = f"Stock {ticker} will {'increase 📈' if result == 1 else 'not increase 📉'} tomorrow"
    return JSONResponse(content={"success": True, "message": output_msg})

    # except Exception as e:
    # print(f"Prediction error: {e}")
    # return JSONResponse(
    #     content={"success": False, "message": f"Error during prediction: {str(e)}"}
    # )


# @app.post("/api/predict")
# def api_predict(ticker: str = Form(...)):
#     try:
#
#
#         print(f"API called with ticker: {ticker}")
#
#         model = get_or_create_model(ticker)
#         if not model:
#             print(f"Model not found or failed for {ticker}")
#             return JSONResponse(
#                 content={"success": False, "message": f"Model load/train failed for {ticker}."}
#             )
#
#         # Get input data
#         data_from_yesterday = get_stock_from_yesterday(ticker)
#         if data_from_yesterday is None or data_from_yesterday.empty:
#             return JSONResponse(
#                 content={"success": False, "message": f"No recent stock data for {ticker}."}
#             )
#
#         prediction = model.predict(data_from_yesterday)
#         result = prediction[-1].argmax()
#
#         output_msg = f"Stock {ticker} will {'increase 📈' if result == 1 else 'not increase 📉'} tomorrow"
#         return JSONResponse(content={"success": True, "message": output_msg})
#
#
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return JSONResponse(
#             content={"success": False, "message": f"Error during prediction: {str(e)}"}
#         )


# uvicorn api:app --reload