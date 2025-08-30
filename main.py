import os
import pandas as pd
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta

from MLpPrediction import get_or_create_model
from Stocks import get_stock_from_yesterday
from SaveModelsWhenSleep import build_model_mlp

# Define absolute path base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STOCKS_PATH = os.path.join(BASE_DIR, "Stocks_name.json")

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Track last activity
last_activity = datetime.utcnow()

# Background task for idle retraining
async def idle_checker():
    global last_activity
    while True:
        await asyncio.sleep(60)  # check every 60 seconds
        if datetime.utcnow() - last_activity > timedelta(minutes=5):
            print("Idle for 5 minutes, starting model retraining...")
            try:
                build_model_mlp()
            except Exception as e:
                print(f"Error during idle retraining: {e}")
            # reset timer so it won’t run again immediately
            last_activity = datetime.utcnow()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(idle_checker())

@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    global last_activity
    last_activity = datetime.utcnow()  # update on each request
    try:
        df_stocks = pd.read_json(STOCKS_PATH)
        tickers = df_stocks["symbol"].tolist()
    except Exception as e:
        tickers = []
        print(f"Error loading stocks list: {e}")
    return templates.TemplateResponse("index.html", {"request": request, "tickers": tickers})


@app.post("/api/predict")
def api_predict(ticker: str = Form(...)):
    global last_activity
    last_activity = datetime.utcnow()  # update on each API call
    try:
        print(f"API called with ticker: {ticker}")

        model = get_or_create_model(ticker)
        if not model:
            print(f"Model not found or failed for {ticker}")
            return JSONResponse(
                content={"success": False, "message": f"Model load/train failed for {ticker}."}
            )

        # Get input data
        data_from_yesterday = get_stock_from_yesterday(ticker)
        if data_from_yesterday is None or (
                hasattr(data_from_yesterday, "empty") and data_from_yesterday.empty
        ) or (
                hasattr(data_from_yesterday, "size") and data_from_yesterday.size == 0
        ):
            return JSONResponse(
                content={"success": False, "message": f"No recent stock data for {ticker}."}
            )

        prediction = model.predict(data_from_yesterday)
        result = prediction[-1].argmax()

        output_msg = f"Stock {ticker} will {'increase 📈' if result == 1 else 'not increase 📉'} tomorrow"
        return JSONResponse(content={"success": True, "message": output_msg})

    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(
            content={"success": False, "message": f"Error during prediction: {str(e)}"}
        )




# import os
#
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import JSONResponse
#
#
# import pandas as pd
#
# import MLpPrediction
# import Stocks
# from MLpPrediction import get_or_create_model
# from ModelPrediction import get_train_test_data, evaluate_classification_models
#
# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
#
#
# @app.get("/", response_class=HTMLResponse)
# def show_form(request: Request):
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     STOCKS_PATH = os.path.join(BASE_DIR, 'Stocks_name.json')
#     df_stocks = pd.read_json(STOCKS_PATH)
#     tickers = df_stocks["symbol"].tolist()
#     return templates.TemplateResponse("index.html", {"request": request, "tickers": tickers})
#
#
#
#
# @app.post("/api/predict")
# def api_predict(ticker: str = Form(...)):
#     try:
#
#         print("start api_predict")
#         model = MLpPrediction.get_or_create_model(ticker)
#         print("get model for ", ticker)
#         if not model:
#             return JSONResponse(content={"success": False, "message": f"Failed to load or create model for {ticker}  {model}."})
#
#         # Prepare yesterday’s data
#         data_from_yesterday = Stocks.get_stock_from_yesterday(ticker)
#         prediction = model.predict(data_from_yesterday)
#         result = prediction[-1].argmax()
#
#         output_t = f"Stock {ticker} will {'increase' if result == 1 else 'not increase'} tomorrow"
#
#         return JSONResponse(content={"success": True, "message": output_t})
#     except Exception as e:
#         return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})
#
# # @app.post("/api/predict")
# # def api_predict(ticker: str = Form(...)):
# #     try:
# #         X_train, X_test, Y_train, Y_test = get_train_test_data(ticker)
# #         result = evaluate_classification_models(X_train, X_test, Y_train, Y_test, ticker)
# #
# #         if result:
# #             output_t=f"Stock {ticker} will increase tomorrow"
# #         else:
# #             output_t = f"Stock {ticker} will not increase tomorrow"
# #
# #         return JSONResponse(content={"success": True, "message": output_t})
# #
# #     except Exception as e:
# #         return JSONResponse(content={"success": False, "message": "Try another stock"})
#
# # @app.post("/api/predict")
# # def api_predict(ticker: str = Form(...)):
# #     try:
# #
# #         print("start api_predict")
# #         model = get_or_create_model(ticker)
# #         print("get model for ", ticker)
# #         if not model:
# #             return JSONResponse(content={"success": False, "message": f"Failed to load or create model for {ticker}."})
# #
# #         # Prepare yesterday’s data
# #         data_from_yesterday = Stocks.get_stock_from_yesterday(ticker)
# #         prediction = model.predict(data_from_yesterday)
# #         result = prediction[-1].argmax()
# #
# #         output_t = f"Stock {ticker} will {'increase' if result == 1 else 'not increase'} tomorrow"
# #
# #         return JSONResponse(content={"success": True, "message": output_t})
# #     except Exception as e:
# #         return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})

# uvicorn main:app --reload