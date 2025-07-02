from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse


import pandas as pd
from ModelPrediction import get_train_test_data, evaluate_classification_models

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    df_stocks = pd.read_json('Stocks_name.json')
    tickers = df_stocks["symbol"].tolist()
    return templates.TemplateResponse("index.html", {"request": request, "tickers": tickers})


@app.post("/api/predict")
def api_predict(ticker: str = Form(...)):
    try:
        X_train, X_test, Y_train, Y_test = get_train_test_data(ticker)
        result = evaluate_classification_models(X_train, X_test, Y_train, Y_test, ticker)

        if result:
            output_t=f"Stock {ticker} will increase tomorrow"
        else:
            output_t = f"Stock {ticker} will not increase tomorrow"

        return JSONResponse(content={"success": True, "message": output_t})

    except Exception as e:
        return JSONResponse(content={"success": False, "message": "Try another stock"})