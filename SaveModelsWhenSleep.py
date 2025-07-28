import os
from datetime import datetime, timedelta
import pandas as pd

from MLpPrediction import find_best_model


def build_model_mlp():
    # Load stock symbols
    df_stocks = pd.read_json('Stocks_name.json')
    json_path = "saved_model/model_data.json"

    # Load existing model metadata or initialize new DataFrame
    if os.path.exists(json_path):
        model_data = pd.read_json(json_path, orient="index")
    else:
        model_data = pd.DataFrame(columns=["filename", "date", "f1_score"])

    # Pick a random ticker
    ticker = df_stocks["symbol"].sample().iloc[0]
    today = datetime.today()

    # Check if model needs retraining (older than 30 days or not present)
    if ticker in model_data.index:
        last_trained = pd.to_datetime(model_data.loc[ticker, "date"])
        if today - last_trained < timedelta(days=30):
            print(f"Model for {ticker} is up to date (last trained: {last_trained.date()})")
            return

    print(f"Training new model for {ticker}...")

    # Train model and get score
    f1_score, model = find_best_model(ticker)

    # Save model
    model_filename = f"model_for_{ticker}.keras"
    model.save(os.path.join("saved_model", model_filename))

    # Update metadata
    model_data.loc[ticker] = [model_filename, today.isoformat(), f1_score]
    model_data.to_json(json_path, orient="index", date_format="iso")

    print(f"Model for {ticker} saved. F1 score: {f1_score:.4f}")










