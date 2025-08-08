# import pandas as pd
# from ModelPrediction import get_train_test_data, evaluate_classification_models
#
# def get_prediction_for_random_stock():
#     df_stocks = pd.read_json('Stocks_name.json')
#     ticker = df_stocks["symbol"].sample().iloc[0]
#
#     X_train, X_test, Y_train, Y_test = get_train_test_data(ticker)
#     results = evaluate_classification_models(X_train, X_test, Y_train, Y_test, ticker)
#
#     return {"ticker": ticker, "prediction_result": results}