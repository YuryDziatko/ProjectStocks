

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from when_started import download_and_save_indexes_to_json



def get_data_stock(ticker, MLP=False, days_stock=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days_stock)
    # print(ticker)

    try:
        data_temp = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        # print("data_temp ",data_temp.shape)
        # print("data_temp ",data_temp.sample())

        if MLP:
            indices_df = download_and_save_indexes_to_json(json_path="index_data.json",days_index=days_stock)

            data_temp_for_MLP = data_temp.merge(indices_df, left_index=True, right_index=True)

            return data_temp_for_MLP
        else:
            return data_temp
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def get_stock_from_yesterday(ticker):
    # Load dataset
    df_clf = get_data_stock(ticker, MLP=True, days_stock=7)
    df_clf = df_clf.iloc[1:]


    # Impute numerical features if needed
    imputer = SimpleImputer(strategy='mean')
    X_clf = pd.DataFrame(imputer.fit_transform(df_clf), columns=df_clf.columns)


    # Standardize the features
    scaler = StandardScaler()
    X_clf_scaled = scaler.fit_transform(X_clf)


    return X_clf_scaled

def create_data_output(data_stock):
    # Create new DataFrame with comparison
    increase_temp = pd.DataFrame()
    increase_temp['Close'] = data_stock['Close']
    increase_temp['Next_Open'] = data_stock['Open'].shift(+1)
    increase_temp['Next_Open_gt_Close'] = increase_temp['Next_Open'] > increase_temp['Close']


    return increase_temp["Next_Open_gt_Close"]
