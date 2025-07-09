
# import yfinance as yf
# from datetime import datetime, timedelta
# import pandas as pd
# # === Settings ===
# #print(str(df_stocks["symbol"].sample()))
# df_stocks = pd.read_json('Stocks_name.json')
# ticker = df_stocks["symbol"].sample().iloc[0]  # Change this to any stock symbol you want
# end_date = datetime.today()
# start_date = end_date - timedelta(days=365)
#
# #
#
# # === Download Data ===
# data_temp = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
# print("data_temp    " ,data_temp.sample())
#
#
# # Shift the Open column BACK by -1 to get the next day's Open for each row
# Increase_temp = pd.DataFrame()
# Increase_temp['Next_Open'] = data_temp['Open'].shift(-1)
#
# # Compare: is next day's Open > today's Close ?
# Increase_temp['Next_Open_gt_Close'] = Increase_temp['Next_Open'] > data_temp['Close']
# print("Increase_temp    /n" , Increase_temp.sample())
#
# # === Save to Excel and JSON ===
# # excel_filename = f"{ticker}_last_year.xlsx"
# # json_filename = f"{ticker}_last_year.json"
# #
# # data.to_excel(excel_filename)
# # data.to_json(json_filename, orient="records", date_format="iso")
# #
# # print(f"Data for {ticker} saved as:\n - Excel: {excel_filename}\n - JSON: {json_filename}")


import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def download_and_save_indices(json_path="index_data.json"):
    tickers = ['^IXIC', '^GSPC', '^DJI']
    today = datetime.today()
    start_date = today - timedelta(days=365)

    data_frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=today + timedelta(days=1))[['Close']]
        df.columns = [ticker]
        data_frames.append(df)

    merged_df = pd.concat(data_frames, axis=1)
    merged_df.index.name = "Date"

    merged_df.to_json(json_path, orient="index", date_format="iso")
    print(f"Index data saved to {json_path}")

    # Return merged_df **with Date index** to allow proper alignment later
    return merged_df

# Usage




def get_data_stock(ticker, MLP=False):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    try:
        data_temp = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if MLP:
            indices_df = download_and_save_indices()
            # Combine on Date index, keep all rows from data_temp
            data_temp_for_MLP = pd.concat([data_temp, indices_df], axis=1)
            return data_temp_for_MLP
        else:
            return data_temp
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def get_stock_from_yesterday(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=7)
    data_temp = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    imputer = SimpleImputer(strategy='mean')
    data_temp_imputer = pd.DataFrame(imputer.fit_transform(data_temp), columns=data_temp.columns)
    scaler = StandardScaler()
    return scaler.fit_transform(data_temp_imputer)


def create_data_output(data_stock):
    # Create new DataFrame with comparison
    increase_temp = pd.DataFrame()
    increase_temp['Close'] = data_stock['Close']
    increase_temp['Next_Open'] = data_stock['Open'].shift(-1)
    increase_temp['Next_Open_gt_Close'] = increase_temp['Next_Open'] > increase_temp['Close']

    # print("Sample from Increase_temp:\n", increase_temp.sample())
    return increase_temp["Next_Open_gt_Close"]

# data_yury=get_data_stock(ticker)
# create_data_output(data_yury)