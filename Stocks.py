
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
            data_temp.columns = data_temp.columns.get_level_values(0)
            data_temp_for_MLP = data_temp.merge(indices_df, left_index=True, right_index=True)



            return data_temp_for_MLP
        else:
            return data_temp
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# def download_indices(num_days: int = 300) -> pd.DataFrame:
#     """
#     Return a dataframe containing the last `num_days` rows of the
#     NASDAQ (^IXIC), S&P‑500 (^GSPC) and Dow Jones (^DJI) closing values.
#     Columns are renamed to their ticker symbols.
#     """
#     end = datetime.today()
#     # fetch twice the requested span to ensure we keep delisted / holiday gaps
#     start = end - timedelta(days=num_days * 2)
#
#     tickers = ['^IXIC', '^GSPC', '^DJI']
#     dfs = []
#
#     for t in tickers:
#         df = yf.download(t, start=start, end=end + timedelta(days=1))[['Close']]
#         df.columns = [t]                       #  ->  ^IXIC, ^GSPC, ^DJI
#         dfs.append(df)
#
#     idx_df = pd.concat(dfs, axis=1).tail(num_days)
#     idx_df.index.name = "Date"
#     return idx_df


# ---------------------------------------------------------------------
# 2.  Stock‑data + label generator
# # ---------------------------------------------------------------------
# def get_stock_data(
#     ticker: str,
#     num_days: int = 300,
#     merge_indices: bool = True
# ):
#     end = datetime.today()
#     start = end - timedelta(days=num_days * 2)
#
#     # 1 ──────────────────────────────────────────────────────────
#     raw = yf.download(ticker, start=start, end=end + timedelta(days=1))
#
#     # keep only what we need and rename once
#     raw = (raw[['Open', 'Close']]
#            .rename(columns={'Open': 'open_t', 'Close': 'close_t'})
#            .tail(num_days)
#            .copy())
#
#     raw = raw.loc[:, ~raw.columns.duplicated(keep='first')]  # ⟵ NEW
#
#     raw['label'] = (raw['open_t'].shift(-1) > raw['close_t']).astype(int)
#     raw = raw.iloc[:-1]  # drop last row (no next_open)
#
#     # 3 ──────────────────────────────────────────────────────────
#     if merge_indices:
#         idx = download_indices(num_days)           # ^IXIC, ^GSPC, ^DJI
#         idx = idx.reindex(raw.index)               # perfectly align dates
#         raw = pd.concat([raw, idx], axis=1)
#
#     # 4 ──────────────────────────────────────────────────────────
#     y = raw.pop('label')
#     raw.drop(columns='next_open', inplace=True)
#
#     # example preprocessing
#     imputed  = SimpleImputer(strategy='mean').fit_transform(raw)
#     X_scaled = StandardScaler().fit_transform(imputed)
#     X        = pd.DataFrame(X_scaled, index=raw.index, columns=raw.columns)
#
#     return X, y


# def get_stock_from_yesterday(ticker):
#     end_date = datetime.today()
#     start_date = end_date - timedelta(days=7)
#     data_temp = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
#     imputer = SimpleImputer(strategy='mean')
#     data_temp_imputer = pd.DataFrame(imputer.fit_transform(data_temp), columns=data_temp.columns)
#     scaler = StandardScaler()
#     return scaler.fit_transform(data_temp_imputer)

def get_stock_from_yesterday(ticker):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=7)

        # Download stock data
        stock_df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if stock_df.empty:
            raise ValueError(f"No stock data returned for {ticker}")

        # Download index data for the same date range
        index_tickers = ['^IXIC', '^GSPC', '^DJI']
        index_frames = []
        for index_ticker in index_tickers:
            idx_df = yf.download(index_ticker, start=start_date, end=end_date + timedelta(days=1))[['Close']]
            idx_df.columns = [index_ticker]
            index_frames.append(idx_df)

        # Combine indexes into one DataFrame
        indices_df = pd.concat(index_frames, axis=1)

        # Merge stock and indices on date
        stock_df.columns = stock_df.columns.get_level_values(0)  # flatten in case of multi-index
        merged_df = stock_df.merge(indices_df, left_index=True, right_index=True)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        imputed_df = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns)

        # Standardize
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(imputed_df)

        return scaled_array

    except Exception as e:
        print(f"Error in get_stock_from_yesterday({ticker}): {e}")
        return None


def create_data_output(data_stock):
    # Create new DataFrame with comparison
    increase_temp = pd.DataFrame()
    increase_temp['Close'] = data_stock['Close']
    increase_temp['Next_Open'] = data_stock['Open'].shift(+1)
    increase_temp['Next_Open_gt_Close'] = increase_temp['Next_Open'] > increase_temp['Close']

    # print("Sample from Increase_temp:\n", increase_temp.sample())
    return increase_temp["Next_Open_gt_Close"]

# data_yury=get_data_stock(ticker)
# create_data_output(data_yury)