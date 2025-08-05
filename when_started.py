
# # df = investpy.get_stocks(country='united states')
# # df_stocks = df[['symbol', 'name']]
# #
# # df = investpy.get_stocks(country='united states')
# df_stocks = investpy.get_stocks(country='united states')[['symbol', 'name']]
# #json_filename = "Stocks_name.json"
# df_stocks.to_json("Stocks_name.json", orient="records", date_format="iso")
#
# print(df_stocks.sample())
import investpy
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stocks_name(json_path="Stocks_name.json"):
    try:
        df_stocks = investpy.get_stocks(country='united states')[['symbol', 'name']]

        # Simply save to json — this overwrites if file exists
        df_stocks.to_json(json_path, orient="records", date_format="iso")
        print(f"Saved {len(df_stocks)} stock entries to {json_path}")
        print(df_stocks.sample())
        return df_stocks

    except Exception as e:
        print(f"Failed to retrieve stock list: {e}")
        return pd.DataFrame()



def download_and_save_indexes_to_json(json_path="index_data.json",days_index=365):
    index_tickers = {
        "NASDAQ": "^IXIC",
        "S&P500": "^GSPC",
        "DOWJONES": "^DJI"
    }

    end_date = datetime.today()
    start_date = end_date - timedelta(days_index)

    data_frames = []

    for name, ticker in index_tickers.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        if df.empty:
            print(f"No data for {name}")
            continue

        # Rename columns to include index name for clarity
        df = df[["Close"]].rename(columns={"Close": name})
        data_frames.append(df)

    if not data_frames:
        print("No index data was downloaded.")
        return



    # Merge all data on date
    merged_df = pd.concat(data_frames, axis=1)
    merged_df.index.name = "Date"
    print(merged_df.shape)

    # Save to JSON (records by date)
    merged_df.to_json(json_path, orient="index", date_format="iso")
    print(f"Index data saved to {json_path}")
    return merged_df

get_stocks_name()
download_and_save_indexes_to_json()

