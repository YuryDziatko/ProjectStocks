
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
def get_stocks_name():
    df_stocks = investpy.get_stocks(country='united states')[['symbol', 'name']]
    df_stocks.to_json("Stocks_name.json", orient="records", date_format="iso")
    print(df_stocks.sample())
    return df_stocks