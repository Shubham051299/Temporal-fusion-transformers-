import pandas as pd
import yfinance as yf
from stockstats import wrap, unwrap
import numpy as np

# Define constants
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-04'
TEST_END_DATE = '2023-12-29'
DOW_30_TICKER = [
    'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON',
    'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
    'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'GOOGL', 
    'AMZN', 'TSLA' , 'META', 'NVDA', 'NFLX', 'BABA', 'MA', 'PYPL', 'BAC',
    'XOM', 'PFE', 'ORCL', 'ABT', 'PEP',
]

# Get the historical price data from yahoo finance
total_df = pd.DataFrame()
for ticker in DOW_30_TICKER:
    stock = yf.Ticker(ticker)
    data = stock.history(start=TRAIN_START_DATE, end=TEST_END_DATE)
    data["Ticker"] = ticker  # Ensure the 'Ticker' column is properly named

    # Calculate technical indicators using stockstats library
    stock_df = wrap(data)
    stock_df[["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]]
    unwrap_df = unwrap(stock_df)
    unwrap_df = unwrap_df.fillna(0)
    unwrap_df = unwrap_df.replace(np.inf, 0)

    total_df = total_df._append(unwrap_df)

total_df = total_df.drop(columns=["dividends", "stock splits", "macds", "macdh", "boll"])
total_df.reset_index(inplace=True)

# Convert all column names to lowercase to ensure consistency
total_df.columns = [col.lower() for col in total_df.columns]

# Save preprocessed data
total_df.to_csv("preprocessed_data.csv", index=False)

# Feature engineering
def calculate_turbulence(data, time_period=252):
    df = data.copy()
    df_price_pivot = df.pivot(index="date", columns="ticker", values="close")  # Ensure 'ticker' column is used
    df_price_pivot = df_price_pivot.pct_change()

    unique_date = df.date.unique()
    start = time_period
    turbulence_index = [0] * start
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[(df_price_pivot.index < unique_date[i]) & (df_price_pivot.index >= unique_date[i - time_period])]
        filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)
        cov_temp = filtered_hist_price.cov()
        current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
        temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(current_temp.values.T)
        turbulence_temp = temp[0][0] if temp > 0 else 0
        turbulence_index.append(turbulence_temp if count > 2 else 0)
        if temp > 0:
            count += 1

    turbulence_index = pd.DataFrame({"date": df_price_pivot.index, "turbulence": turbulence_index})
    return turbulence_index

turbulence_index = calculate_turbulence(total_df, time_period=252)
total_df = total_df.merge(turbulence_index, on="date")
total_df.to_csv("preprocessed_data_with_turbulence.csv", index=False)
