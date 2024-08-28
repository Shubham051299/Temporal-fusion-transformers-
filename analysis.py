import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Load the predictions file
predictions_df = pd.read_csv(r'C:\Users\shubh\Desktop\AI_Final_Project\predictions.csv')

# Ensure the 'date' column is treated as datetime and convert to UTC
predictions_df['date'] = pd.to_datetime(predictions_df['date'], utc=True).dt.tz_convert(None)

# Define the tickers and the date range for fetching actual values
tickers = predictions_df['ticker'].unique()
start_date = pd.to_datetime(predictions_df['date'].min()).date()
end_date = pd.to_datetime(predictions_df['date'].max()).date()

# Fetch the actual stock prices from Yahoo Finance
actual_values_df = pd.DataFrame()

for ticker in tickers:
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data['ticker'] = ticker
    data['date'] = data.index
    data = data.reset_index(drop=True)
    actual_values_df = pd.concat([actual_values_df, data])

# Ensure consistent date format without timezone
actual_values_df['date'] = pd.to_datetime(actual_values_df['date'], utc=True).dt.tz_convert(None)
predictions_df['date'] = pd.to_datetime(predictions_df['date'], utc=True).dt.tz_convert(None)

# Merge actual values with predictions
merged_df = pd.merge(predictions_df, actual_values_df[['date', 'ticker', 'Close']], on=['date', 'ticker'], suffixes=('', '_actual'))

# Rename 'Close' column to 'Close_actual' to avoid confusion
merged_df.rename(columns={'Close': 'Close_actual'}, inplace=True)

# Calculate MAE and MAPE for each stock
for col in ['0', '1', '2', '3', '4']:
    merged_df[f'mae_{col}'] = (merged_df[col] - merged_df['Close_actual']).abs()
    merged_df[f'mape_{col}'] = (merged_df[f'mae_{col}'] / merged_df['Close_actual']).abs() * 100  # Convert to percentage

# Calculate the average MAE and MAPE for each stock
merged_df['mae'] = merged_df[[f'mae_{col}' for col in ['0', '1', '2', '3', '4']]].mean(axis=1)
merged_df['mape'] = merged_df[[f'mape_{col}' for col in ['0', '1', '2', '3', '4']]].mean(axis=1)

# Calculate the average predicted price for each stock
merged_df['average_predicted_price'] = merged_df[['0', '1', '2', '3', '4']].mean(axis=1)

# Calculate the trend by comparing the first and last predicted prices
merged_df['predicted_trend'] = merged_df['4'] - merged_df['0']

# Group by ticker and calculate mean average_predicted_price, mean predicted_trend, mean mae, mean mape, actual price, and predicted price
summary_df = merged_df.groupby('ticker').agg(
    average_predicted_price=('average_predicted_price', 'mean'),
    predicted_trend=('predicted_trend', 'mean'),
    mae=('mae', 'mean'),
    mape=('mape', 'mean'),
    actual_price=('Close_actual', 'mean'),  # Average actual price
    predicted_price=('average_predicted_price', 'mean')  # Average predicted price
).reset_index()

# Sort by predicted_trend in descending order to find stocks with increasing trends
sorted_summary_df = summary_df.sort_values(by='predicted_trend', ascending=False)

# Plot the top 10 stocks with the highest predicted trend
top_10_stocks = sorted_summary_df.head(10)
plt.figure(figsize=(10, 6))
plt.bar(top_10_stocks['ticker'], top_10_stocks['predicted_trend'], color='green')
plt.xlabel('Ticker')
plt.ylabel('Predicted Trend')
plt.title('Top 10 Stocks with Highest Predicted Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_10_stocks_predicted_trend.png')
plt.show()

# Save the sorted summary to a CSV file
sorted_summary_df.to_csv("sorted_stock_predictions_summary.csv", index=False)

# Print out the MAE and MAPE for each stock
print("MAE and MAPE for each stock:")
print(sorted_summary_df[['ticker', 'mae', 'mape', 'actual_price', 'predicted_price']])

print("Analysis complete. Results saved to sorted_stock_predictions_summary.csv and top_10_stocks_predicted_trend.png")
