import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch data for a specific stock (e.g., Apple)
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-09-01')
print(stock_data.head())
stock_data.info()

stock_data.fillna(method='ffill', inplace=True)  # Fill missing values
stock_data.reset_index(inplace=True)  # Reset index to make 'Date' a column

fig = px.line(stock_data, x='Date', y='Close', title='Stock Price Over Time')
fig.show()

stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()  # 50-day EMA

stock_data['Lag_1'] = stock_data['Close'].shift(1)  # Previous day's closing price
stock_data['Lag_7'] = stock_data['Close'].shift(7)  # Closing price a week ago
stock_data.fillna(method='bfill', inplace=True) 

print(stock_data.head())


# Compare the actual stock price (Close) with the Exponential Moving Average (EMA_50) to identify trends and signals (e.g., potential buy/sell points).
plt.figure(figsize=(10,6))
plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data['Date'], stock_data['EMA_50'], label='EMA_50', color='red')
plt.title('Close Price vs EMA_50')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


#Examine the correlation between different lagged features and the EMA to see if past values have predictive power.

# Correlation matrix
corr_matrix = stock_data[['Close', 'EMA_50', 'Lag_1', 'Lag_7']].corr()
# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap: Close, EMA_50, Lag_1, Lag_7')
plt.show()

