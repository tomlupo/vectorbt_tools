import yfinance as yf

# Define the tickers as a space-separated string
tickers = "AAPL MSFT AMZN GOOGL META TSLA NVDA"

# Create a Tickers object and download historical data for the past 5 years
data = yf.Tickers(tickers).history(period="5y")

# Display the first few rows of the data
print(data.head())

# Save the data to pickle file
data.to_pickle("data/yfinance_mag7.pkl")
