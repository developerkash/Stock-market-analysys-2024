#import libraries
import streamlit as st # for building web app 
import yfinance as yf # to get stock data from yahoo finance API 
import pandas as pd # for data manipulation and analysis 
import numpy as np # for numerical computation 
import matplotlib.pyplot as plt # for data visualization 
import seaborn as sns   # for data visualization 
import plotly.graph_objects as go # for data visualization 
import plotly.express as px    # for data visualization 
import datetime # for date and time manipulation
from datetime import date, timedelta # for date manipulation
from statsmodels.tsa.seasonal import seasonal_decompose # for time series analysis
import statsmodels.api as sm # for time series analysis
from statsmodels.tsa.stattools import adfuller # for time series analysis

#Title 
st.set_page_config(page_title="Stock Market Analysis", page_icon="📈", layout="wide")
app_name = "Stock Market Analysis"
st.title(app_name)
st.subheader("Welcome to the Stock Market Analysis and Prediction App")

# Add a image
# st.image("https://media.istockphoto.com/id/1478940432/photo/investment-background-recession-global-market-crisis-inflation-deflation-digital-data.jpg?s=2048x2048&w=is&k=20&c=O4lmjIxlcO3UJx38xBGNckxEKSlZZfdy1kYR17z-Cyk=", use_column_width=True, width=500)

# loading data from yfinance API



# Take user input of stock ticker symbol and number of days
# Sidebar
st.sidebar.header("User Input Features")

# Add a ticker symbol input field 
ticker = st.sidebar.text_input("Enter the Stock Ticker Symbol", 'CYIENT.NS')

# Add a date range selection
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2021, 1, 1))

# Add a period selection
selected_period = st.sidebar.selectbox("Select Period", ["", "1d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])

# Fetch the stock data from Yfinance API
if(selected_period):
    data = yf.download(ticker, period=selected_period)
    fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Stock Prices')
    st.plotly_chart(fig, use_container_width=True)
else:
    data =yf.download(ticker, start=start_date, end=end_date) 
    fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Stock Prices')
    st.plotly_chart(fig, use_container_width=True)

# Company's Stock Information
company_his = yf.Ticker(ticker)

c1, c2 = st.columns((1,1))

with c1:
        col1, col2 = st.columns((1,1))
        with col1:
            st.write("All time Low: ")
            st.write("52 Week Low: ")        
        with col2:
            st.write(f"{company_his.info['fiftyTwoWeekLow']}")    
            st.write(f"{company_his.info['regularMarketDayLow']}")  
        
with c2:
        col1, col2 = st.columns((1,1))
        with col1:
            st.write("All time High: ") 
            st.write("52 Week High: ")   
        with col2:
            st.write(f"{company_his.info['fiftyTwoWeekHigh']}")
            st.write(f"{company_his.info['regularMarketDayHigh']}")


# Fetching company info
company_info = yf.Ticker(ticker)

fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Stock Prices')

# Company's Basic Information
st.subheader("Company Basic Information")
with st.expander("Company Basic Information"):
    c1, c2 = st.columns((1,1))

    with c1:
        col1, col2 = st.columns((1,1))
        with col1:
            st.write("Company Name: ")
            st.write("Symbol: ")
            st.write("sector: ")
            st.write("Industry: ")
        with col2:
            st.write(f"{company_info.info['longName']}")
            st.write(f"{company_info.info['symbol']}")
            st.write(f"{company_info.info['sector']}")
            st.write(f"{company_info.info['industry']}")
       
    with c2:
        col1, col2 = st.columns((1,1))
        with col1:
            st.write("CEO & Executive Director: ")
            st.write("Phone: ")   
            st.write("fax: ") 
            st.write("Website: ")
        with col2:
            st.write(f"{company_info.info['companyOfficers'][0]['name' ]}")
            st.write(f"{company_info.info['phone']}")
            st.write(f"{company_info.info['fax']}")
            st.write(f"{company_info.info['website']}")
        

# making a div 
with st.expander("Company financial Information"):
    c1, c2 = st.columns((1,1))

    with c1:
        col1, col2 = st.columns((1,1))
        with col1:
            st.write("Company P/E Ratio: ")
            st.write("Dividend Yield: ")
            st.write("Market Cap: ") 
        with col2:
            st.write(f"{company_info.info['trailingPE']}")
            st.write(f"{company_info.info['dividendYield']}")
            st.write(f"{company_info.info['marketCap']}")

    with c2:
        col1, col2 = st.columns((1,1))
        with col1:
            st.write("Market Cap: ")
            st.write("Beta: ")
        with col2:
            st.write(f"{company_info.info['marketCap']}")
            st.write(f"{company_info.info['beta']}")




st.subheader("Company financial Statement")

Statements = yf.Ticker(ticker)
balance_sheet = Statements.balance_sheet
income_statement = Statements.financials
cash_flow = Statements.cashflow


with st.expander("Company Balance Sheet"):
    st.dataframe(balance_sheet, use_container_width=True)

with st.expander("Company Income Statement"):
    st.dataframe(income_statement, use_container_width=True)

with st.expander("Company Cash Flow"):
    st.dataframe(cash_flow, use_container_width=True)





def get_shareholder_pattern(ticker):
    # Fetch shareholder information from Yahoo Finance
    stock = yf.Ticker(ticker)
    shareholder_df = stock.institutional_holders
    
    return shareholder_df

# Streamlit app
st.title('Shareholder Pattern')

# Input box for user to enter company symbol

# Button to fetch and display shareholder pattern
if st.button('Fetch Shareholder Pattern'):
    shareholder_df = get_shareholder_pattern(ticker)
    st.write(shareholder_df)





# Technical Analysis of Stock Prices 
st.subheader("Technical Analysis of Stock Prices")
# Simple Moving Average
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_100'] = data['Close'].rolling(window=100).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Exponential Moving Average
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_100'], name='SMA 100', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], name='SMA 200', line=dict(color='red', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices with Moving Averages', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Exponential Moving Average
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], name='EMA 50', line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA_100'], name='EMA 100', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], name='EMA 200', line=dict(color='red', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices with Exponential Moving Averages', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Bollinger Bands
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Upper'] = data['SMA_20'] + 2*data['Close'].rolling(window=20).std()
data['Lower'] = data['SMA_20'] - 2*data['Close'].rolling(window=20).std()

# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], name='Upper Band', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], name='Lower Band', line=dict(color='red', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices with Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Relative Strength Index
delta = data['Close'].diff()
gain = (delta.where(delta>0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta<0, 0)).rolling(window=14).mean()
RS = gain/loss
data['RSI'] = 100 - (100/(1+RS))

# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='blue', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices with Relative Strength Index', xaxis_title='Date', yaxis_title='RSI')
st.plotly_chart(fig, use_container_width=True)

# Moving Average Convergence Divergence
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], name='Signal Line', line=dict(color='red', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices with Moving Average Convergence Divergence', xaxis_title='Date', yaxis_title='MACD')
st.plotly_chart(fig, use_container_width=True)

# Forecasting Stock Prices
st.subheader("Forecasting Stock Prices")

# Extracting the Close Price
original_stock_data = data
data = data[['Close']]

# Drop rows with missing values
data = data.dropna()

# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Decomposing the Time Series
result = seasonal_decompose(data['Close'], model='multiplicative', period=30)
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=result.trend, name='Trend', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=result.seasonal, name='Seasonal', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=result.resid, name='Residual', line=dict(color='green', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices Decomposition', xaxis_title='Date', yaxis_title='Price')

# Forecasting the Time Series
# Splitting the data into training and testing data
train = data[:int(0.8*(len(data)))]
test = data[int(0.8*(len(data))):]
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=test.index, y=test['Close'], name='Test', line=dict(color='red', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices Train and Test Split', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Augmented Dickey-Fuller Test
result = adfuller(data['Close'])
st.write('ADF Statistic:', result[0])
st.write('p-value:', result[1])
st.write('Critical Values:', result[4])

# ARIMA Model
model = sm.tsa.ARIMA(train, order=(5,1,0))
model = model.fit()
# Forecasting the data
forecast = model.forecast(steps=len(test))
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=test.index, y=test['Close'], name='Test', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=test.index, y=forecast, name='Forecast', line=dict(color='green', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices ARIMA Model', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# show the data
data = original_stock_data
st.subheader("Stock Data")
st.write(data)

# candelstick chart
st.subheader("Candlestick Chart")
# Plotting the data
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Line chart
st.subheader("Line Chart")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Area chart
st.subheader("Area Chart")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', fill='tozeroy', line=dict(color='blue', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

# Bar chart
st.subheader("Bar Chart")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Date', yaxis_title='Volume')
st.plotly_chart(fig, use_container_width=True)

# Scatter plot
st.subheader("Scatter Plot")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Open'], y=data['Close'], mode='markers', marker=dict(color='blue', size=8)))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Open Price', yaxis_title='Close Price')
st.plotly_chart(fig, use_container_width=True)

# Histogram
st.subheader("Histogram")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Histogram(x=data['Close'], marker_color='blue'))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Close Price', yaxis_title='Frequency')
st.plotly_chart(fig, use_container_width=True)

# Box plot
st.subheader("Box Plot")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Box(y=data['Close'], marker_color='blue'))
fig.update_layout(title=f'{ticker} Stock Prices', yaxis_title='Close Price')
st.plotly_chart(fig, use_container_width=True)

# Heatmap
st.subheader("Heatmap")
# Plotting the data
import seaborn as sns
correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig)


# Line chart with multiple lines
st.subheader("Line Chart with Multiple Lines")
# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name='Open Price', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['High'], name='High Price', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['Low'], name='Low Price', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='orange', width=2)))
fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)





# Closing Note
st.subheader("Closing Note")
st.write("This is a simple Stock Market Analysis and Prediction App. The data is fetched from Yahoo Finance API and the predictions are made using ARIMA and LSTM models. The app is for educational purposes only and not for financial advice. Please do your own research before making any investment decisions.")
st.write("Thank you for using the app. Have a great day!")

# Add a footer
st.markdown(
    """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
    , unsafe_allow_html=True
)



