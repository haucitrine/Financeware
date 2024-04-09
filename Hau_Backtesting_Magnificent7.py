import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

###### DATA EXTRACTION ######

available_symbols = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOG", "META", "TSLA"]
# Display available symbols for selection
print("Available stock symbols:")
for i, symbol in enumerate(available_symbols, 1):
    print(f"{i}. {symbol}")

# Prompt the user to choose a symbol
while True:
    selection = input("Enter the number corresponding to the stock symbol or enter the stock symbol: ")
    try:
        selection_index = int(selection) - 1
        if 0 <= selection_index < len(available_symbols):
            selected_symbol = available_symbols[selection_index]
            print(f"You've selected: {selected_symbol}")
            break  # Exit the loop if a valid selection is made
        else:
            print("Invalid selection. Please enter a number within the given range.")
    except ValueError:
        # Check if the input matches any of the available symbols (case-insensitive)
        selected_symbol = next((symbol for symbol in available_symbols if symbol.lower() == selection.lower()), None)
        if selected_symbol:
            print(f"You've selected: {selected_symbol}")
            break  # Exit the loop if a valid selection is made
        else:
            print("Invalid input. Please enter a valid symbol.")

# Define start and end dates
start_date = "2013-01-01"
end_date = "2023-12-31"

if selected_symbol:
    def extract_data(selected_symbol, start_date, end_date):
        data = yf.download(selected_symbol, start=start_date, end=end_date)
        return data

    # Extract historical data
    historical_data = extract_data(selected_symbol, start_date, end_date)

# Print the first row of the table
print("5 first rows of the table:")
print(historical_data.head(5))

# Print the last row of the table
print("\n5 last row of the table:")
print(historical_data.tail(5))


###### DOUBLE BOLLINGER BANDS INDICATOR ######
# Definition
period = 20
deviations_wide = 2
deviations_narrow = 1

# Calculate Bollinger Bands
historical_data['SMA'] = historical_data['Close'].rolling(window=period).mean()
historical_data['std'] = historical_data['Close'].rolling(window=period).std()

historical_data['A1'] = historical_data['SMA'] + (historical_data['std'] * deviations_wide)
historical_data['A2'] = historical_data['SMA'] - (historical_data['std'] * deviations_wide)

historical_data['B1'] = historical_data['SMA'] + (historical_data['std'] * deviations_narrow)
historical_data['B2'] = historical_data['SMA'] - (historical_data['std'] * deviations_narrow)

historical_data = historical_data.dropna()

# Trading Signal Strategy
def trading_strategy(row):
    if row['B1'] < row['Close'] < row['A1']:
        return "Buy"
    elif row['A2'] < row['Close'] < row['B2']:
        return "Sell"
    elif row['B2'] <= row['Close'] <= row['B1']:
        return "Neutral"
    else:
        return "No Trade"

# Add signals to the DataFrame
historical_data['Signal'] = historical_data.apply(trading_strategy, axis=1)

# Position Management Strategy
# Buy Position Management
def buy_position_management(data):
    buy_position = False
    for index, row in data.iterrows():
        if row['Signal'] == "Buy":
            if not buy_position:  # If not already in a buy position
                buy_position = True
                data.at[index, 'Position'] = "Buy"  # Add position information to the dataframe
                data.at[index, 'Entry Price'] = row['Close']  # Add entry price
                data.at[index, 'Stop Loss'] = row['B1']  # Add stop loss level
        elif buy_position and row['Close'] >= row['A1']:  # Take profits if price exceeds upper wide band (A1)
            buy_position = False
            data.at[index, 'Position'] = "Close Buy"  # Add position closing information
            data.at[index, 'Exit Price'] = row['Close']  # Add exit price
            entry_price = data.at[index, 'Entry Price'] if data.at[index, 'Entry Price'] is not None else 0
            data.at[index, 'Profit'] = row['Close'] - entry_price  # Calculate profit

# Sell Position Management
def sell_position_management(data):
    sell_position = False
    for index, row in data.iterrows():
        if row['Signal'] == "Sell":
            if not sell_position:  # If not already in a sell position
                sell_position = True
                data.at[index, 'Position'] = "Sell"  # Add position information to the dataframe
                data.at[index, 'Entry Price'] = row['Close']  # Add entry price
                data.at[index, 'Stop Loss'] = row['B2']  # Add stop loss level
        elif sell_position and row['Close'] <= row['A2']:  # Take profits if price falls below lower wide band (A2)
            sell_position = False
            data.at[index, 'Position'] = "Close Sell"  # Add position closing information
            data.at[index, 'Exit Price'] = row['Close']  # Add exit price
            entry_price = data.at[index, 'Entry Price'] if data.at[index, 'Entry Price'] is not None else 0
            data.at[index, 'Profit'] = entry_price - row['Close']  # Calculate profit


# Initialize additional columns for position information
historical_data['Position'] = ""
historical_data['Entry Price'] = None
historical_data['Exit Price'] = None
historical_data['Stop Loss'] = None
historical_data['Profit'] = None

# Implementing the strategy
buy_position_management(historical_data)
sell_position_management(historical_data)

# Display the updated dataframe with position information
print(historical_data)

# Define a function to plot the data
def plot_data(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price', color='black')
    plt.plot(data.index, data['SMA'], label='SMA', color='blue')
    plt.fill_between(data.index, data['A1'], data['A2'], color='gray', alpha=0.3, label='Bollinger Bands Wide')
    plt.fill_between(data.index, data['B1'], data['B2'], color='gray', alpha=0.5, label='Bollinger Bands Narrow')
    
    # Plot buy and sell signals
    plt.plot(data[data['Signal'] == 'Buy'].index, 
             data[data['Signal'] == 'Buy']['Close'], 
             '^', markersize=10, color='b', lw=0, label='Buy Signal')
    plt.plot(data[data['Signal'] == 'Sell'].index, 
             data[data['Signal'] == 'Sell']['Close'], 
             'v', markersize=10, color='m', lw=0, label='Sell Signal')
    
    # Plot entry and exit prices
    plt.plot(data[data['Position'] == 'Buy'].index, 
             data[data['Position'] == 'Buy']['Entry Price'], 
             '^', markersize=10, color='g', lw=0, label='Buy Entry')
    plt.plot(data[data['Position'] == 'Close Buy'].index, 
             data[data['Position'] == 'Close Buy']['Exit Price'], 
             'o', markersize=10, color='g', lw=0, label='Buy Exit')
    plt.plot(data[data['Position'] == 'Sell'].index, 
             data[data['Position'] == 'Sell']['Entry Price'], 
             'v', markersize=10, color='r', lw=0, label='Sell Entry')
    plt.plot(data[data['Position'] == 'Close Sell'].index, 
             data[data['Position'] == 'Close Sell']['Exit Price'], 
             'o', markersize=10, color='r', lw=0, label='Sell Exit')
    
    plt.title('Bollinger Bands and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the data
plot_data(historical_data)


###### BACKTESTING LOOP ###### 
# Initialize parameters
initial_capital = 10000  # Initial capital
portfolio_value = initial_capital  # Portfolio value starts with initial capital
cash = initial_capital  # Initially, all capital is in cash
risk_per_trade = 0.02  # Risk per trade (2% of portfolio value)
minimum_transaction_size = 1  # Minimum transaction size (1 share per trade)
stop_loss_percentage = 0.05  # 5% stop-loss

# Initialize variables to track trades and portfolio value
trades = []
portfolio_values = []
completed_trades = []

# Loop through each row in the historical data
for index, row in historical_data.iterrows():
    # Calculate position size based on volatility
    position_size = max(int((cash * risk_per_trade) / (row['Close'] * row['std'])), minimum_transaction_size)
    
    # Buy signal
    if row['Position'] == 'Buy':
        # Calculate transaction cost
        transaction_cost = position_size * row['Close']
        
        # Check if enough cash available for the transaction
        if cash >= transaction_cost:
            # Execute buy trade
            cash -= transaction_cost  # Reduce cash
            portfolio_value -= transaction_cost  # Reduce portfolio value by transaction cost
            trades.append(('Buy', row.name, row['Close'], position_size))  # Record trade
            print(f"Buy Trade at {row.name}, Price: {row['Close']}, Size: {position_size}")
    
    # Sell signal
    elif row['Position'] == 'Close Buy':
        # Get the entry price from the corresponding 'Buy' trade
        entry_price = None
        for trade in reversed(trades):
            if trade[0] == 'Buy':
                entry_price = trade[2]
                trades.remove(trade)  # Remove the 'Buy' trade from the list
                completed_trades.append(trade)  # Add completed trade to the list
                break
        
        if entry_price is not None:
            # Calculate transaction revenue
            transaction_revenue = position_size * row['Close']
            
            # Execute sell trade
            cash += transaction_revenue  # Add revenue to cash
            portfolio_value += transaction_revenue  # Add revenue to portfolio value
            trades.append(('Sell', row.name, row['Close'], position_size, entry_price))  # Record trade
            print(f"Sell Trade at {row.name}, Price: {row['Close']}, Size: {position_size}")
    
    # Short signal
    elif row['Position'] == 'Sell':
        # Calculate transaction cost for short position
        transaction_cost = position_size * row['Close']
        
        # Check if enough cash available for the transaction
        if cash >= transaction_cost:
            # Execute short trade
            cash -= transaction_cost  # Reduce cash
            portfolio_value -= transaction_cost  # Reduce portfolio value by transaction cost
            trades.append(('Sell', row.name, row['Close'], position_size))  # Record trade
            print(f"Sell Trade at {row.name}, Price: {row['Close']}, Size: {position_size}")
    
    # Close short position signal
    elif row['Position'] == 'Close Sell':
        # Get the entry price from the corresponding 'Sell' trade
        entry_price = None
        for trade in reversed(trades):
            if trade[0] == 'Sell':
                entry_price = trade[2]
                trades.remove(trade)  # Remove the 'Sell' trade from the list
                completed_trades.append(trade)  # Add completed trade to the list
                break
        
        if entry_price is not None:
            # Calculate transaction revenue for closing short position
            transaction_revenue = position_size * row['Close']
            
            # Execute close short position trade
            cash += transaction_revenue  # Add revenue to cash
            portfolio_value += transaction_revenue  # Add revenue to portfolio value
            trades.append(('Close Sell', row.name, row['Close'], position_size, entry_price))  # Record trade
            print(f"Close Sell Trade at {row.name}, Price: {row['Close']}, Size: {position_size}")
    
    # Check for stop-loss conditions
    for trade in trades:
        if trade[0] == 'Buy':
            stop_loss_price = trade[2] * (1 - risk_per_trade)
            if row['Close'] <= stop_loss_price:
                # Execute stop-loss for buy position
                cash += trade[3] * row['Close']  # Add cash from stop-loss
                portfolio_value += trade[3] * row['Close']  # Adjust portfolio value
                trades.remove(trade)  # Remove the trade from the list
                print(f"Stop-loss executed for Buy Trade at {row.name}, Price: {row['Close']}")
        elif trade[0] == 'Sell':
            stop_loss_price = trade[2] * (1 + risk_per_trade)
            if row['Close'] >= stop_loss_price:
                # Execute stop-loss for sell position
                cash += trade[3] * row['Close']  # Add cash from stop-loss
                portfolio_value += trade[3] * row['Close']  # Adjust portfolio value
                trades.remove(trade)  # Remove the trade from the list
                print(f"Stop-loss executed for Sell Trade at {row.name}, Price: {row['Close']}")
    
    # Record portfolio value after each trade
    portfolio_values.append(portfolio_value)

# Add remaining cash to the final portfolio value
portfolio_value += cash

# Calculate total profit
total_profit = portfolio_value - initial_capital

# Print results
print("\nFinal Portfolio Value: $", portfolio_value)
print("Total Profit/Loss: $", total_profit)
print("Total Number of Trades:", len(completed_trades))


###### PERFORMANCE METRICS ###### 
def calculate_performance_metrics(portfolio_value, initial_capital, historical_data):
    # Calculate Total Return
    total_return = ((portfolio_value - initial_capital) / initial_capital) * 100

    # Calculate Annual Return
    num_years = len(historical_data) / 252  # Assuming 252 trading days in a year
    annual_return = ((portfolio_value / initial_capital) ** (1 / num_years) - 1) * 100

    # Calculate Daily Returns
    daily_returns = historical_data['Close'].pct_change()

    # Calculate Annual Volatility
    annual_volatility = daily_returns.std() * np.sqrt(252)

    # Calculate Sharpe Ratio (assuming risk-free rate of 0)
    daily_rf_rate = 0
    sharpe_ratio = (annual_return - daily_rf_rate) / annual_volatility

    # Calculate Sortino Ratio (assuming risk-free rate of 0)
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - daily_rf_rate) / downside_volatility

    # Calculate Maximum Drawdown
    cumulative_returns = (daily_returns + 1).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Total Return (%)": total_return,
        "Annual Return (%)": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": max_drawdown
    }

performance_metrics = calculate_performance_metrics(portfolio_value, initial_capital, historical_data)

print("\nPerformance Metrics:")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value}")

################# NGUYEN DUY HAU #################