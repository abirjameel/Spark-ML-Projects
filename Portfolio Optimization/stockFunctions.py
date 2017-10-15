import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt 
import stockFunctions as sf 
import datetime as dt 

def compute_daily_returns(df):
    """Compute and return the daily return values.
    takes a data frame, and apply operations over its columns 
    """
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    df_ret = df
    df_ret[1:] = (df[1:] / df[:-1].values) - 1
    df_ret.ix[0, :] = 0
    return df_ret

def normalize_data(df):
    """returns the normalized data divided by the first observation"""
    return df/ df.ix[0]

def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    ax = df.ix[start_index:end_index, columns].plot(title="Stock Prices for Multiple Index", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', 
                              usecols= ['Date', 'Adj Close'], na_values=['nan'], parse_dates=True)
        df_temp = df_temp.rename(columns = {'Adj Close': symbol})
        df = df.join(df_temp)
        #df = df.dropna(subset = symbol)
        if symbol == 'SPY':
            df = df.dropna(subset = ["SPY"])
    return df

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join("{}.csv".format(str(symbol)))

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    ##########################################################
    df_data.fillna(method = "ffill", inplace = True)
    df_data.fillna(method = "backfill", inplace = True)
    return df_data

def Compute_stats(df, allocated_tuple):
    pos_vals = df*list(allocated_tuple)
    pos_vals['port_val'] = pos_vals.sum(axis = 1)
    daily_returns = compute_daily_returns(pos_vals)
    daily_returns = daily_returns[1:]
    cum_ret = (daily_returns['port_val'].ix[-1] / daily_returns['port_val'].ix[0]) -1
    avg_daily_ret = daily_returns['port_val'].mean()
    std_daily_ret = daily_returns['port_val'].std()
    sr = (avg_daily_ret/std_daily_ret) * np.sqrt(252)
    return cum_ret, avg_daily_ret, std_daily_ret, sr

def Optimize_alloc(fn , allocated_tuple):
    from scipy.optimize import minimize
    bnds = []
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
           {'type': 'eq', 'fun': lambda x: x > 0})
    for i in range(0, len(allocated_tuple)):
        bnds.append((0.0,1.0))
    bnds = tuple(bnds)
    result = dict(minimize(fn,allocated_tuple,method='SLSQP',bounds=bnds, constraints=cons ,options={'disp':False}))
    return result['x']

def Compute_daily_port_val(df, allocations):
    pos_vals = df*allocations
    pos_vals['port_val'] = pos_vals.sum(axis = 1)
    return pos_vals['port_val']