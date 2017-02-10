---
layout: post
title: "Yet Another Algorithmic Trading Tutorial (Part I)"
description: "We use Python to visualize stock market data and optimize a portfolio based on historical data."
comments: true
keywords: "finance, trading, stocks, python, portfolio"
---

Since this is my first post, I am going to start off with a small digression. I am a big fan of continued education
from online sources. My favorite places to look are [Standford SEE](https://see.stanford.edu/),
[MIT OCW](https://ocw.mit.edu/), [Harvard OL](https://www.extension.harvard.edu/open-learning-initiative), 
[coursera](https://www.coursera.org/) and [edX](https://www.edx.org/). If you want a 
referesher on undergraduate probability theory, the best course I have found is 
[STAT110](http://projects.iq.harvard.edu/stat110/home), a course by 
Professor Joe Blitzstein at Harvard. He also helps run a data science course, 
[CS109](http://cs109.github.io/2015/index.html), which gives a nice introduction to using
machine learning with Python. 

This tutorial actually follows [Machine Learning for Trading](https://classroom.udacity.com/courses/ud501/), a 
course on Udacity (not my favorite, sorry Udacity!), which you can take yourself and follow along. The course is by 
Dr. Tucker Balch at Georgia Tech, but sadly doesn't actually link to any of the homework 
files. However, you can find a short description of them
[here](http://quantsoftware.gatech.edu/Machine_Learning_for_Trading_Course). This 
description will be what we use to develop our tools for this tutorial. In Part I, we will
develop tools to visualize stock data, evaluate the performance of a portfolio and optimize
a portfolio on historical data. Okay, let's get started!

<div class="divider"></div>

**DISCLAIMER!** I am not a financial professional, so do not use this tutorial to attempt
to trade stocks without EXTREME caution!

<div class="divider"></div>

I will be using Python 3 to do this analysis. You can find the python files associated with this tutorial
[here](https://github.com/danielwo/algo-trading). You can check your Python libraries against mine using:

~~~python
import sys
# Get python version :D
print("Python version:            %6.6s (my version: 3.5.2)" % sys.version)

# Numpy is a library for working with Arrays
import numpy as np
print("Numpy version:             %6.6s (my version: 1.12.0)" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print("SciPy version:             %6.6s (my version: 0.18.1)" % sp.__version__)

# Pandas makes working with data tables easier
import pandas as pd
print("Pandas version:            %6.6s (my version: 0.19.2)" % pd.__version__)

# Pandas_datareader makes downloading stock market data easy
import pandas_datareader as web
print("Pandas_datareader version: %6.6s (my version: 0.3.0)" % web.__version__)

# Module for plotting
import matplotlib
print("Mapltolib version:        %6.6s  (my version: 2.0.0)" % matplotlib.__version__)

# Seaborn makes matplotlib look good and provides some plotting functions
import seaborn
print("Seaborn version:          %6.6s  (my version: 0.7.1)" % seaborn.__version__)
~~~
>~~~
Python version:            3.5.2  (my version: 3.5.2)
Numpy version:             1.12.0 (my version: 1.12.0)
SciPy version:             0.18.1 (my version: 0.18.1)
Pandas version:            0.19.2 (my version: 0.19.2)
Pandas_datareader version: 0.3.0. (my version: 0.3.0)
Mapltolib version:         2.0.0  (my version: 2.0.0)
Seaborn version:           0.7.1  (my version: 0.7.1)
~~~

As you can see we will be using numpy, scipy, pandas, matplotlib, and seaborn.
Even if you have pandas, you might not have [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/),
so you may want to install that first. We will also be using [seaborn](http://seaborn.pydata.org/)
which makes it much easier (at least for me) to make nice looking graphs. If you are unfamiliar
with pandas, you may want to take some time to figure out how it works. If you've used 
R, then pandas contains basic functionality for dataframes.

Okay, so pandas_datareader will make our work much easier, since we won't have to deal with
wondering how to get our hands on stock data. In particular, we will use the `get_data_yahoo`
function download historical stock market data from Yahoo! Finance. In order to use this function,
we need to know the ticker symbol for the stock we are interested in (e.g. 'GOOG' for Google),
the start date, the end date,  and the unit of time between individual records (here we will be using days).

We probably also want to save the data somewhere, so let's set up a function to get the
path to our data:

~~~python
import os

def symbol_to_path(symbol, base_dir="stock_data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))
~~~

Here we take in a symbol (e.g. 'GOOG'), and the directory where the file is. Here I use
"stock_data" as my base directory. The function then returns the file path. Now let's use
pandas and pandas_datareader to scrape some stock data:

~~~python
import pandas as pd
import pandas_datareader.data as web

def scrape_stock_data(symbols, start, end, interval='d', base_dir="stock_data"):
    """Get yahoo stock data from the web and save it as .csv files"""
    if isinstance(symbols, str): 
        symbols = [symbols] # if string is passed in, place it in a list
        
    for symbol in symbols:
        df = web.get_data_yahoo(symbol, start = start, end = end, interval=interval)
        df.to_csv(symbol_to_path(symbol))
~~~

We will input a list of symbols, a start date, and an end date, and the function will
scrape the data from Yahoo! Finance and save it in our base directory. The code:

~~~python
if isinstance(symbols, str): 
    symbols = [symbols] # if string is passed in, place it in a list
~~~

is just used to put the symbol into a list if you happened to have passed in a single
string. Also, notice that the output of the `get_data_yahoo` is actually a pandas 
dataframe, which we then save as a comma separated value file using `to_csv`. Cool, 
now we can get data for our favorite stocks!

~~~python
symbols = ['AAPL', 'SPY', 'IBM', 'GOOG']
start_date = '2016-01-01' # dates should be in YYYY-MM-DD format, or a datetime object.
end_date = '2016-12-31'
scrape_stock_data(symbols, start_date, end_date)
~~~

Now we have a directory with stock data in it, so let's try to do some analysis on our data.
First, lets print it to get an idea of what the data looks like:

~~~python
df = pd.read_csv(symbol_to_path('GOOG'))
print(df.head())
~~~

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-04</td>
      <td>743.000000</td>
      <td>744.059998</td>
      <td>731.257996</td>
      <td>741.840027</td>
      <td>3272800</td>
      <td>741.840027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-05</td>
      <td>746.450012</td>
      <td>752.000000</td>
      <td>738.640015</td>
      <td>742.580017</td>
      <td>1950700</td>
      <td>742.580017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-06</td>
      <td>730.000000</td>
      <td>747.179993</td>
      <td>728.919983</td>
      <td>743.619995</td>
      <td>1947000</td>
      <td>743.619995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-07</td>
      <td>730.309998</td>
      <td>738.500000</td>
      <td>719.059998</td>
      <td>726.390015</td>
      <td>2963700</td>
      <td>726.390015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-08</td>
      <td>731.450012</td>
      <td>733.229980</td>
      <td>713.000000</td>
      <td>714.469971</td>
      <td>2450900</td>
      <td>714.469971</td>
    </tr>
  </tbody>
</table>

The dataframe has seven columns, all of which tell us some interesting things about the
stock, however we are going to only focus on two of them:

1. Date: the day the stock was traded at
2. Adj Close: the value of the stock at close, adjusted for dividends and stock splits to 
allow for comparability across time.

There are a couple things we have to take into consideration. Some stocks aren't traded
everyday, and some might not even have existed during a given time period. It would be
useful to have a stock we know is active which can be used as a measure of the market.
We will use SPY, which is an ETF traded on the NYSE which follows the S&P 500. Lets make
a function which adds in SPY, drops NA values and returns a dataframe of Adjusted Close
for each of our stocks with the date as the index.

~~~python
def get_adj_close(symbols, start, end):
    """Read stock data (adjusted close) for given symbols from CSV files.""" 
    if isinstance(symbols, str): 
        symbols = [symbols] # if string is passed in, place it in a list
        
    dates = pd.date_range(start, end).date
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", 
                              parse_dates=True, usecols=['Date','Adj Close'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp) # use default how='left' 
        
    df = df.dropna()
    return df
~~~ 

~~~python
df = get_adj_close(symbols, start_date, end_date)
print(df.head())
~~~

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>SPY</th>
      <th>IBM</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>103.057063</td>
      <td>196.794026</td>
      <td>129.932320</td>
      <td>741.840027</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>100.474523</td>
      <td>197.126874</td>
      <td>129.836755</td>
      <td>742.580017</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>98.508268</td>
      <td>194.640278</td>
      <td>129.186847</td>
      <td>743.619995</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>94.350769</td>
      <td>189.970552</td>
      <td>126.979099</td>
      <td>726.390015</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>94.849671</td>
      <td>187.885326</td>
      <td>125.803548</td>
      <td>714.469971</td>
    </tr>
  </tbody>
</table>

Okay, enough looking at tables. Let's actually plot the data! To do this we will use
matplotlib and seaborn.

~~~python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df, title="Adjusted Close", xlabel="Date", ylabel="Price", size=(12,9)):
    """Plot Stock Data"""
    with sns.plotting_context(font_scale=1.5, rc={"figure.figsize": size}), sns.axes_style("ticks"):       
        ax = df.plot(title=title)
        sns.despine() # remove top and right borders
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
~~~

The code:

~~~python
with sns.plotting_context(font_scale=1.5, rc={"figure.figsize": size}), sns.axes_style("ticks"): 
~~~
adjusts the font size, the figure size, and the plot style without changing the default
settings in your main program. The `axes_style` method has several different style options
which change how the figure looks. The `despine` method removes the borders.

~~~python
plot_data(df)
~~~

<img src="/assets/images/trading_1/plot_1.png"/>

Looking good! However, Google's stock price is so much higher than the other stock prices
so it makes them hard to compare. One way we can better compare them is by normalizing the
starting price to 1.

~~~python
def normalize_data(df):
    """Normalize stock prices using the first row of the datafame."""
    return df/df.ix[0,:]
~~~

~~~python
df_norm = normalize_data(df)
plot_data(df_norm, title="Normalized Adjusted Close")
~~~

<img src="/assets/images/trading_1/plot_2.png"/>

There we go! We can get some more information about the data by calculating some statistics. 
We can either use individual `mean` and `std` pandas
methods to get the mean and standard deviation of the time series, or we can use the 
`describe` method to get a fuller description.

~~~python
print("Mean\n" + "==================\n" + str(df.mean()))
print("Standard deviation\n" + "===================\n" + str(df.std()))
~~~
>~~~
Mean
==================
AAPL    103.597547
SPY     206.868675
IBM     146.922627
GOOG    743.486707
dtype: float64
Standard deviation
===================
AAPL     8.033139
SPY     11.098166
IBM     12.838304
GOOG    34.455758
dtype: float64
~~~

~~~python
print(df.describe())
~~~
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>SPY</th>
      <th>IBM</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>103.597547</td>
      <td>206.868675</td>
      <td>146.922627</td>
      <td>743.486707</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.033139</td>
      <td>11.098166</td>
      <td>12.838304</td>
      <td>34.455758</td>
    </tr>
    <tr>
      <th>min</th>
      <td>89.394274</td>
      <td>179.015794</td>
      <td>113.783991</td>
      <td>668.260010</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>95.808918</td>
      <td>201.203923</td>
      <td>142.010338</td>
      <td>713.242493</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>104.201492</td>
      <td>207.807558</td>
      <td>149.054243</td>
      <td>742.845001</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>110.293901</td>
      <td>214.902820</td>
      <td>156.587131</td>
      <td>772.640000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>117.645979</td>
      <td>226.425423</td>
      <td>167.188056</td>
      <td>813.109985</td>
    </tr>
  </tbody>
</table>

Some more interesting statistics are rolling statistcs such as the moving average. We will
compute the average over a rolling window of say 20 days. That is we average day 1 through 20, 
then we shift the window right by one and average day 2 through 21, and so on. This has the
effect of smoothing out the data, so that we can better see the trends. The larger the window,
the smoother the data gets.

~~~python
def get_rolling_mean(df, symbol, window=20):
    """Return rolling mean of given symbol, using specified window size."""
    return df[symbol].rolling(window=window).mean()

def get_rolling_std(df, symbol, window=20):
    """Return rolling standard deviation of given symbol, using specified window size."""
    return df[symbol].rolling(window=window).std()
~~~

With the rolling mean and rolling standard deviation, we can actually compute something called
[Bollinger Bands &reg;](https://en.wikipedia.org/wiki/Bollinger_Bands) which attempt to define
a relative high and low for the stock prices by considering two standard deviations away
from the mean in a rolling window. They are sometimes used as techinical indicators
for knowing when to buy and sell. Whenever the stock leaves and reenters the bands gives a
buy/sell signal. I don't recommend trying this however.

~~~python
def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    return upper_band, lower_band

def plot_bollinger_bands(df, symbol, window=20):
    """Plot Bollinger Bands(R) for a given symbol."""
    rm = get_rolling_mean(df, symbol, window=window)
    rstd = get_rolling_std(df, symbol, window=window)
    upper_band, lower_band = get_bollinger_bands(rm, rstd) 
    
    with sns.plotting_context(font_scale=1.5, rc={"figure.figsize": (12,9)}), sns.axes_style("ticks"):  
        ax = df[symbol].plot(title="Bollinger Bands", label=symbol)
        sns.despine() # remove top and right borders
        rm.plot(color='firebrick',label='Rolling mean', ax=ax)
        upper_band.plot(label='',color='gray', ax=ax)
        lower_band.plot(label='',color='gray', ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')
        plt.show()
~~~

~~~python
plot_bollinger_bands(df, 'AAPL')
~~~

<img src="/assets/images/trading_1/plot_3.png"/>

Now let's talk about daily returns. The daily return is the signed percentage change in
a stock price from day to day. We will be using the daily returns extensively when optimizing
our portfolios!

~~~python
def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0,:] = 0
        
    return daily_returns
~~~

That's it! Note that we have to set the first value to 0, since there is no previous value
to compare it to.

~~~python
daily_ret = compute_daily_returns(df)
plot_data(daily_ret, title="Daily Returns")
~~~

<img src="/assets/images/trading_1/plot_4.png"/>

Okay, great, but that is a bit noisy. Let's just compare two stocks.

~~~python
plot_data(daily_ret[['SPY', 'IBM']], title="Daily Returns")
~~~

<img src="/assets/images/trading_1/plot_5.png"/>

One more statistic to check is the cumulative return. The cumulative return considers the signed percent change
in stock price for an entire interval as opposed to a relative to the previous day. Generally,
we care about the cumulative return over the entire period of interest.

~~~python
def compute_cum_returns(df):
    """Compute and return the cumulative returns."""
    cum_returns = normalize_data(df) - 1
    return cum_returns
~~~

~~~python
cum_ret = compute_cum_returns(df)
print(cum_ret.ix[-1])
plot_data(cum_ret, title="Cumulative Returns")
~~~

>~~~
AAPL    0.123843
SPY     0.135858
IBM     0.267489
GOOG    0.040413
Name: 2016-12-30 00:00:00, dtype: float64
~~~

<img src="/assets/images/trading_1/plot_6.png"/>

We may also want to consider the distribution of daily returns without regard to time, as 
a way of visualizing how the daily return fluctuates. We may also want to compare daily returns
of more than one stock. To do this we can create histograms and scatter plots. First, let's
make a histogram function which can plot multiple stocks. In the following code, whenever
only one stock is passed in, the mean, standard deviation, and kurtosis is calculated.
[Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) is a measure of how fat the tails of the 
distribution are. For comparison, the kurtosis of the normal distribution is 3.

~~~python
def plot_hist(df, symbols, bins=20, title="Distribution of daily return", xlabel="Daily return", 
              ylabel="", size=(12,9), plot_stat=True):
    """Plot Histogram"""
    if isinstance(symbols, str): 
        symbols = [symbols] # if string is passed in, place it in a list 
    if len(symbols) != 1:  # don't plot stats if there is more than one histogram
        plot_stat=False
    
    with sns.plotting_context(font_scale=1.5, rc={"figure.figsize": size}), sns.axes_style("ticks"):    
        for symbol in symbols:
            mean = df[symbol].mean()
            std = df[symbol].std()
            kurtosis = df[symbol].kurtosis()
            df[symbol].hist(bins=bins, label=symbol, alpha=0.7)
            if plot_stat:
                plt.axvline(mean, color='b', label='mean = ' + str(round(mean,3)), linestyle='dashed', linewidth=2)
                plt.axvline(mean+std, color='r',  label='stdev = ' + str(round(std,3)), linestyle='dashed', linewidth=2)
                plt.axvline(mean-std, color='r', label='', linestyle='dashed', linewidth=2)
                plt.axvline(mean, visible=False, label='kurtosis = ' + str(round(kurtosis,3)), \
                            linestyle='dashed', linewidth=2) # hack to get kurtosis to show in legend
        sns.despine() # remove top and right borders
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
~~~

~~~python
plot_hist(daily_ret, 'IBM')
~~~

<img src="/assets/images/trading_1/plot_7.png"/>

~~~python
plot_hist(daily_ret, ['SPY', 'IBM'])
~~~

<img src="/assets/images/trading_1/plot_8.png"/>

The scatterplot function uses seaborn's `JointGrid` method to do a scatter plot,
`regplot` to plot a regression line, and scipy's stats module to annotate with the Pearson correlation coefficient.

~~~python
from scipy import stats
def plot_scatter(df, symbols, title="Daily return", size=(9,9)):
    """Takes a dataframe and two symbols and creates a scatter plot with linear regression line"""
    with sns.plotting_context(font_scale=1.5, rc={"figure.figsize": size}), sns.axes_style("ticks"): 
        x = df[symbols[0]]
        y = df[symbols[1]]
        g = sns.JointGrid(x, y, ratio=100)
        sns.despine() # remove top and right borders
        g.plot_joint(sns.regplot)
        g.annotate(stats.pearsonr)
        g.ax_marg_x.set_axis_off()
        g.ax_marg_y.set_axis_off()
        g.fig.set_figwidth(size[0])
        g.fig.suptitle(title)
        plt.show()
~~~

The correlation coefficient describes how closely the data 'bunches up' along the line.

~~~
plot_scatter(daily_ret, ['SPY', 'IBM'])
~~~

<img src="/assets/images/trading_1/plot_9.png"/>

Okay, finally we are done with all of our visualization functions. Let's now talk about
optimizing a portfolio. To do this we will maximize the [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio),
which measures the excess return per unit of deviation. The Sharpe ratio is sometimes referred
to as the reward-to-variability ratio. More formally it is the ratio of the expected portfolio rate of return
minus the risk-free rate of return divided by the standard deviation of portfolio rate of return minus the
risk-free rate of return. The risk-free rate return is sometimes taken as the LIBOR rate,
the rate of 3 month T bill or the rate of a benchmark such as SPY. The risk-free
rate is often approximated as 0 since the T bill rate has been near 0 for so long. However, 
this appears to be changing in the near future.

The Sharpe ratio depends on how frequently we sample our data, that is daily, weekly, monthly, yearly, etc.
We want to consider the Sharpe ratio is as an annual measure. In order to annualize it
we need to modify it by a factor dependent on how we sampled our data. We are currently sampling
our data daily, so we multiply the Sharpe ratio by a factor K=sqrt(252). The 252 comes from the
fact that there are on average 252 [trading days](https://en.wikipedia.org/wiki/Trading_day) a year.
The square root comes from the assumption that the volatility (standard deviation) of the stock
scales with the square root of time. See [this](http://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio)
stackexchange post.

~~~python
def compute_sharpe_ratio(daily_returns, annual_rf=0):
    """Computes Sharpe Ratio based on a daily approximation"""
    # annual_rf = 0 is an approximation of annual risk free rate
    daily_rf = ((1 + annual_rf) - 1)**(1/252)  # approximation based on interest rate and 252 trading days
    K = (252)**(1/2) # assumes daily data
    sharpe_ratio = K *(daily_returns - daily_rf).mean() / (daily_returns - daily_rf).std()
    return sharpe_ratio
~~~

~~~python
sharpe_ratio = compute_sharpe_ratio(daily_ret)
print(sharpe_ratio)
~~~

>~~~
AAPL    0.616840
SPY     1.046393
IBM     1.301543
GOOG    0.298970
dtype: float64
~~~ 

Cool! Now for a given allocation of funds into a set of stocks we can calculate the value
of a portfolio by first normalizing the data:

~~~python
port_symbols = ['AAPL', 'IBM', 'GOOG']
df_norm = normalize_data(df_norm[port_symbols])
print(df_norm.head().to_html())
~~~

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>IBM</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>0.974941</td>
      <td>0.999265</td>
      <td>1.000998</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>0.955861</td>
      <td>0.994263</td>
      <td>1.002399</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>0.915520</td>
      <td>0.977271</td>
      <td>0.979173</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>0.920361</td>
      <td>0.968224</td>
      <td>0.963105</td>
    </tr>
  </tbody>
</table>

multiplying the data by our initial allocations: 

~~~python
start_value = 1000000 # initial amount of money to invest
allocations = [0.5, 0.3, 0.2] # must add to 1
position_value = df_norm * allocations * start_value
print(position_value.head())
~~~

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>IBM</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>500000.000000</td>
      <td>300000.000000</td>
      <td>200000.000000</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>487470.339612</td>
      <td>299779.350511</td>
      <td>200199.501233</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>477930.697482</td>
      <td>298278.781600</td>
      <td>200479.879202</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>457759.838353</td>
      <td>293181.324708</td>
      <td>195834.678249</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>460180.351734</td>
      <td>290467.101642</td>
      <td>192621.035532</td>
    </tr>
  </tbody>
</table>

and then summing up the value of all the stocks at each time point: 

~~~python
port_value = position_value.sum(axis=1)
print(port_value.head())
~~~

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>987449.191356</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>976689.358283</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>946775.841310</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>943268.488908</td>
    </tr>
  </tbody>
</table>

Now lets put all of these statistics into a portfolio object; the stock values and symbols, allocations, value at
position, the portfolio value, the daily returns, cumulative returns, average daily return,
standard deviation of daily returns and the Sharpe ratio.

~~~python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Portfolio:
    """Keeps portfolio and portfolio statistics"""
    def __init__(self, df, symbols, start_value, allocations):
        self.df = df
        self.symbols = symbols
        self.start_value = start_value
        self.update(allocations)
     
    def update(self, allocations):
        """Update portfolio given new allocations"""
        self.allocations = allocations
        self.position_values = normalize_data(self.df[self.symbols]) * self.allocations * self.start_value
        self.value = self.position_values.sum(axis=1).to_frame(name='Value')
        self.daily_returns = compute_daily_returns(self.value)
        self.cum_returns = compute_cum_returns(self.value)
        self.final_cum_return = self.cum_returns.ix[-1]
        self.avg_daily_return = self.daily_returns[1:].mean() # ignore first 0 element for computing stats
        self.std_daily_return = self.daily_returns[1:].std()
        self.sharpe_ratio = compute_sharpe_ratio(self.daily_returns[1:])

    def plot_vs_spy(self):
        """Plot daily portfolio value against SPY value"""
        # First concatenate the portfolio value with the SPY value and normalize
        comparison_df = normalize_data(pd.concat([self.value, self.df['SPY']], axis=1))
        comparison_df = comparison_df.rename(columns = {'Value':'Portfolio'})
        plot_data(comparison_df, title="Daily portfolio value and SPY", xlabel="Date", ylabel="Price", size=(12,9))
~~~

Of particular interest is the Sharpe ratio and the final cumulative return of our portfolio:

~~~python
port_symbols = ['AAPL', 'IBM', 'GOOG']
start_value = 1000000 # initial amount of money to invest
allocations = [0.5, 0.3, 0.2] # must add to 1
pt = Portfolio(df, port_symbols, start_value, allocations)
print("Sharpe Ratio\n" + "==================\n" + str(pt.sharpe_ratio))
print("Final Cumulative Return\n" + "===================\n" + str(pt.sharpe_ratio))
~~~

>~~~
Sharpe Ratio
==================
Value    0.915735
dtype: float64
Final Cumulative Return
===================
Value    0.915735
dtype: float64
~~~

We can use the `update` method to update the portfolio to new initial allocations. I also
added in a method to plot the portfolio value versus the SPY value in order to compare
our portfolio to the "market":

~~~python
new_alloc = [0.1, 0.1, 0.8]
pt.update(new_alloc)
print("Sharpe Ratio\n" + "==================\n" + str(pt.sharpe_ratio))
print("Final Cumulative Return\n" + "===================\n" + str(pt.sharpe_ratio))
pt.plot_vs_spy()
~~~

>~~~
harpe Ratio
==================
Value    0.47621
dtype: float64
Final Cumulative Return
===================
Value    0.47621
dtype: float64
~~~

<img src="/assets/images/trading_1/plot_10.png"/>

Finally, we can optimize our portfolio! The easiest way is to maximize the cumulative return.
To do this, we just check we stock has the greatest cumulative return and then put all of our
money into that stock. That's kind of boring and not very insightful. Another way is to maximize
the Sharpe ratio. Doing this will optimize our reward compared to risk. To do this we will
use scipy's `minimize` method as most optimization routines only consider minimization problems.
We must first formulate our problem into something scipy can understand. We can think of it 
as minimizing the negative of the Sharpe ratio (hence maximizing it) subject to the constraints
that the initial allocations to each stock most be between 0 and 1, and also must sum to 1.

~~~python
    def opt_sharpe_ratio(self, allocations):
        """Update portfolio and return the negative of the sharpe_ratio for optimization purposes"""
        self.update(allocations)
        return -self.sharpe_ratio
        
    def optimize(self):
        """Optimize portfolio based on Sharpe Ratio"""
        # optimize for negative sharpe ratio
        n = len(self.df[self.symbols].columns) # number of stocks
        allocations = np.ones(n)*(1.0/n) # give each stock an equal starting allocation  
        constraints = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})  # allocations must sum to 1   
        bounds = tuple((0,1) for x in allocations) # Required to have values between 0 and 1
        opt_stats = minimize(self.opt_sharpe_ratio, allocations, method='SLSQP', bounds=bounds ,constraints=constraints)
        self.update(opt_stats.x) # x is the container for the optimized allocations
~~~

And the entire class:

~~~python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Portfolio:
    """Keeps portfolio and portfolio statistics"""
    def __init__(self, df, symbols, start_value, allocations):
        self.df = df
        self.symbols = symbols
        self.start_value = start_value
        self.update(allocations)
     
    def update(self, allocations):
        """Update portfolio given new allocations"""
        self.allocations = allocations
        self.position_values = normalize_data(self.df[self.symbols]) * self.allocations * self.start_value
        self.value = self.position_values.sum(axis=1).to_frame(name='Value')
        self.daily_returns = compute_daily_returns(self.value)
        self.cum_returns = compute_cum_returns(self.value)
        self.final_cum_return = self.cum_returns.ix[-1]
        self.avg_daily_return = self.daily_returns[1:].mean() # ignore first 0 element for computing stats
        self.std_daily_return = self.daily_returns[1:].std()
        self.sharpe_ratio = compute_sharpe_ratio(self.daily_returns[1:])
    
    def opt_sharpe_ratio(self, allocations):
        """Update portfolio and return the negative of the sharpe_ratio for optimization purposes"""
        self.update(allocations)
        return -self.sharpe_ratio
        
    def optimize(self):
        """Optimize portfolio based on Sharpe Ratio"""
        # optimize for negative sharpe ratio
        n = len(self.df[self.symbols].columns) # number of stocks
        allocations = np.ones(n)*(1.0/n) # give each stock an equal starting allocation  
        constraints = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})  # allocations must sum to 1   
        bounds = tuple((0,1) for x in allocations) # Required to have values between 0 and 1
        opt_stats = minimize(self.opt_sharpe_ratio, allocations, method='SLSQP', bounds=bounds ,constraints=constraints)
        self.update(opt_stats.x) # x is the container for the optimized allocations
    
    def get_stats(self):
        """Print out portfolio statistics"""
        # np.datetime_as_string(self.df.index.values[0], timezone='local')[:10] is a hack to get the date      
        print("Start Date: " + np.datetime_as_string(self.df.index.values[0], timezone='local')[:10]) 
        print("End Date: " + np.datetime_as_string(self.df.index.values[-1], timezone='local')[:10])
        print("Symbols: " + str(self.symbols))
        print("Allocations: " + str(self.allocations))
        print("Sharpe Ratio: " + str(self.sharpe_ratio.values[0]))
        print("Volatility (stdev of daily returns): " + str(self.std_daily_return.values[0]))
        print("Average Daily Return: " + str(self.avg_daily_return.values[0]))
        print("Cumulative Return: " + str(self.final_cum_return.values[0]))
        
    def plot_vs_spy(self):
        """Plot daily portfolio value against SPY value"""
        # First concatenate the portfolio value with the SPY value and normalize
        comparison_df = normalize_data(pd.concat([self.value, self.df['SPY']], axis=1))
        comparison_df = comparison_df.rename(columns = {'Value':'Portfolio'})
        plot_data(comparison_df, title="Daily portfolio value and SPY", xlabel="Date", ylabel="Price", size=(12,9))
        
    def analysis(self):
        """Output analysis of portfolio!"""
        self.get_stats()
        self.plot_vs_spy()
~~~

Now we can optimize our portfolio:

~~~python
port_symbols = ['AAPL', 'IBM', 'GOOG']
start_value = 1000000 # initial amount of money to invest
allocations = [0.5, 0.3, 0.2] # must add to 1
pt = Portfolio(df, port_symbols, start_value, allocations)
print(pt.final_cum_return)
~~~

>~~~
Value    0.150251
Name: 2016-12-30 00:00:00, dtype: float64
~~~

~~~python
pt.optimize()
pt.analysis()
~~~

>~~~
Start Date: 2016-01-03
End Date: 2016-12-29
Symbols: ['AAPL', 'IBM', 'GOOG']
Allocations: [  6.17693491e-02   9.38230651e-01   1.04083409e-17]
Sharpe Ratio: 1.30647357752
Volatility (stdev of daily returns): 0.012018384716
Average Daily Return: 0.000989114258415
Cumulative Return: 0.258616463303
~~~

<img src="/assets/images/trading_1/plot_11.png"/>

It looks like we beat the market! Congratulations! Of course, this is easy since hindsight
is 20/20. We will look to use machine learning to come up with strategies to optimize future
gains in a later tutorial. Good luck with your studies!

