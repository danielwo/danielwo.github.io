Bollinger Bands &reg;](https://en.wikipedia.org/wiki/Bollinger_Bands) which attempt to define
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
plot_data(daily_ret)
~~~

<img src="/assets/images/trading_1/plot_4.png"/>

Okay, great, but that is a bit noisy. Let's just compare two stocks.

~~~python
plot_data(daily_ret[['SPY', 'IBM']])
~~~

<img src="/assets/images/trading_1/plot_5.png"/>

It is a little hard to tell, but they appear to be tracking eachother pretty well. 

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
plot_data(cum_ret)
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
