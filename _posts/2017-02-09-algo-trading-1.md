---
layout: post
title: "Yet Another Algorithmic Trading Tutorial (Part I)"
description: "We use Python to visualize stock market data and optimize a portfolio based on historical data."
comments: true
keywords: "finance, trading, stocks, python, portfolio"
---

I am going to start off with a small digression. I am a big fan of continued education
from online sources. My favorite places too look are [Standford SEE](https://see.stanford.edu/),
[MIT OCW](https://ocw.mit.edu/), [Harvard OL](https://www.extension.harvard.edu/open-learning-initiative), 
[coursera](https://www.coursera.org/) and [edX](https://www.edx.org/). If you want a 
referesher on undergraduate probability theory, the best course I have found is 
[STAT110](http://projects.iq.harvard.edu/stat110/home), a course by 
Professor Joe Blitzstein at Harvard. He also helps run a data science course, 
[CS109](http://cs109.github.io/2015/index.html), which gives a nice introduction in using
machine learning with Python. 

This tutorial actually follows, [Machine Learning for Trading](https://classroom.udacity.com/courses/ud501/), a 
course on Udacity (not my favorite, sorry Udacity!), which you can take yourself and follow along. The course is by 
Dr. Tucker Balch at Georgia Tech, but sadly doesn't actually link to any of the homework 
files. However, you can find a short description of them
[here](http://quantsoftware.gatech.edu/Machine_Learning_for_Trading_Course). This 
description will be what we use to develop our tools for this tutorial. Okay, let's get started!

<div class="divider"></div>

**DISCLAIMER!** I am not a financial professional, so do not use this tutorial to attempt
to trade stocks without EXTREME caution!

<div class="divider"></div>

I will be using Python 3 to do this analysis. You can find the associated python files
here (TODO). You can check your Python libraries against mine using:

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

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print("Scikit-Learn version:      %6.6s (my version: 0.18.1)" % sklearn.__version__)

# Seaborn makes matplotlib look good and provides some plotting functions
import seaborn
print("Seaborn version:          %6.6s  (my version: 0.7.1)" % seaborn.__version__)
~~~
~~~
Python version:            3.5.2  (my version: 3.5.2)
Numpy version:             1.12.0 (my version: 1.12.0)
SciPy version:             0.18.1 (my version: 0.18.1)
Pandas version:            0.19.2 (my version: 0.19.2)
Pandas_datareader version: 0.3.0. (my version: 0.3.0)
Mapltolib version:         2.0.0  (my version: 2.0.0)
Scikit-Learn version:      0.18.1 (my version: 0.18.1)
Seaborn version:           0.7.1  (my version: 0.7.1)
~~~

As you can see we will be using numpy, scipy, pandas, matplotlib, scikit-learn and seaborn.
Even if you have pandas, you might not have [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/),
so you may want to install that first. We will also be using [seaborn](http://seaborn.pydata.org/)
which makes it much easier (at least for me) to make nice looking graphs. If you are unfamiliar
with pandas, you may want to take some time to figure out how it works. If you've used 
R, then pandas is basic functionality for dataframes.

Okay, so pandas_datareader will make our work much easier, since we won't have to deal with
wondering how to get our hands on stock data. In particular, we will use the `get_data_yahoo()`
function download historical stock market data from Yahoo! Finance. In order to use this function,
we need to know the ticker symbol for the stock we are interested in (e.g. 'GOOG' for Google),
the start date, and the unit of time between individual records (here we will be using days).

We probably also want to save the data somewhere, so lets set up a function to get the
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
string. Also, notice that the output of the `get_data_yahoo()` is actually a pandas 
dataframe, which we then save as a comma separated value file using `.to_csv()`. Cool, 
now we can get data for our favorite stocks!

~~~python
symbols = ['AAPL', 'SPY', 'IBM', 'GOOG']
start_date = '2016-01-01' # dates should be in YYYY-MM-DD format, or a datetime object.
end_date = '2016-12-31'
scrape_stock_data(symbols, start_date, end_date)
~~~

Now we have a directory with stock data in it, so let's try to do some analysis on our data.

(I'm just starting this blog, so if you are reading this, just know this is for testing purposes, and 
I haven't finished this!)
