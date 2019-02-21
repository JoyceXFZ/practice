#risk constrained portfolio optimization

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodel.api as sm
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import Foundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.research import run_pipeline

start = "2009-01-01"
end = "2011-01-01"

def qtus_returns(start_date, end_date):
    pipe = Pipeline(
        columns={'Close': USEquityPricing.close.latest},
        screen = QTradableStocksUS()
    )
    stocks = run_pipeline(pipe, start_date, end_date)  
    unstacked_results = stocks.unstack()
    
    prices = (unstacked_results['Close'].fillna(method='ffill').fillna(method='bfill')
                  .dropna(axis=1,how='any').shift(periods=-1).dropna())  
    qus_returns = prices.pct_change()[1:]
    return qus_returns

R = qtus_returns(start, end)
assets = R.columns

def make_pipeline():
	market_cap = Foundamentals.shares_outstanding.latest * USEquityPricing.close.latest
	book_to_price = 1 / Foundamentals.pb_ratio.latest
	biggest = market_cap.top(500, mask=QTradableStocksUS())
	smallest = market_cap.bottom(500,mask=QTradableStocksUS())
	
	highpb = book_to_price.top(500, mask=QTradableStocksUS())
	lowpb  = book_to_price.bottom(500, mask=QTradableStocksUS())
	
	universe = biggest|smallest|highpb|lowpb
	
	pipe = Pipeline(
	columns = {
		'returns': Returns(window_length=2),
		'market_cap': market_cap,
		'book_to_price': book_to_price,
		'biggest': biggest,
		'smallest': smallest,
		'highpb': highpb,
		'lowpb': lowpb
		},
		screen=universe
	)
	return pipe
	
pipe = make_pipeline()
results = run_pipeline(pipe, start, end)
###########################################