import numpy as np
import pandas as pd
import quandl 
quandl.ApiConfig.api_key = "m7NajB6MzkwkvdTm97zn"

from backtest import Strategy, Portfolio

class RandomForecastingStrategy(Strategy):
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
    
    def generate_signals(self):
        signals = pd.DataFrame(index = self.bars.index)
        signals['signal'] = np.sign(np.random.randn(len(signals)))
        
        signals['signal'][0:5] = 0.0
        return signals

class MarketOnOpenPortfolio(Portfolio):
    """requires:
    symbol - A stock symbol
    bars   - A DataFrame of bars
    signals - A pandas DataFrame of signals (1, 0, -1)
    initial capital"""
    
    def __init__(self, symbol, bars, signals, initial_capital = 100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions[self.symbol] = 500 * signals['signal']
        return positions
        
    def backtest_portfolio(self):
        portfolio = self.positions * self.bars['Open']
        pos_diff = self.positions.diff()
        
        portfolio['holdings'] = (self.positions * self.bars['Open']).sum(axis = 1)
        portfolio['cash'] = self.initial_capital - (pos_diff * self.bars['Open']).sum(axis = 1).cumsum()
        
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio
        
if __name__ == "__main__":
    symbol = 'SPY'
    bars = quandl.get("GOOG/NYSE_%s" % symbol, collapse="daily")
    # comment: have no access to the quandl data. 
    
    rfs = RandomForecastingStrategy(symbol, bars)
    signals = rfs.generate_signals()
    
    portfolio = MarketOnOpenPortfolio(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    
    print returns.tail(20)