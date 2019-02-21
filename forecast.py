#need to do in the commond window
#pip install fix_yahoo_finance

import datetime
import numpy as np
import sklearn

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

def create_lagged_series(symbol, start_date, end_date, lags=5):
    ts = pdr.get_data_yahoo(symbol, start_date - datetime.timedelta(days = 365), end_date)
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]
    
    for i in xrange(0, lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
        
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0
    
    for i,x in enumerate(tsret["Today"]):
        if(abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001
    
    for i in xrange(0, lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change() * 100.0

    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]
    
    return tsret
        

def fit_model(name, model, X_train, y_train, X_test, pred):
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)
    
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print "%s: %.3f" % (name, hit_rate)

if __name__ == "__main__":
    snpret = create_lagged_series("SPY", datetime.datetime(2001, 1, 10), datetime.datetime(2005, 12, 31), lags =5)
    
    X = snpret[["Lag1", "Lag2"]]
    y = snpret["Direction"]
    
    start_test = datetime.datetime(2005,1,1)
     
    X_train = X[X.index < start_test]
    X_test  = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test  = y[y.index >= start_test]
    
    pred = pd.DataFrame(index = y_test.index)
    pred["Actual"] = y_test
    
    print "Hit Rates: "
    models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
    for m in models:
        fit_model(m[0], m[1], X_train, y_train, X_test, pred)