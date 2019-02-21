import datetime
import pandas as pd 
import pandas_datareader.data as web
import urllib2 

def readDataFromWeb(assets, startDate, endDate, source = 'yahoo'):
    
    PROXIES = {'https': "https://proxy.companyname.net:8080"}
    Proxy   = urllib2.ProxyHandler(PROXIES)
    Opener  = urllib2.build_opener(proxy)
    
    urllib2.install_opener(opener)
    prices  = {}
    volumes = {}
    for asset in assets:
        try:
            df              = web.DataReader(assets, source, start=startDate, end=endDate)
            prices[assets]  = df['Adj Close']
            volumes[assets] = df['Volume']
        except:
            print "Error: skipping", asset
    prices  = pd.DataFrame(prices)
    volumes = pd.DataFrame(volumes)
    return pd.Panel({'Price' : prices, 'Return' : prices.pct_chang(), 'Volume':volumes})

def main():
    start     = datetime.date(2016,12,20)
    end       = datetime.date.today() - datetime.timedelta(1)
    Assetlist = ['YHOO', 'AAPL', 'IBM','F']
    Data      = readDataFromWeb(Assetlist,start,end)
    
    Data.Price