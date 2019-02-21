#pip install PyMySQL
#pip install schedule

import pandas as pd, numpy as np, datetime
import pymysql.cursors
import os
import time
import schedule

def job(frequency, period):
    
    df = pd.read_csv('C:/python/S&amp;P 500.csv')
    tickers = df['Ticker'].values.tolist()
    
    for ticker in tickers:
        try:
            for ch in ['/', '+', '-', '^' ]:
                if ch in ticker:
                    ticker=ticker.replace(ch, ".")
            url ='https://www.bloomberg.com/markets2/api/intraday/AAPL%3AUS?days=1&interval=1&volumeInterval=15&currency=USD'
            x = np.array(pd.read_csv(url,skiprows=7,header=None, error_bad_lines=False))
            
            date = []
            symbol = []
            
            for i in range(0, len(x)):
                if x[i][0][0] == 'a':
                    t= datetime.datetime.fromtimestamp(int(x[i][0].replace('a', ' ')))
                    date.append(t)
                else:
                    date.append(t+datetime.timedelta(minutes = int(x[i][0])))
               
            for i in range(0, len(x)):
                symbol.append(ticker)
            
            df = pd.DataFrame(x)
            
            se1 = pd.series(symbol)
            se2 = pd.series(date)
            df['Symbol'] = se1.values
            df['Date'] = se2.values
            
            df.columns = ['a', 'Open', 'High', 'Close', 'Vol', 'Symbol', 'Date']
            df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Vol']]
            
            if not os.path.exists('S&amp;P 500 Intraday Data'):
                os.makedirs('S&amp;P 500 Intraday Data')
            
            ticker_path = os.path.join("C:/python/S&amp;P 500 Intraday Data/", ticker)
            if not os.path.exists(ticker_path):
                os.makedirs(ticker_path)
                
            storage_path = os.path.join("C:/python/S&amp;P 500 Intraday Data/", ticker, ticker+".csv")
            if not os.path.exists(storage_path):
                df.to_csv(storage_path, index=False)
                print ticker + ':Stored on Disk Space'
            
            else:
                os.remove(storage_path)
                df.to_csv(storage_path, index=False)
                print ticker + 'Stored on Disk Space'
        
        except Exception as e:
            print str(e)

            
def store_data():
    df = pd.read_csv("C:/python/Ticker Files/Index Tickers/S&amp;P 500.csv")
    tickers = df['Ticker'].values.tolist()
    
    for ticker in tickers:
        read_path = os.path.join("C:/python/S&amp;P 50 Intraday Data/", ticker, tick+",csv")
        
        f = open(read_path, "r")
        fstring = f.read()
        
        fList = []
        for line in fstring.split('\n'):
            fList.append(line.split(','))
            
        connectionObject = pymysql.connect(host='localhost',
                                    user = 'root',
                                    port = 3306,
                                    password = '',
                                    db = 'intraday_data')
        cursorObject = connectionObject.cursor()
        
        DATE = fList[0][0]; SYMBOL = fList[0][1]; OPEN = fList[0][2]; HIGH = fList[0][3]; LOW = fList[0][4]; CLOSE = fList[0][5]; VOLUME = fList[0][6]
        
        queryTable = """CREATE TABLE IF NOT EXISTS sp500(
                        {} DATETIME NOT NULL,
                        {} VARCHAR(6) NOT NULL,
                        {} DECIMAL(18, 4),
                        {} DECIMAL(18, 4),
                        {} DECIMAL(18, 4),
                        {} DECIMAL(18, 4),
                        {} INT)""".format(DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, VOLUME)
                        
        cursorObject.execute(queryTable)
        
        del fList[0]
        
        rows = ''
        for i in range(len(fList)-1):
            rows += "('{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(fList[i][0], fList[i][1], fList[i][2], fList[i][3], fList[i][4], fList[i][5], fList[i][6])
            
            if i!= len(fList)-2:
                rows += ','
        
        queryInsert = "INSERT INTO sp500 VALUES" + rows
        
        try:
            cursorObject.execute(queryInsert)
            connectionObject.commit()
        except:
            print 'Error'
        connectionObject.close()
        
        print ticker+ ': Is Stored!'
        
    return store_data()
    
schedule.every().monday.at("16:30").do(job, frequency = '60', period = '1')
schedule.every().tuesday.at("16:30").do(job, frequency = '60', period = '1')
schedule.every().wednesday.at("16:30").do(job, frequency = '60', period = '1')
schedule.every().thursday.at("16:30").do(job, frequency = '60', period = '1')
schedule.every().friday.at("16:30").do(job, frequency = '60', period = '1')

while True:
    schedule.run_pending()
    time.sleep(60)