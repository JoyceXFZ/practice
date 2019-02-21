#pip install PyMySQL
#pip install schedule
#pip install pyreadline

import pandas as pd, numpy as np, datetime
import pymysql.cursors
import os
import time
import schedule
import csv
import urllib
from urllib.request import urlopen



url ="https://www.bloomberg.com/markets2/api/intraday/AAPL%3AUS?days=1&interval=1&volumeInterval=15&currency=USD"

response = urllib2.urlopen(url)
cr = csv.read(response)

print cr
