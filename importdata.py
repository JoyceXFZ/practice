#import data from excel to python
#https://www.datacamp.com/community/tutorials/python-excel-tutorial

import pandas as pd
import numpy as np

#assign spreadsheet filename to 'file'
file = 'mydata.xls'

#load spreadsheet
x1 = pd.ExcelFile(file)

#print the sheet names
print(x1.sheet_names)

#load a sheet into a DataFrame 
df1 = x1.parse('Sheet1')

# replace all nan with ''
df = df1.replace(np.nan, '', regex = True)
print(df)

#If load data from csv
# df = pd.read_csv('example.csv')
########################################

# write data into excel 
#pip install XlsxWriter

#Specify a writer
#writer = pd.ExcelWriter('data1.xlsx', engine = 'xlsxwriter')

#write DataFrame to a file, using the to_excel() function 
#df1.to_excel(writer, 'Sheet1')

#save the result
#writer.save()
