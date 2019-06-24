#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression with Python
# 
# For this lecture we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set and very often is a student's first step in machine learning! 
# 
# We'll be trying to predict a classification- survival or deceased.
# Let's begin our understanding of implementing Logistic Regression in Python for classification.
# 
# We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
# 
# ## Import Libraries
# Let's import some libraries to get started!

# In[13]:


import pandas as pd
import numpy as np
import datetime
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[14]:


#train_1958
path = input('Please type in the path of your data folder:')


# In[15]:


path


# In[16]:


# this function reads a csv file and process it by 
# 1. removing the trash
# 2. get date into the same format
# 3. get time into the same format
# 4. fix the wind speed (change into string)
# input: filename str ---eg.'2011-2018ord.csv'
# output: pandas dataframe
def readfile(filename):
    
    # performing task 1
    trash_offset = 25
    trash_index = 0
    train = pd.read_csv(filename, skiprows= range(0,8) )
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
    nrows = train.shape[0]
    #print(nrows)
    for x in range(nrows-trash_offset,nrows):
        if type(train.loc[x]['Time']) != str:
            trash_index = x
            break
    train.drop(range(trash_index,nrows), inplace = True)
   
    # performing task 2
    # check if the date data is in the right form
    date_pattern = re.compile(r'\d\d\d\d-\d\d-\d\d')
    searchObj = re.search(date_pattern, train['Date'][0])
    if not searchObj:
        nrows = train.shape[0]
        for x in range(0,nrows):
            train.at[x,'Date'] = datetime.datetime.strptime(train.at[x,'Date'], "%m/%d/%Y").strftime("%Y-%m-%d")

    # performing task 3
    # check if time data is in the right form
    time_pattern = re.compile(r'^\d:\d\d')
    searchObj = re.search(time_pattern, train['Time'][0])
    if searchObj:
        nrows = train.shape[0]
        for x in range(0,nrows):
            # task 3
            searchObj = re.search(time_pattern, train['Time'][x])
            if searchObj:
                train.at[x,'Time'] = '0' + train.at[x,'Time']  
                
    # performing task 4
    train = train.astype({train.columns[4]:'str'})
    return train


# In[17]:


#train_test = readfile('1991-2000ord.csv')
#train_temp = readfile('1971-1980ord.csv')
#train_test


# In[18]:


# this function takes in a date and calculate the mean min max for the features
# input: date -- string in the form of 'yyyy-mm-dd' eg:'1958-11-01'
#        train -- the main datafram to analyze
# output-- list containing:
#        mean_result -- datafram for mean of all the features 
#        min_result -- datafram of min of all the features 
#        max_result -- datafram of max of all the features 
#        invalid_feature -- 0 list of size 8, to be replaced with 1 to indicate invalid feature
def analyze_by_day(date, train):
    
    #test = '1958-11-01'
    train_found = train[train['Date'] == date]
    #print(train_found)
    invalid_feature = np.zeros(8)

    #train_found.shape[0]
    # out of the 8 features
    for y in range(2,train_found.shape[1]):
        # calculate how many 'm' there are for each feature out of 24 days
        m_count = 0
        for x in range(0, train_found.shape[0]):
            # count the number of 'm'
            if train_found.iloc[x,y].lower() == 'm':
                m_count += 1
        # if there are total of 6 or more 'm' make this feature invalid
        if m_count >= 6:
            invalid_feature[y-2] = 1
        m_count = 0
    #print(invalid_feature) 
    # now we have which feature is invalid, calculate mean etc for each feature
    df2 = train_found.drop(columns =['Date','Time'])
    df1 = df2.apply(pd.to_numeric, errors='coerce')

    for x in range(0,8):
        df1[df1.columns[x]].fillna(value=df1[df1.columns[x]].mean(), inplace = True)

    mean_result = df1.mean()
    min_result = df1.min()
    max_result = df1.max()
    #print(invalid_feature)
    #print(mean_result)
    #based on the invalid array, assign the final result list
    for x in range(0,8):
        if invalid_feature[x] == 1:
            mean_result[x] = float('nan')
            min_result[x] = float('nan')
            max_result[x] = float('nan')
        
    return mean_result,min_result,max_result,invalid_feature
            


# In[19]:


#mean_result = pd.DataFrame()
#max_result = pd.DataFrame()
#min_result = pd.DataFrame()
#invalid_feature = np.zeros(8)
#temp = analyze_by_day('1958-11-01', train_temp)


# In[20]:


# read all the csv files 
listOfFiles = os.listdir(path)
file_pattern = re.compile(r'ord.csv')
train_temp = pd.DataFrame()
for x in range(0,len(listOfFiles)):
    searchObj = re.search(file_pattern, listOfFiles[x])
    if searchObj:
        print (listOfFiles[x] )
        train_temp = pd.concat([train_temp,readfile(listOfFiles[x])], axis = 0, ignore_index=True)
         


# In[21]:


# now that we have read all the files ask user to input a range
first_date = input("Please input the starting date as in yyyy-mm-dd: ")
d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
#print(d1)
second_date = input("Please input the ending date as in yyyy-mm-dd: ")
d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
delta = d2-d1

while delta.days <= 0:
    first_date = input("Please input a valid starting date as in yyyy-mm-dd: ")
    d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
    second_date = input("Please input a valid ending date as in yyyy-mm-dd: ")
    d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
    delta = d2-d1

mean_temp = []
min_temp =[]
max_temp = []
invalid_temp = []
if delta.days >0:
    for i in range(delta.days+1):
        temp_day = d1+datetime.timedelta(days=i)
        day_str = temp_day.strftime('%Y-%m-%d')
        temp = analyze_by_day(day_str, train_temp)
        
        mean_temp.append(temp[0])
        min_temp.append(temp[1])
        max_temp.append(temp[2])
        invalid_temp.append(temp[3])
# group them together
mean_df = pd.DataFrame(mean_temp)
min_df = pd.DataFrame(min_temp)
max_df = pd.DataFrame(max_temp)
invalid_df = pd.DataFrame(invalid_temp)

# calculate mean and other stuff
for x in range(0,8):
    mean_df[mean_df.columns[x]].fillna(value=mean_df[mean_df.columns[x]].mean(), inplace = True)
    min_df[min_df.columns[x]].fillna(value=min_df[min_df.columns[x]].mean(), inplace = True)
    max_df[max_df.columns[x]].fillna(value=mean_df[max_df.columns[x]].mean(), inplace = True)
    
mean_final = mean_df.mean()
min_final = min_df.min()
max_final = max_df.max()

print('The mean of the range is:')
print(mean_final)
print('\nThe min of the range is:')
print(min_final)
print('\nThe max of the range is:')
print(max_final)

cols = ['Temp', 'Dewpt', 'Wind Spd', 'Wind Direction', 'Peak Wind Gust', 'Atm Press', 'Sea Lev Press', 'Precip']
print('the number of invalid values in each features is:\n')
for i in range(0,8):
    print(cols[i]+':'+' ')
    print(invalid_df.sum()[i])


# In[52]:


# stores the output data
invalid_series = invalid_df.sum()
invalid_series.index = (list(mean_final.index))
df_output = pd.concat([mean_final, min_final, max_final, invalid_series], axis = 1)
df_output.columns = ['mean', 'min', 'max', 'No. of invalid']
df_output.to_csv('data_output.csv',encoding='utf-8',index=False)

