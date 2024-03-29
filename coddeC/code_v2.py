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

# In[1]:


import pandas as pd
import numpy as np
import datetime
import re
import os
import multiprocessing


# ## The Data
#
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[2]:
def readfile_noh(filename):


#train_1958

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
    train = pd.read_csv(filename, skiprows= range(0,8), dtype = {'Temp ('+'F)':str, 'Dewpt ('+'F)':str, 'Wind Spd ('+'mph)':str, 'Wind Direction ('+'deg)':str, 'Peak Wind Gust('+'mph)':str, 'Atm Press ('+'hPa)':str, 'Sea Lev Press ('+'hPa)':str, 'Precip ('+'in)':str}  )
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


# In[ ]:


# this function takes in a date and calculate the mean min max for the features
# input: date -- string in the form of 'yyyy-mm-dd' eg:'1958-11-01'
#        train -- the main datafram to analyze
# output-- list containing:
#        mean_result -- datafram for mean of this feature
#        min_result -- datafram of min of this feature
#        max_result -- datafram of max of this feature
#        invalid_feature
def analyze_by_day(date, train):
    #initialize
    mean_result = float('nan')
    min_result = float('nan')
    max_result = float('nan')
    invalid_feature = 0
    #readin feature data

    train_found = train[train['Date'] == date]

    #print(train_found)

    #train_found.shape[0]
    # calculate how many 'm' there are for each feature out of 24 days
    m_count = 0
    for x in range(0, train_found.shape[0]):
            # count the number of 'm'
        if train_found.iloc[x,2].lower() == 'm':
            m_count += 1
    # if there are total of 6 or more 'm' make this feature invalid
    if m_count >= 6:
        invalid_feature = 1

    #print(invalid_feature)
    if invalid_feature != 1:
        # now we caculate the info from this legit feature
        df2 = train_found.drop(columns =['Date','Time'])
        df1 = df2.apply(pd.to_numeric, errors='coerce')
        df1.fillna(value=df1.mean(), inplace = True)

        mean_result = df1.mean()[0]
        min_result = df1.min()[0]
        max_result = df1.max()[0]



    return mean_result,min_result,max_result,invalid_feature



def analyze_by_feature(feature):
    mean_temp = []
    min_temp =[]
    max_temp = []
    invalid_temp = []
    train_feature = pd.read_csv(feature+'.csv', dtype = str)


    train_index = pd.unique(train_feature['Date'])
    train_index = list(train_index)
    #print(train_index)
    for i in range(len(train_index)):
        #print(feature)
        #print(train_index[i])
        temp = analyze_by_day(train_index[i], train_feature)

        mean_temp.append(temp[0])
        min_temp.append(temp[1])
        max_temp.append(temp[2])
        invalid_temp.append(temp[3])

        '''
        print('this is the ')
        print(i)
        print('\n')
        print(mean_temp)
        print('\n')
        print(min_temp)
        print('\n')
        print(max_temp)
        print('\n')
        print(invalid_temp)
        '''
    # group them together


    mean_df = pd.DataFrame(mean_temp)
    min_df = pd.DataFrame(min_temp)
    max_df = pd.DataFrame(max_temp)
    invalid_df = pd.DataFrame(invalid_temp)


    # calculate mean and other stuff
    mean_df.fillna(value=mean_df.mean(), inplace = True)
    min_df.fillna(value=min_df.mean(), inplace = True)
    max_df.fillna(value=mean_df.mean(), inplace = True)

    mean_final = mean_df.mean()
    min_final = min_df.min()
    max_final = max_df.max()
    invalid_final = invalid_df.sum()




    return mean_final[0],min_final[0],max_final[0],invalid_final[0]

if __name__ == '__main__':

    path = input('Please type in the path of your data folder:')
    path
    # read all the csv files

    listOfFiles = os.listdir(path)

    file_pattern_ord = re.compile(r'\d\d\d\dord.csv')
    file_pattern_ugn = re.compile(r'\d\d\d\dugn.csv')
    file_pattern_dugn = re.compile(r'\d\d\d\dugn.csv')

    train_temp_ord = pd.DataFrame()
    train_temp_ugn = pd.DataFrame()

    for x in range(0,len(listOfFiles)):
        searchObj_ord = re.search(file_pattern_ord, listOfFiles[x])
        searchObj_ugn = re.search(file_pattern_ugn, listOfFiles[x])
        if searchObj_ord:
            #print (listOfFiles[x] )
            train_temp_ord = pd.concat([train_temp_ord,readfile(path+'/'+listOfFiles[x])], axis = 0, ignore_index=True)

        if searchObj_ugn:
            train_temp_ugn = pd.concat([train_temp_ugn,readfile(path+'/'+listOfFiles[x])], axis = 0, ignore_index=True)

    #print(train_temp)
    # check ord time span
    ord_time = ['1958-11-01']
    for x in range(1959,2016):
        ord_time.append(str(x-1)+'-12-31')
        ord_time.append(str(x)+'-01-01')

    while file_selection == 'ord':
        first_date = input("From 1958-11-01 to 2018-12-30, please input a valid starting date as in yyyy-mm-dd: ")
        d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
        if d1 >=datetime.date(1958,11,1) and d1 <=datetime.date(2018,12,30):
            break

    while file_selection == 'ord':
        second_date = input("From 1958-11-02 to 2018-12-31, please input the ending date as in yyyy-mm-dd: ")
        d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
        if d2 >=datetime.date(1958,11,2) and d2 <=datetime.date(2018,12,31):
            break

    # check ugn time span
    while file_selection == 'ugn':
        first_date = input("From 1989-04-21 to 2018-12-30, please input a valid starting date as in yyyy-mm-dd: ")
        d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
        if d1 >=datetime.date(1989,4,21) and d1 <=datetime.date(2018,12,30):
            break

    while file_selection == 'ugn':
        second_date = input("From 1989-04-22 to 2018-12-31, please input the ending date as in yyyy-mm-dd: ")
        d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
        if d2 >=datetime.date(1989,4,22) and d2 <=datetime.date(2018,12,31):
            break

    delta = d2-d1

    while delta.days <= 0:
        print('Your starting date is later than your ending date, try again please')
        first_date = input("Please input a valid starting date as in yyyy-mm-dd: ")
        d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
        second_date = input("Please input a valid ending date as in yyyy-mm-dd: ")
        d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
        delta = d2-d1

    if delta.days >0:

        first_index_list = train_temp.index[train_temp['Date'] == first_date].tolist()
        second_index_list = train_temp.index[train_temp['Date'] == second_date].tolist()

        while(len(first_index_list) == 0):
            d1 = d1 + datetime.timedelta(days=1)
            first_date = d1.strftime('%Y-%m-%d')
            #print(second_date)
            first_index_list = train_temp.index[train_temp['Date'] == first_date].tolist()
        first_index = first_index_list[0]

        while(len(second_index_list) == 0):
            d2 = d2 - datetime.timedelta(days=1)
            second_date = d2.strftime('%Y-%m-%d')
            #print(second_date)
            second_index_list = train_temp.index[train_temp['Date'] == second_date].tolist()
        second_index = second_index_list[-1]

        if d1>d2:
            print('there is no data in your time span')
            exit()
        #print(second_index)

        else:
            train_temp = train_temp.iloc[first_index:second_index+1]


        #print(train_temp)
    #split the data into 8 different files
    file_col = ['temp','dewpt','windS','windD','peak','atm','sea','precip']
    train_1 = train_temp.iloc[:,[0,1,2]]
    train_1.to_csv(file_col[0]+'.csv',encoding = 'utf-8',index = False)
    train_2 = train_temp.iloc[:,[0,1,3]]
    train_2.to_csv(file_col[1]+'.csv',encoding = 'utf-8',index = False)
    train_3 = train_temp.iloc[:,[0,1,4]]
    train_3.to_csv(file_col[2]+'.csv',encoding = 'utf-8',index = False)
    train_4 = train_temp.iloc[:,[0,1,5]]
    train_4.to_csv(file_col[3]+'.csv',encoding = 'utf-8',index = False)
    train_5 = train_temp.iloc[:,[0,1,6]]
    train_5.to_csv(file_col[4]+'.csv',encoding = 'utf-8',index = False)
    train_6 = train_temp.iloc[:,[0,1,7]]
    train_6.to_csv(file_col[5]+'.csv',encoding = 'utf-8',index = False)
    train_7 = train_temp.iloc[:,[0,1,8]]
    train_7.to_csv(file_col[6]+'.csv',encoding = 'utf-8',index = False)
    train_8 = train_temp.iloc[:,[0,1,9]]
    train_8.to_csv(file_col[7]+'.csv',encoding = 'utf-8',index = False)

    #parallel process each feature
    pool = multiprocessing.Pool(4)
    result = pool.map(analyze_by_feature, file_col)
    #result = analyze_by_feature('temp')
    #print(result)
    result_index = train_temp.columns[2:11]
    final_result = pd.DataFrame(result, index =result_index, columns = ['mean','min','max','No. of invalid'],dtype=float)
    #final_result = pd.DataFrame(result, columns = ['mean','min','max','No. of invalid'],dtype=float)
    final_result.to_csv( first_date+'-'+second_date+file_selection+'.csv',encoding='utf-8',na_rep = float('nan'))
    for i in range(0,8):
        os.remove(file_col[i]+".csv")
