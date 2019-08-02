#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import math
import re
import os
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
register_matplotlib_converters()

#validates the date
def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# output: pandas dataframe
def readfile_noh(filename):
    #performing task 1
    trash_offset = 25
    trash_index = 0
    train = pd.read_csv(filename, skiprows= range(0,7), dtype = str)
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
    nrows = train.shape[0]
    #print(nrows)
    for x in range(nrows-trash_offset,nrows):
         if type(train.loc[x]['TMAX']) != str:
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


    return train

#train_1958

# this function reads a csv file and process it by
# 1. removing the trash
# 2. get date into the same format
# 3. get time into the same format
# 4. fix the wind speed (change into string)
# input: filename str ---eg.'2011-2018ord.csv'
# output: pandas dataframe
def readfile_ord(filename):

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

# this function takes in a date and calculate the mean min max for the features
# input: date -- string in the form of 'yyyy-mm-dd' eg:'1958-11-01'
#        train -- the main datafram to analyze
# output-- list containing:
#        sum_result -- datafram for sum of each day
#        invalid_feature
def analyze_by_day_precip(date, train):
    #initialize
    sum_result = float('nan')
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
        df1.fillna(value=0, inplace = True)

        sum_result = df1.sum()[0]

    return sum_result,min_result,max_result,invalid_feature


def analyze_by_feature_1(feature):
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

"""
takes the feature, read the data from feature.csv, for each day, calculate the daily mean/max/min and invalid days

"""
def analyze_by_feature_2(feature):

    print('im here')
    sum_temp = []
    mean_temp = []
    min_temp =[]
    max_temp = []
    invalid_temp = []
    train_feature = pd.read_csv(feature+'.csv', dtype = str)


    train_index = pd.unique(train_feature['Date'])
    train_index = list(train_index)
    #print(train_index)
    if feature == 'precip':

        #if current feature is precip, calculate the sum for each day and invalid
        for i in range(len(train_index)):
            #print(feature)
            #print(train_index[i])
            temp = analyze_by_day_precip(train_index[i], train_feature)

            sum_temp.append(temp[0])
            min_temp.append(temp[1])
            max_temp.append(temp[2])
            invalid_temp.append(temp[3])

        #group these days together
        train_df = pd.DataFrame(train_index)
        sum_df = pd.DataFrame(sum_temp)
        min_df = pd.DataFrame(min_temp)
        max_df = pd.DataFrame(max_temp)
        invalid_df = pd.DataFrame(invalid_temp)

        #write the feature out into another file
        temp_write = pd.concat([train_df,max_df,min_df,sum_df],axis = 1)
        temp_write.columns = ['Date','max','min','sum']
        temp_write.to_csv(feature+'_test.csv',index = False, na_rep = float('nan'))
        return

    else:
        # for other features, do the same except get max/min/mean
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
        train_df = pd.DataFrame(train_index)
        mean_df = pd.DataFrame(mean_temp)
        min_df = pd.DataFrame(min_temp)
        max_df = pd.DataFrame(max_temp)
        invalid_df = pd.DataFrame(invalid_temp)


        #write the feature out into another file
        temp_write = pd.concat([train_df,max_df,min_df,mean_df],axis = 1)
        temp_write.columns = ['Date','max','min','mean']
        temp_write.to_csv(feature+'_test.csv',index = False, na_rep = float('nan'))
        return

#draws the daily feature data as box plot and daily lake depth as line
#input: string of the feature name etc: feature from feature.csv
#output: none
def draw_by_feature(feature):
    #read feature data
    train_feature = pd.read_csv(feature+'.csv', dtype = str)

    #convert the M/m into nan and date into datetime
    train_feature.iloc[:,2].replace('M','nan',inplace = True)
    train_feature.iloc[:,2].replace('m','nan',inplace = True)
    train_feature.columns = ['Date', 'Time', feature]
    train_feature[feature] = pd.to_numeric(train_feature[feature],errors='coerce')
    train_feature['Date'] = pd.to_datetime(train_feature['Date'], errors='coerce')
    #type(train_1.iloc[0,0])

    #train_feature[feature] = pd.to_numeric(train_feature[feature])


    #read in the score data
    score_temp = pd.read_csv('temp_score.csv',dtype={'Date': str, 'lake depth':float})
    score_temp['Date'] = pd.to_datetime(score_temp['Date'], errors = 'coerce')
    date_tick = score_temp['Date'].dt.date
    start_date = date_tick[0].strftime('%Y-%m-%d')
    end_date = date_tick[len(date_tick)-1].strftime('%Y-%m-%d')

    # if we have feature value to plot, plot as box
    if not train_feature.empty:
        #find the first and last day of the range
        x1 = pd.to_datetime( train_feature.iloc[0,0] )
        x2 = pd.to_datetime( train_feature.iloc[train_feature.shape[0]-1,0] )

        # now get ready for the line plots
        score_temp['Date'] = (score_temp['Date'] - x1).dt.days
        # plot the score as line plot
        fig = plt.figure(figsize=(300,20))
        ax1 = fig.add_subplot(111)
        ax1 = sns.lineplot( x="Date", y = 'lake depth', data = score_temp)
        ax1.set_xticklabels( date_tick, rotation=60 )
        # convert the date into int with start date = 0
        train_feature['Date'] = (train_feature['Date'] - x1).dt.days

        ax2 = ax1.twinx()
        ax2= sns.boxplot( x="Date", y=feature, data=train_feature)
        ax2.set_xticklabels( date_tick, rotation=60 )
        #ax2.set(ylim=(0,1300))

    else:
        fig = plt.figure(figsize=(300,20))
        ax1 = fig.add_subplot(111)
        ax1.plot( date_tick, score_temp['lake depth'])
    # give the plot a title and output
    ax1.title.set_text(feature + '(box) and lake depth vs time')
    output_str = start_date + '_' + end_date + '_' + feature + '_box_visulization.png'
    fig.savefig(output_str)
    return

# read the ice coverage data
# input the filename of the ice coverge data
# output dataframe containing the read ice coverage data
def readic(filename):
    #read the csv file for ice coverage
    train = pd.read_csv(filename)
    # get rid of the last two columns as they are repetead
    train = train.iloc[:, 0:50]
    #rename the days column and make it index
    temp = train.columns.values
    temp[0] = 'days'
    train.columns = temp
    train.set_index('days')
    #get rid of the bottom nan rows
    nrows = train.shape[0]
    bot_bound = nrows-10
    for i in range(bot_bound, nrows):
        if type(train.iloc[i]['days']) == float:
            print(i)
            train.drop(range(i,nrows),inplace=True)
            break
    return train



def main():

    multiprocessing.freeze_support()
    flag = input('Please input what you want to do( 1 for outputting a range and 2 for outputting daily result)')
    path = input('Please type in the path of your data folder:')
    # read all the csv files
    file_selection = ''
    while 1==1:
        file_selection = input('Please input the location of data you want to select '+ '(ugn, ord'+'):')

        #if file_selection == 'ugn' or file_selection == 'ord' or file_selection == 'noh':
        if file_selection == 'ugn' or file_selection == 'ord' :
            break
    listOfFiles = os.listdir(path)
    listOfFiles.sort()

    # use re to detect which location we need
    file_pattern_ord = re.compile(r'\d\d\d\dord.csv')
    file_pattern_ugn = re.compile(r'\d\d\d\dugn.csv')
    file_pattern_noh = re.compile(r'\d\d\d\ddugn.csv')
    if file_selection == 'ugn':
        file_pattern = file_pattern_ugn
    elif file_selection == 'ord':
        file_pattern = file_pattern_ord
    else:
        file_pattern = file_pattern_noh
    train_temp = pd.DataFrame()
    for x in range(0,len(listOfFiles)):
        searchObj = re.search(file_pattern, listOfFiles[x])
        if searchObj:
            print (listOfFiles[x] )
            if file_selection == 'ugn' or file_selection == 'ord':
                train_temp = pd.concat([train_temp,readfile_ord(path+'/'+listOfFiles[x])], axis = 0, ignore_index=True)
            else:
                train_temp = pd.concat([train_temp,readfile_noh(path+'/'+listOfFiles[x])], axis = 0, ignore_index=True)
    if train_temp.empty:
        print('Cannot find any file please check your file name again.')

    #print(train_temp)
    # check ord time span
    while file_selection == 'ord':
        first_date = input("From 1958-11-01 to 2018-12-30, please input a valid starting date as in yyyy-mm-dd: ")
        while validate(first_date) == False:
            first_date = input("Wrong input! From 1958-11-01 to 2018-12-30, please input a valid starting date as in yyyy-mm-dd: ")

        d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
        if d1 >=datetime.date(1958,11,1) and d1 <=datetime.date(2018,12,30):
            break

    while file_selection == 'ord':
        second_date = input("From 1958-11-02 to 2018-12-31, please input the ending date as in yyyy-mm-dd: ")
        while validate(second_date) == False:
            second_date = input("Wrong input! From 1958-11-02 to 2018-12-31, please input the ending date as in yyyy-mm-dd: ")
        d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
        if d2 >=datetime.date(1958,11,2) and d2 <=datetime.date(2018,12,31):
            break

    # check ugn time span
    while file_selection == 'ugn':
        first_date = input("From 1989-04-21 to 2018-12-30, please input a valid starting date as in yyyy-mm-dd: ")
        while validate(first_date) == False:
            first_date = input("Wrong input! From 1989-04-21 to 2018-12-30, please input a valid starting date as in yyyy-mm-dd: ")
        d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
        if d1 >=datetime.date(1989,4,21) and d1 <=datetime.date(2018,12,30):
            break

    while file_selection == 'ugn':
        second_date = input("From 1989-04-22 to 2018-12-31, please input the ending date as in yyyy-mm-dd: ")
        while validate(second_date) == False:
            second_date = input("Wrong input! From 1989-04-22 to 2018-12-31, please input the ending date as in yyyy-mm-dd: ")
        d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
        if d2 >=datetime.date(1989,4,22) and d2 <=datetime.date(2018,12,31):
            break

    # check noh time span
    while file_selection == 'noh':
        first_date = input("From 1923-01-01 to 2002-07-30, please input a valid starting date as in yyyy-mm-dd: ")
        while validate(first_date) == False:
            first_date = input("Wrrong input! From 1923-01-01 to 2002-07-30, please input a valid starting date as in yyyy-mm-dd: ")
        d1 = datetime.datetime.strptime(first_date, "%Y-%m-%d").date()
        if d1 >=datetime.date(1923,1,1) and d1 <=datetime.date(2002,7,30):
            break

    while file_selection == 'noh':
        second_date = input("From 1923-01-02 to 2002-07-31, please input the ending date as in yyyy-mm-dd: ")
        while validate(second_date) == False:
            second_date = input("Wrong input! From 1923-01-02 to 2002-07-31, please input the ending date as in yyyy-mm-dd: ")
        d2 = datetime.datetime.strptime(second_date, "%Y-%m-%d").date()
        if d2 >=datetime.date(1923,1,2) and d2 <=datetime.date(2002,7,31):
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
            sys.exit()
        #print(second_index)

        else:
            train_temp = train_temp.iloc[first_index:second_index+1]


        #print(train_temp)
    # if the file is noh, simply loop through and replace T with 0.001 and M with nan
    if file_selection == 'noh':
        train_temp.replace({'t':'0.001','T':'0.001','m':'nan','M':'nan' },inplace = True)
        train_temp.rename(columns = {'PRCP':'precip','TMAX':'Tmax','TMIN':'Tmin','MEAN':'Tmean'},inplace = True)
        train_temp = train_temp.astype({'precip':'float','Tmax':'float','Tmin':'float','Tmean':'float'})


    else:
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






    # In[3]:


    ##### reads the lake depth into daily_score
    #read lake depth
    monly_score = pd.read_csv('miHuron1918.csv',skiprows=2)
    #create a date array consisting everyday from 1918.1.1 to today
    today = datetime.datetime.today().date()
    base = datetime.date(1918, 1, 1)
    delta = today - base
    date_list = [base + datetime.timedelta(days=x) for x in range(0, delta.days)]
    #month dictionary, 1 for jan, 2 for feb...
    month_dic = ['nan','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    #fill the daily score
    daily_score = np.empty(delta.days)
    daily_score[:] = np.nan
    for i in range(0,delta.days):
        cur_year = date_list[i].year
        cur_mon = date_list[i].month
        cur_score = monly_score.iloc[cur_year-1918][month_dic[cur_mon]]
        daily_score[i] = cur_score

    #### read ends

    #find the range of days in the date_list of score
    start_index = date_list.index(pd.to_datetime( train_1.iloc[0,0] ).date()) # start of feature date
    end_index = date_list.index(pd.to_datetime( train_1.iloc[train_1.shape[0]-1,0] ).date()) # end of feature date
    score_output_date = date_list[start_index:end_index+1]
    score_output_score = daily_score[start_index:end_index+1]

    # writes the processed flight data into csv
    score_output = pd.DataFrame({'Date': score_output_date, 'lake depth':score_output_score})
    score_output.to_csv('temp_score.csv',index = False)

    file_col
    # start the parallel
    pool = multiprocessing.Pool(12)
    pool.map(draw_by_feature, file_col)



#######################################
    # now we draw the line plots
    #read in data
    ice_coverage = readic('mic.csv')
    monly_score = pd.read_csv('miHuron1918.csv',skiprows=2)
    #create a date array consisting everyday from 1918.1.1 to today
    today = datetime.datetime.today().date()
    base = datetime.date(1918, 1, 1)
    delta = today - base
    date_list = [base + datetime.timedelta(days=x) for x in range(0, delta.days)]


    # In[11]:


    date_list[1]


    # In[12]:


    #month dictionary, 1 for jan, 2 for feb...
    month_dic = ['nan','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']


    # In[13]:


    date_list[37087]


    # In[14]:


    daily_score = np.empty(delta.days)
    daily_score[:] = np.nan
    for i in range(0,delta.days):
        cur_year = date_list[i].year
        cur_mon = date_list[i].month
        cur_score = monly_score.iloc[cur_year-1918][month_dic[cur_mon]]
        daily_score[i] = cur_score


    # In[15]:


    #read in ord, ugn, dugn data
    ord_data = pd.read_csv('1958-11-01-2018-12-31ord.csv')
    ugn_data = pd.read_csv('1989-04-21-2018-12-31ugn.csv')
    noh_data = pd.read_csv('1923-01-01-2002-07-31noh.csv')


    # In[16]:


    #convert the date into datetime
    ord_data['Date']= pd.to_datetime(ord_data['Date'])
    ugn_data['Date']= pd.to_datetime(ugn_data['Date'])
    noh_data['Date']= pd.to_datetime(noh_data['Date'])
    ugn_date_list = ugn_data['Date'].dt.date
    ord_date_list = ord_data['Date'].dt.date
    noh_date_list = noh_data['Date'].dt.date


    # In[17]:


    #create two series, reshaped_ic containing the daily ice coverage data and reshaped_dates cotaining
    #corresponding dates
    reshaped_ic = pd.Series([])
    ic_days = ice_coverage['days']
    reshaped_dates = pd.Series([])
    for i in range(1973,2020):
        #print(ice_coverage[str(i)])
        reshaped_ic = reshaped_ic.append(ice_coverage[str(i)],ignore_index=True)
        reshaped_dates = reshaped_dates.append(ic_days+'-'+str(i),ignore_index=True)
    reshaped_dates = pd.to_datetime(reshaped_dates, errors='coerce')
    reshaped_dates = reshaped_dates.dt.date


    # In[18]:


    #plot the temperature max data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['max'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
    axes2.plot(ugn_date_list, ugn_data['max'],color = 'green',linewidth = 0.4, label = 'ugn temp')
    axes2.plot(noh_date_list, noh_data['Tmax'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('temp max')
    axes2.set_ylim([-100,200])
    axes1.title.set_text('max temp vs lake depth')
    fig.tight_layout()
    fig.savefig('maxtemp_visulization.png')



    # In[19]:


    #plot the temperature min data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['min'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
    axes2.plot(ugn_date_list, ugn_data['min'],color = 'green',linewidth = 0.4, label = 'ugn temp')
    axes2.plot(noh_date_list, noh_data['Tmin'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('temp min')
    axes2.set_ylim([-100,200])
    axes1.title.set_text('min temp vs lake depth')
    fig.tight_layout()
    fig.savefig('mintemp_visulization.png')



    # In[20]:


    #plot the temperature mean data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['mean'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
    axes2.plot(ugn_date_list, ugn_data['mean'],color = 'green',linewidth = 0.4, label = 'ugn temp')
    axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('temp mean')
    axes2.set_ylim([-100,200])
    axes1.title.set_text('mean temp vs lake depth')
    fig.tight_layout()
    fig.savefig('meantemp_visulization.png')



    # In[21]:


    #plot the dewpt data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['dewpt'],color = '#DA7C30',linewidth = 0.4, label = 'ord dewpt')
    axes2.plot(ugn_date_list, ugn_data['dewpt'],color = 'green',linewidth = 0.4, label = 'ugn dewpt')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('dewpt')
    axes2.set_ylim([-50,100])
    axes1.title.set_text('dewpt vs lake depth')
    fig.tight_layout()
    fig.savefig('dewpt_visulization.png')



    # In[22]:


    #plot the wind speed data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['windS'],color = '#DA7C30',linewidth = 0.4, label = 'ord wind speed')
    axes2.plot(ugn_date_list, ugn_data['windS'],color = 'green',linewidth = 0.4, label = 'ugn wind speed')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('wind speed')
    axes2.set_ylim([-30,50])
    axes1.title.set_text('wind speed vs lake depth')
    fig.tight_layout()
    fig.savefig('windspeed_visulization.png')



    # In[23]:


    #plot the wind direction data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['windD'],color = '#DA7C30',linewidth = 0.4, label = 'ord wind direction')
    axes2.plot(ugn_date_list, ugn_data['windD'],color = 'green',linewidth = 0.4, label = 'ugn wind direction')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('wind direction')
    axes2.set_ylim([-400,400])
    axes1.title.set_text('wind direction vs lake depth')
    fig.tight_layout()
    fig.savefig('winddirection_visulization.png')



    # In[44]:


    #plot the wind peak data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['peak'],color = '#DA7C30',linewidth = 1, label = 'ord wind peak')
    axes2.plot(ugn_date_list, ugn_data['peak'],color = 'green',linewidth = 1, label = 'ugn wind peak')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('wind peak')
    axes2.set_ylim([-10,400])
    axes1.title.set_text('wind peak vs lake depth')
    fig.tight_layout()
    fig.savefig('windpeak_visulization.png')



    # In[45]:


    #plot the atm pressure data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['atm'],color = '#DA7C30',linewidth = 1, label = 'ord atm pressure')
    axes2.plot(ugn_date_list, ugn_data['atm'],color = 'green',linewidth = 1, label = 'ugn atm pressure')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('atm pressure')
    axes2.set_ylim([900,1100])
    axes1.title.set_text('atm pressure vs lake depth')
    fig.tight_layout()
    fig.savefig('atmpressure_visulization.png')



    # In[46]:


    #plot the sea pressure data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['sea'],color = '#DA7C30',linewidth = 1, label = 'ord sea pressure')
    axes2.plot(ugn_date_list, ugn_data['sea'],color = 'green',linewidth = 1, label = 'ugn sea pressure')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('sea pressure')
    axes2.set_ylim([900,1100])
    axes1.title.set_text('sea pressure vs lake depth')
    fig.tight_layout()
    fig.savefig('seapressure_visulization.png')



    # In[47]:


    #plot the precip data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_data['precip'],color = '#DA7C30',linewidth = 1, label = 'ord precip')
    axes2.plot(ugn_date_list, ugn_data['precip'],color = 'green',linewidth = 1, label = 'ugn precip')
    axes2.plot(noh_date_list, noh_data['precip'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn precip')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('precip')
    #axes2.set_ylim([900,1100])
    axes1.title.set_text('precip vs lake depth')
    fig.tight_layout()
    fig.savefig('precip_visulization.png')



    # In[48]:


    #plot the ice coverage data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.scatter(reshaped_dates, reshaped_ic ,color = '#DA7C30',s=5, label = 'ice coverage')
    #axes2.plot(reshaped_dates, reshaped_ic ,color = '#DA7C30', label = 'ice coverage')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('ice coverage')
    #axes2.set_ylim([900,1100])
    fig.tight_layout()
    fig.savefig('icecoverage_visulization.png')



    # In[25]:


    #calculate the ord deltas for 8 features

    #####max
    num_data_ord = len(ord_data['max'])
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['max'][0]
    new_data[1:num_data_ord] = ord_data['max'][0:num_data_ord-1]
    ord_max_delta = ord_data['max'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_max_delta],axis =1)
    temp.columns = ['Date', 'max']
    temp.to_csv( 'delta_max_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####min
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['min'][0]
    new_data[1:num_data_ord] = ord_data['min'][0:num_data_ord-1]
    ord_min_delta = ord_data['min'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_min_delta],axis =1)
    temp.columns = ['Date', 'min']
    temp.to_csv( 'delta_min_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)


    #####mean
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['mean'][0]
    new_data[1:num_data_ord] = ord_data['mean'][0:num_data_ord-1]
    ord_mean_delta = ord_data['mean'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_mean_delta],axis =1)
    temp.columns = ['Date', 'mean']
    temp.to_csv( 'delta_mean_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####dewpt
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['dewpt'][0]
    new_data[1:num_data_ord] = ord_data['dewpt'][0:num_data_ord-1]
    ord_dewpt_delta = ord_data['dewpt'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_dewpt_delta],axis =1)
    temp.columns = ['Date', 'dewpt']
    temp.to_csv( 'delta_dewpt_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####windS
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['windS'][0]
    new_data[1:num_data_ord] = ord_data['windS'][0:num_data_ord-1]
    ord_windS_delta = ord_data['windS'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_windS_delta],axis =1)
    temp.columns = ['Date', 'windS']
    temp.to_csv( 'delta_windS_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####windD
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['windD'][0]
    new_data[1:num_data_ord] = ord_data['windD'][0:num_data_ord-1]
    ord_windD_delta = ord_data['windD'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_windD_delta],axis =1)
    temp.columns = ['Date', 'windD']
    temp.to_csv( 'delta_windD_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####peak
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['peak'][0]
    new_data[1:num_data_ord] = ord_data['peak'][0:num_data_ord-1]
    ord_peak_delta = ord_data['peak'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_peak_delta],axis =1)
    temp.columns = ['Date', 'peak']
    temp.to_csv( 'delta_peak_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####atm
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['atm'][0]
    new_data[1:num_data_ord] = ord_data['atm'][0:num_data_ord-1]
    ord_atm_delta = ord_data['atm'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_atm_delta],axis =1)
    temp.columns = ['Date', 'atm']
    temp.to_csv( 'delta_atm_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####sea
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['sea'][0]
    new_data[1:num_data_ord] = ord_data['sea'][0:num_data_ord-1]
    ord_sea_delta = ord_data['sea'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_sea_delta],axis =1)
    temp.columns = ['Date', 'sea']
    temp.to_csv( 'delta_sea_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####precip
    new_data = pd.Series(np.zeros(num_data_ord))
    new_data[0] = ord_data['precip'][0]
    new_data[1:num_data_ord] = ord_data['precip'][0:num_data_ord-1]
    ord_precip_delta = ord_data['precip'] - new_data

    # export to csv
    temp = pd.concat([ord_date_list, ord_precip_delta],axis =1)
    temp.columns = ['Date', 'precip']
    temp.to_csv( 'delta_precip_ord.csv',encoding='utf-8',na_rep = float('nan'),index = False)


    # In[50]:


    #calculate the ugn deltas for 8 features
    num_data_ugn = len(ugn_data['max'])
    #####max
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['max'][0]
    new_data[1:num_data_ugn] = ugn_data['max'][0:num_data_ugn-1]
    ugn_max_delta = ugn_data['max'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_max_delta],axis =1)
    temp.columns = ['Date', 'max']
    temp.to_csv( 'delta_max_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####min
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['min'][0]
    new_data[1:num_data_ugn] = ugn_data['min'][0:num_data_ugn-1]
    ugn_min_delta = ugn_data['min'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_min_delta],axis =1)
    temp.columns = ['Date', 'min']
    temp.to_csv( 'delta_min_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)


    #####mean
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['mean'][0]
    new_data[1:num_data_ugn] = ugn_data['mean'][0:num_data_ugn-1]
    ugn_mean_delta = ugn_data['mean'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_mean_delta],axis =1)
    temp.columns = ['Date', 'mean']
    temp.to_csv( 'delta_mean_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####dewpt
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['dewpt'][0]
    new_data[1:num_data_ugn] = ugn_data['dewpt'][0:num_data_ugn-1]
    ugn_dewpt_delta = ugn_data['dewpt'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_dewpt_delta],axis =1)
    temp.columns = ['Date', 'dewpt']
    temp.to_csv( 'delta_dewpt_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####windS
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['windS'][0]
    new_data[1:num_data_ugn] = ugn_data['windS'][0:num_data_ugn-1]
    ugn_windS_delta = ugn_data['windS'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_windS_delta],axis =1)
    temp.columns = ['Date', 'windS']
    temp.to_csv( 'delta_windS_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####windD
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['windD'][0]
    new_data[1:num_data_ugn] = ugn_data['windD'][0:num_data_ugn-1]
    ugn_windD_delta = ugn_data['windD'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_windD_delta],axis =1)
    temp.columns = ['Date', 'windD']
    temp.to_csv( 'delta_windD_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####peak
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['peak'][0]
    new_data[1:num_data_ugn] = ugn_data['peak'][0:num_data_ugn-1]
    ugn_peak_delta = ugn_data['peak'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_peak_delta],axis =1)
    temp.columns = ['Date', 'peak']
    temp.to_csv( 'delta_peak_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####atm
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['atm'][0]
    new_data[1:num_data_ugn] = ugn_data['atm'][0:num_data_ugn-1]
    ugn_atm_delta = ugn_data['atm'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_atm_delta],axis =1)
    temp.columns = ['Date', 'atm']
    temp.to_csv( 'delta_atm_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####sea
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['sea'][0]
    new_data[1:num_data_ugn] = ugn_data['sea'][0:num_data_ugn-1]
    ugn_sea_delta = ugn_data['sea'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_sea_delta],axis =1)
    temp.columns = ['Date', 'sea']
    temp.to_csv( 'delta_sea_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####precip
    new_data = pd.Series(np.zeros(num_data_ugn))
    new_data[0] = ugn_data['precip'][0]
    new_data[1:num_data_ugn] = ugn_data['precip'][0:num_data_ugn-1]
    ugn_precip_delta = ugn_data['precip'] - new_data

    # export to csv
    temp = pd.concat([ugn_date_list, ugn_precip_delta],axis =1)
    temp.columns = ['Date', 'precip']
    temp.to_csv( 'delta_precip_ugn.csv',encoding='utf-8',na_rep = float('nan'),index = False)


    # In[51]:


    #noh data
    num_data_noh = len(noh_data['precip'])
    #####precip
    new_data = pd.Series(np.zeros(num_data_noh))
    new_data[0] = noh_data['precip'][0]
    new_data[1:num_data_noh] = noh_data['precip'][0:num_data_noh-1]
    noh_precip_delta = noh_data['precip'] - new_data

    # export to csv
    temp = pd.concat([noh_date_list, noh_precip_delta],axis =1)
    temp.columns = ['Date', 'precip']
    temp.to_csv( 'delta_precip_noh.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####tmax
    new_data = pd.Series(np.zeros(num_data_noh))
    new_data[0] = noh_data['Tmax'][0]
    new_data[1:num_data_noh] = noh_data['Tmax'][0:num_data_noh-1]
    noh_max_delta = noh_data['Tmax'] - new_data

    # export to csv
    temp = pd.concat([noh_date_list, noh_max_delta],axis =1)
    temp.columns = ['Date', 'max']
    temp.to_csv( 'delta_max_noh.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####tmin
    new_data = pd.Series(np.zeros(num_data_noh))
    new_data[0] = noh_data['Tmin'][0]
    new_data[1:num_data_noh] = noh_data['Tmin'][0:num_data_noh-1]
    noh_min_delta = noh_data['Tmin'] - new_data

    # export to csv
    temp = pd.concat([noh_date_list, noh_min_delta],axis =1)
    temp.columns = ['Date', 'min']
    temp.to_csv( 'delta_min_noh.csv',encoding='utf-8',na_rep = float('nan'),index = False)

    #####tmean
    new_data = pd.Series(np.zeros(num_data_noh))
    new_data[0] = noh_data['Tmean'][0]
    new_data[1:num_data_noh] = noh_data['Tmean'][0:num_data_noh-1]
    noh_mean_delta = noh_data['Tmean'] - new_data

    # export to csv
    temp = pd.concat([noh_date_list, noh_mean_delta],axis =1)
    temp.columns = ['Date', 'mean']
    temp.to_csv( 'delta_mean_noh.csv',encoding='utf-8',na_rep = float('nan'),index = False)


    # In[53]:


    #graph delta
    #plot the delta temperature max data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_max_delta,color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
    axes2.plot(ugn_date_list, ugn_max_delta,color = 'green',linewidth = 0.4, label = 'ugn temp')
    axes2.plot(noh_date_list, noh_max_delta,color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta temp max')
    axes2.set_ylim([-100,100])
    axes1.title.set_text('delta max temp vs lake depth')

    fig.tight_layout()
    fig.savefig('delta_maxtemp_visulization.png')



    # In[55]:


    #plot the delta temperature min data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_min_delta,color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
    axes2.plot(ugn_date_list, ugn_min_delta,color = 'green',linewidth = 0.4, label = 'ugn temp')
    axes2.plot(noh_date_list, noh_min_delta,color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta temp min')
    axes2.set_ylim([-100,200])
    axes1.title.set_text('delta min temp vs lake depth')
    fig.tight_layout()

    fig.savefig('delta_mintemp_visulization.png')



    # In[56]:


    #plot the delta temperature mean data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_mean_delta,color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
    axes2.plot(ugn_date_list, ugn_mean_delta,color = 'green',linewidth = 0.4, label = 'ugn temp')
    axes2.plot(noh_date_list, noh_mean_delta,color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta temp mean')
    axes2.set_ylim([-100,200])
    axes1.title.set_text('delta mean temp vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_meantemp_visulization.png')



    # In[57]:


    #plot the delta dewpt data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_dewpt_delta,color = '#DA7C30',linewidth = 0.4, label = 'ord dewpt')
    axes2.plot(ugn_date_list, ugn_dewpt_delta,color = 'green',linewidth = 0.4, label = 'ugn dewpt')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta dewpt')
    axes2.set_ylim([-50,100])
    axes1.title.set_text('delta dewpt vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_dewpt_visulization.png')



    # In[28]:


    #plot the delta wind speed data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_windS_delta,color = '#DA7C30',linewidth = 0.4, label = 'ord wind speed')
    axes2.plot(ugn_date_list, ugn_windS_delta,color = 'green',linewidth = 0.4, label = 'ugn wind speed')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta wind speed')
    axes2.set_ylim([-100,100])
    axes1.title.set_text('delta wind speed vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_windspeed_visulization.png')



    # In[29]:


    #plot the delta wind direction data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_windD_delta,color = '#DA7C30',linewidth = 0.4, label = 'ord wind direction')
    axes2.plot(ugn_date_list, ugn_windD_delta,color = 'green',linewidth = 0.4, label = 'ugn wind direction')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta wind direction')
    axes2.set_ylim([-400,400])
    axes1.title.set_text('delta wind direction vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_winddirection_visulization.png')



    # In[78]:


    #plot the delta wind peak data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_peak_delta,color = '#DA7C30',linewidth = 1, label = 'ord wind peak')
    axes2.plot(ugn_date_list, ugn_peak_delta,color = 'green',linewidth = 1, label = 'ugn wind peak')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta wind peak')
    axes2.set_ylim([-10,400])
    axes1.title.set_text('delta wind peak vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_windpeak_visulization.png')



    # In[79]:


    #plot the delta atm pressure data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_atm_delta,color = '#DA7C30',linewidth = 1, label = 'ord atm pressure')
    axes2.plot(ugn_date_list, ugn_atm_delta,color = 'green',linewidth = 1, label = 'ugn atm pressure')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta atm pressure')
    axes2.set_ylim([-100,100])
    axes1.title.set_text('delta atm pressure vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_atmpressure_visulization.png')



    # In[80]:


    #plot the delta sea pressure data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_sea_delta,color = '#DA7C30',linewidth = 1, label = 'ord sea pressure')
    axes2.plot(ugn_date_list, ugn_sea_delta,color = 'green',linewidth = 1, label = 'ugn sea pressure')
    #axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta sea pressure')
    axes2.set_ylim([-100,100])
    axes1.title.set_text('delta sea pressure vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_seapressure_visulization.png')



    # In[81]:


    #plot the delta precip data and the score
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(100)
    axes1=fig.add_subplot(111)
    axes1.plot(date_list, daily_score, label = 'lake depth')
    axes1.set_xlabel('years')
    axes1.set_ylabel('lake depth')

    axes2 = axes1.twinx()
    axes2.plot(ord_date_list, ord_precip_delta,color = '#DA7C30',linewidth = 1, label = 'ord precip')
    axes2.plot(ugn_date_list, ugn_precip_delta,color = 'green',linewidth = 1, label = 'ugn precip')
    axes2.plot(noh_date_list, noh_precip_delta,color = '#6B4C9A',linewidth = 0.4, label = 'dugn precip')
    h1, l1 = axes1.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes1.legend(h1+h2, l1+l2, loc=0)
    axes2.set_ylabel('delta precip')
    #axes2.set_ylim([900,1100])
    axes1.title.set_text('delta precip vs lake depth')
    fig.tight_layout()
    fig.savefig('delta_precip_visulization.png')


    return

if __name__ == '__main__':
    main()
    restart = input('Would you like to restart the program? (y for yes, anything else for no)')
    while restart == 'y':
        main()
        restart = input('Would you like to restart the program? (y for yes, anything else for no)')

    sys.exit()
