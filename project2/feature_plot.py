#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import re
import os
import multiprocessing
import matplotlib.pyplot as plt

#validates the date
def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

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


#read ice coverage file
ice_coverage = readic('mic.csv')
ice_coverage.drop(columns='jday')

#read flight score file
monly_score = pd.read_csv('miHuron1918.csv',skiprows=2)

'''
train=pd.read_csv('miHuron1918.csv')
# this is for auto process, ignore for now
start = 0
for i in range(0, train.shape[0]):
    if type(train.iloc[i,4]) == str:
        start = i-1
        break
print(start)
'''


# In[10]:


#datelist = pd.date_range()


# In[84]:


#create a date array consisting everyday from 1918.1.1 to today
today = datetime.datetime.today().date()
base = datetime.date(1918, 1, 1)
delta = today - base
date_list = [base + datetime.timedelta(days=x) for x in range(0, delta.days)]


# In[85]:


date_list[1]


# In[86]:


#month dictionary, 1 for jan, 2 for feb...
month_dic = ['nan','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']


# In[87]:


date_list[37087]


# In[88]:


daily_score = np.empty(delta.days)
daily_score[:] = np.nan
for i in range(0,delta.days):
    cur_year = date_list[i].year
    cur_mon = date_list[i].month
    cur_score = monly_score.iloc[cur_year-1918][month_dic[cur_mon]]
    daily_score[i] = cur_score


# In[89]:


#read in ord, ugn, dugn data
ord_data = pd.read_csv('1958-11-01-2018-12-31ord.csv')
ugn_data = pd.read_csv('1989-04-21-2018-12-31ugn.csv')
noh_data = pd.read_csv('1923-01-01-2002-07-31noh.csv')


# In[90]:


#convert the date into datetime
ord_data['Date']= pd.to_datetime(ord_data['Date'])
ugn_data['Date']= pd.to_datetime(ugn_data['Date'])
noh_data['Date']= pd.to_datetime(noh_data['Date'])
ugn_date_list = ugn_data['Date'].dt.date
ord_date_list = ord_data['Date'].dt.date
noh_date_list = noh_data['Date'].dt.date


# In[91]:


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


# In[25]:


#plot the temperature max data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['max'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['max'],color = 'green',linewidth = 0.4, label = 'ugn temp')
axes2.plot(noh_date_list, noh_data['Tmax'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('temp max')
axes2.set_ylim([-100,200])
fig.tight_layout()


# In[27]:


#plot the temperature min data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111, label="1")
axes1.plot(date_list, daily_score)
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['min'],color = 'red',linewidth = 0.4)
axes2.plot(ugn_date_list, ugn_data['min'],color = 'yellow',linewidth = 0.4)
axes2.set_ylabel('temp min')
axes2.set_ylim([-100,200])
fig.tight_layout()


# In[28]:


#plot the temperature mean data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['mean'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['mean'],color = 'green',linewidth = 0.4, label = 'ugn temp')
axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('temp mean')
axes2.set_ylim([-100,200])
fig.tight_layout()


# In[ ]:


#plot the temperature mean data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['mean'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['mean'],color = 'green',linewidth = 0.4, label = 'ugn temp')
axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('temp mean')
axes2.set_ylim([-100,200])
fig.tight_layout()


# In[32]:


#plot the dewpt data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['dewpt'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['dewpt'],color = 'green',linewidth = 0.4, label = 'ugn temp')
#axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('dewpt')
axes2.set_ylim([-50,100])
fig.tight_layout()


# In[104]:


#plot the wind speed data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['windS'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['windS'],color = 'green',linewidth = 0.4, label = 'ugn temp')
#axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('wind speed')
axes2.set_ylim([-30,50])
fig.tight_layout()


# In[43]:


#plot the wind speed data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['windD'],color = '#DA7C30',linewidth = 0.4, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['windD'],color = 'green',linewidth = 0.4, label = 'ugn temp')
#axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('wind direction')
axes2.set_ylim([-10,400])
fig.tight_layout()


# In[45]:


#plot the wind speed data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['peak'],color = '#DA7C30',linewidth = 1, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['peak'],color = 'green',linewidth = 1, label = 'ugn temp')
#axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('wind peak')
axes2.set_ylim([-10,400])
fig.tight_layout()


# In[48]:


#plot the wind speed data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['atm'],color = '#DA7C30',linewidth = 1, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['atm'],color = 'green',linewidth = 1, label = 'ugn temp')
#axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('atm pressure')
axes2.set_ylim([900,1100])
fig.tight_layout()


# In[49]:


#plot the wind speed data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['sea'],color = '#DA7C30',linewidth = 1, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['sea'],color = 'green',linewidth = 1, label = 'ugn temp')
#axes2.plot(noh_date_list, noh_data['Tmean'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('sea pressure')
axes2.set_ylim([900,1100])
fig.tight_layout()


# In[102]:


#plot the wind speed data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.plot(ord_date_list, ord_data['precip'],color = '#DA7C30',linewidth = 1, label = 'ord temp')
axes2.plot(ugn_date_list, ugn_data['precip'],color = 'green',linewidth = 1, label = 'ugn temp')
axes2.plot(noh_date_list, noh_data['precip'],color = '#6B4C9A',linewidth = 0.4, label = 'dugn temp')
h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('atm pressure')
#axes2.set_ylim([900,1100])
fig.tight_layout()


# In[107]:


#plot the ice coverage data and the score
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(100)
axes1=fig.add_subplot(111)
axes1.plot(date_list, daily_score, label = 'flight score')
axes1.set_xlabel('years')
axes1.set_ylabel('flight score')

axes2 = axes1.twinx()
axes2.scatter(reshaped_dates, reshaped_ic ,color = '#DA7C30',s=5, label = 'ice coverage')

h1, l1 = axes1.get_legend_handles_labels()
h2, l2 = axes2.get_legend_handles_labels()
axes1.legend(h1+h2, l1+l2, loc=0)
axes2.set_ylabel('ice coverage')
#axes2.set_ylim([900,1100])
fig.tight_layout()
