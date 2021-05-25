#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Libraries
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from scipy.stats import norm
from scipy.special import ndtr as ndtr
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
np.random.seed(8)


# In[3]:


stock = input("Enter Ticker: ")


# In[4]:


stock = web.DataReader(stock, data_source='yahoo', start='2001-07-15')
stock = stock.dropna()
stock


# In[5]:


stock['logReturn'] = np.log(stock['Close'].shift(-1)) - np.log(stock['Close'])


# In[6]:


#Calculations and Histogram
mu = stock['logReturn'].mean()
sigma = stock['logReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(stock['logReturn'].min()-0.01, stock['logReturn'].max()+0.01, 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

stock['logReturn'].hist(bins=500, figsize=(18, 10))
plt.plot(density['x'], density['pdf'], color='red')
plt.show()


# In[7]:


#Value at Risk
VaR = norm.ppf(0.05, mu, sigma)
print('Single Day Value at Risk ', VaR)


# In[8]:


#Value at Risk
print('5% quantile ', norm.ppf(0.05, mu, sigma))
print('95% quantile', norm.ppf(0.95, mu, sigma))


# In[15]:


#Value at Risk
q25 = norm.ppf(0.25, mu, sigma)
print('25% quantile', q25)
q75 = norm.ppf(0.75, mu, sigma)
print('75% quantile', q75)


# In[9]:


#Probability
prob_return1 = norm.cdf(-0.05, mu, sigma)
print('The probability is ', prob_return1)


# In[10]:


#Probability
mu220 = 220*mu
sigma220= (220**0.5) * sigma
print('The probability of dropping over 40% in 220 days is ', norm.cdf(-0.4, mu220, sigma220))


# In[11]:


#Probability
mu220 = 220*mu
sigma220 = (220**0.5) * sigma
drop20 = norm.cdf(-0.2, mu220, sigma220)
print('The probability of dropping over 20% in 220 days is ', drop20)


# In[12]:


#Probability
mu220 = 220*mu
sigma220 = (220**0.5) * sigma
drop10 = norm.cdf(-0.1, mu220, sigma220)
print('The probability of dropping over 10% in 220 days is ', drop10)


# In[14]:


#Probability
mu220 = 220*mu
sigma220 = (220**0.5) * sigma
drop5 = norm.cdf(-0.05, mu220, sigma220)
print('The probability of dropping over 5% in 220 days is ', drop5)


# In[ ]:




