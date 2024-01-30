#!/usr/bin/env python
# coding: utf-8

# In[1]:


URL = "https://www.amazon.in/Columbia-Mens-wind-\ resistant-Glove/dp/B0772WVHPS/?_encoding=UTF8&pd_rd\ _w=d9RS9&pf_rd_p=3d2ae0df-d986-4d1d-8c95-aa25d2ade606&pf\ _rd_r=7MP3ZDYBBV88PYJ7KEMJ&pd_rd_r=550bec4d-5268-41d5-\ 87cb-8af40554a01e&pd_rd_wg=oy8v8&ref_=pd_gw_cr_cartx&th=1" 


# In[2]:


import numpy as np #importing the numpy module which we will be using in this project
import pandas as pd #importing the pandas module which will be used in this porject
import joblib
import string#importing the pandas module which will be used in this porject
from sklearn.model_selection import train_test_split, GridSearchCV#importing the test_train_split module which will be used in this porject
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #importing the classification report adn the confusion matrix module which will be used in this porject
import nltk#importing the nltk module which will be used in this porject
from nltk.corpus import stopwords#importing the nltk.corpus.stopwords module which will be used in this porject
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer#importing the extraction.text.CountVectorizer and TfidfTransformer module which will be used in this porject
from sklearn.pipeline import Pipeline#importing the sklearn.pipeline.Pipeline module which will be used in this porject
from sklearn.ensemble import RandomForestClassifier#importing the sklearn.ensemble.RandomForestClassifier module which will be used in this porject
from sklearn.svm import SVC#importing the sklearn.svm.SVC module which will be used in this porject
from sklearn.linear_model import LogisticRegression#importing the sklearn.linear_model.LogisticRegression module which will be used in this porject


# In[3]:


import nltk
nltk.download('stopwords')


# In[4]:


def convertmyTxt(rv): #here we areA defining a function
    np = [c for c in rv if c not in string.punctuation] #this function is checking if it is present in punctuation or not.
    np = ''.join(np) #the character which are not in punctuation, we are storing them in a separate string
    return [w for w in np.split() if w.lower() not in stopwords.words('english')] #here we are returning a list of words from the sentences we just made in above line and checking if it is not a stopword


# In[5]:


pipeline = joblib.load('pipeline.pkl')


# In[6]:


pipeline


# In[8]:


# import module 
import requests 
from bs4 import BeautifulSoup 

HEADERS = ({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \ AppleWebKit/537.36 (KHTML, like Gecko) \ Chrome/90.0.4430.212 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'}) 

# user define function 
# Scrape the data 
def getdata(url): 
    r = requests.get(url, headers=HEADERS,verify=False) 
    return r.text 


def html_code(url): 

	# pass the url 
	# into getdata function 
    htmldata = getdata(url) 
    soup = BeautifulSoup(htmldata, 'html.parser') 

	# display html code 
    return (soup) 


url = URL
soup = html_code(url) 


def cus_rev(soup): 
	# find the Html tag 
	# with find() 
	# and convert into string 
    data_str = "" 
    for item in soup.find_all("div", class_= "a-expander-content reviewText review-text-content a-expander-partial-collapse-content"): 
        data_str = data_str + item.get_text() 
#     for item in soup.find_all("div", class_="a-expander-content \ reviewText review-text-content a-expander-partial-collapse-content"): 
#         print(item)
#         data_str = data_str + item.get_text() 

    result = data_str.split("\n") 
    return (result) 

rev_data = cus_rev(soup) 
rev_result = [] 
for i in rev_data: 
        if i is "": 
            pass
        else: 
            rev_result.append(i) 
df = pd.DataFrame(rev_result)
df.rename(columns = {0:"review"},inplace=True)
df.to_csv('amazon_review.csv') 


# In[9]:


validation = pd.read_csv('amazon_review.csv')


# In[10]:


val_result  = pipeline.predict(validation['review'])
final_result = pd.concat([pd.DataFrame(validation['review']).reset_index(drop=True),pd.DataFrame(val_result)],axis = 1)
final_result.to_csv("Categorize_amazon_reviews")


# In[11]:


final_result


# In[ ]:




