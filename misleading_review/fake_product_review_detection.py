#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np #importing the numpy module which we will be using in this project
import pandas as pd #importing the pandas module which will be used in this porject
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


# In[29]:


import nltk
nltk.download('stopwords')


# In[30]:


dataframe = pd.read_csv('dataset.csv') #reading our dataset which contains text and a label whether it is fake or real
dataframe.head() #printing the first 5 columsn in our dataset


# In[31]:


dataframe.drop('Unnamed: 0',axis=1,inplace=True)## dropping the unnecessary column 'UNAMED'


# In[32]:


dataframe.head() #pritning the dataset again after dropping the column


# In[33]:


dataframe.dropna(inplace=True) #dropping alll the null rows in the dataset


# In[34]:


dataframe['length'] = dataframe['text_'].apply(len) #storing the length of all the text into a separate column called 'length'


# Let's extract the largest review...

# In[35]:


dataframe[dataframe['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_ ##so here we are just collecting the words which are most common in the fake reviews so that we can identify these wrods to detect for future text


# In[9]:


def convertmyTxt(rv): #here we areA defining a function
    np = [c for c in rv if c not in string.punctuation] #this function is checking if it is present in punctuation or not.
    np = ''.join(np) #the character which are not in punctuation, we are storing them in a separate string
    return [w for w in np.split() if w.lower() not in stopwords.words('english')] #here we are returning a list of words from the sentences we just made in above line and checking if it is not a stopword


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(dataframe['text_'],dataframe['label'],test_size=0.25)


# In[11]:


pip = Pipeline([
    ('bow',CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
]) #here we are defining our Random Forest Classifier model in which we will pass the training and testing data


# In[ ]:





# In[12]:


pip.fit(x_train,y_train) #here we are passing the testing and training data into Random Forest Classifier


# In[13]:


randomForestClassifier = pip.predict(x_test) #here we are predicting the accuracy of the Random Forest Classifier model
randomForestClassifier


# In[14]:


model_result_csv = pd.concat([pd.DataFrame(x_test).reset_index(drop=True),pd.DataFrame(randomForestClassifier)],axis = 1)


# In[17]:


import joblib
# joblib.dump(pip, 'pipeline.pkl')


# In[18]:


pipeline = joblib.load('pipeline.pkl')


# In[19]:


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


url = "https://www.amazon.in/Columbia-Mens-wind-\ resistant-Glove/dp/B0772WVHPS/?_encoding=UTF8&pd_rd\ _w=d9RS9&pf_rd_p=3d2ae0df-d986-4d1d-8c95-aa25d2ade606&pf\ _rd_r=7MP3ZDYBBV88PYJ7KEMJ&pd_rd_r=550bec4d-5268-41d5-\ 87cb-8af40554a01e&pd_rd_wg=oy8v8&ref_=pd_gw_cr_cartx&th=1" 

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
df.rename(column = {0:"review"})
df.to_csv('amazon_review.csv') 


# In[24]:


df.rename(columns = {0:"review"},inplace=True)
df.to_csv('amazon_review.csv') 


# In[25]:


validation = pd.read_csv('amazon_review.csv')
validation['review']


# In[26]:


val_result  = pipeline.predict(validation['review'])
pd.concat([pd.DataFrame(validation['review']).reset_index(drop=True),pd.DataFrame(val_result)],axis = 1)


# In[27]:


print('Accuracy of the model: ',str(np.round(accuracy_score(y_test,randomForestClassifier)*100,2)) + '%')#here we are predicting the accuracy of the Random Forest Classifier model


# In[37]:


pip = Pipeline([
    ('bow',CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC())
])#here we are defining our Support Vector Classifier model in which we will pass the training and testing data


# In[ ]:


pip.fit(x_train,y_train)#here we are passing the testing and training data into Random Forest Classifier


# In[ ]:


supportVectorClassifier = pip.predict(x_test)#here we are predicting the accuracy of the Random Forest Classifier model
supportVectorClassifier


# In[ ]:


print('accuracy of the model:',str(np.round(accuracy_score(y_test,supportVectorClassifier)*100,2)) + '%')#here we are predicting the accuracy of the Random Forest Classifier model


# In[ ]:


import joblib
joblib.dump(pip, 'pipeline.pkl')


# In[ ]:


pipeline = joblib.load('pipeline.pkl')


# In[ ]:


val_result  = pipeline.predict(validation['review'])
pd.concat([pd.DataFrame(validation['review']).reset_index(drop=True),pd.DataFrame(val_result)],axis = 1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pip = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression())
])#here we are defining our Logistic Regression model in which we will pass the training and testing data


# In[ ]:


pip.fit(x_train,y_train)#here we are passing the testing and training data into Random Forest Classifier


# In[ ]:


logisticRegression = pip.predict(x_test)#here we are predicting the accuracy of the Random Forest Classifier model
logisticRegression


# In[ ]:


print('accuracy of the model:',str(np.round(accuracy_score(y_test,logisticRegression)*100,2)) + '%')#here we are predicting the accuracy of the Random Forest Classifier model


# In[ ]:




