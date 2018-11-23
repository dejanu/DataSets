# -*- coding: utf-8 -*-
## split data into trainig and testing using pandas
## create model and training it
## serialize model
## RMS check

import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#import cloudpickle
import pickle

def serialize_model(model_name, file_name):
    """ serialize model object into a file"""
    obj_stream = open(file_name,'wb')
    pickle.dump(model_name, obj_stream)
    obj_stream.close()

def deserialize_model(file_name):
    """ deserialize the object from file stream"""
    obj_stream = open(file_name,'rb')
    model = pickle.load(obj_stream)
    return model


#read the dataset from git
try:
    df = pd.read_csv('https://raw.githubusercontent.com/dejanu/DataSets/master/winequality-red.csv', delimiter=";")
except:
    df = pd.read_csv('winequality-red.csv',delimiter=";")

print ("Data features {0}".format(df.head))
print ("Dataframe shape: {0}".format(df.shape))


#try to predict the quality label based on the other features
# features = df.drop('quality',axis =1 )
# label = df['quality']

X_train, X_test, y_train, y_test = train_test_split(df.drop('quality', axis=1), df['quality'], test_size=0.25, random_state=1)

#define linear regression estimator and training it with our wine data
# regr = linear_model.LinearRegression()
# regr.fit(features, label)

regr = linear_model.RidgeCV(alphas= np.arange(0.1,10.0,.5))
regr.fit(X_train, y_train)

##serialize our model
serialize_model(regr,"model.pkl")
#deserialized_model = deserialize_model("model.pkl")

## using out trained model to predict a fake wine where 
## each number represents a feature like pH, acidity
#print (regr.predict([[7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4]]).tolist())

#checking for error
ans = regr.predict(X_test)
print (mean_squared_error(y_test, ans))