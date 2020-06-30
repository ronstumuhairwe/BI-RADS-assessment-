# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:08:22 2020
This project focuses on using an Artificial Neural Network to predict if a breast tumor is malignant or benign. 
It uses the mammographic datset from UCI machine learning repository which contains the variables (Age, Shape, BI-RADS, Margin, Severity)

@author: Ronald Tumuhairwe

"""
import pandas as pd
import numpy as np

#import the dataset
masses_data = pd.read_csv("mammographic_masses.data")

#You can analyze the data using the following methods

#masses_data.columns
#masses+data.describe()
#masses_data.shape()

#Rename the Headings of each column for easy readibility 
#replace all '?' in the dataset with 'NAN'
masses_data = pd.read_csv('mammographic_masses.data', na_values=['?'],names = ['BI-RADS','Age','Shape','Margin','Density','Severity'])
masses_data.describe()

#we will drop each row that has missing data, so we first locate 
#all rows with missing data  
masses_data.loc[
    (masses_data['Age'].isnull()) |
    (masses_data['Shape'].isnull()) |
    (masses_data['Margin'].isnull()) |
    (masses_data['Density'].isnull()) ]

#if the data seems randomly distributed, go ahead and drop rows with missing data
#by using dropna
masses_data.dropna(inplace=True)
masses_data.describe()

#convert the pandas datafromes to numpy arrays that can be used by scikit_learn. 
#create an array which extracts only the feature data we want to work with (age,shape,margin and density) and another 
#array that contains the classes (severity). You'll also need an array of the feature name labels.

all_features = masses_data[['Age','Shape','Margin','Density']].values
all_classes = masses_data['Severity'].values
feature_names = ['Age','Shape','Margin','Density']  

#normalize the data using preprocessing standardScaler()
#remember we wnat to feed our neural networks data is normalized with a mean of zero
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
all_features_scaled

from keras.layers import Dense
from keras.models import Sequential

def create_model():
    model = Sequential()
    
    #4 feature inputs going into a 6-unit layer 
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    #output layer with a binary classification (benign or malignant)
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    return model

from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

#wrap our keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=0)
#Now we can use scikit_learn's cross_val_score to evaluate this model indentically to the others
cv_scores = cross_val_score(estimator,all_features_scaled, all_classes, cv=10)
cv_scores.mean()