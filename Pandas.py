# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:34:13 2023

@author: virtu
"""

import pandas as pd

df = pd.read_csv('/Users/virtu/Desktop/Python/Dummy_Data.csv',encoding='latin-1')
type(df)

df.shape
df.info()
df.head(10)
df.tail(10)

print(df['First Name'])    #Fetching colm

df.Rank
df.columns    #Headers of dataframe

#iloc:: Integer location

df.iloc[[0, 1],2]

#loc:: Label location

df.loc[[0,1], ['Hash ID','Rank']]

df["Counselor Name"]

###########################################################################################################
#CampusX

import numpy as np
import pandas as pd

data = pd.read_csv('matches - matches.csv')
type(data)

#Functions and Attributes
data.head()
data.tail()

data.shape

data.info()

data.describe()


#Fetch colm and rows by iloc() and loc()

data['winner']
type(data['winner'])   #Series:: When single colm

data[["team1", "team2", "winner"]]
type(data[["team1", "team2", "winner"]])   #Dataframe::when more than one colm
data[["team1", "team2", "winner"]].shape

data.iloc[0]    #Single

data.iloc[1:3]    #Multiple

data.iloc[1:11:2]     #Stepwise multiple

df = data.iloc[:,[4,5,10]]       #Rows and colmns once


#Filtering dataframe on a condition
























