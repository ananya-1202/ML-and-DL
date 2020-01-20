import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
import zipfile
#import seaborn as sn


d=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(d.head())
print(d.info())
print(d.describe())
#droping the duplicate row 
d=d.drop_duplicates(keep='first')
shape=d.shape
print('the number of categories is  and the total number of the customers are respectively')
print(shape[1],shape[0])
gen=d['gender'].value_counts()
print('Total number of males and female are respectively', gen[0] ,' and ' , gen[1])
sen=d['SeniorCitizen'].value_counts()
print('Total no of senior citizen',sen[1])
sen=d['PaperlessBilling'].value_counts()
print('number of customer applyong for paperless billing ' , sen[0])
#do the same for others aswell 
sub=d.columns
print('The column of the data are:' , )
#checking for missing data 
d.isnull().sum()
#the type f data stored in each category 
print(d.dtypes)
print("gender:charge cost to them ")
print(d.groupby(['gender','MonthlyCharges']).mean())
print("monthly charges Median : ", d.MonthlyCharges.median())
print("monthly charges Mean : ", d.MonthlyCharges.mode())
print("monthly charges Mean : ", d.MonthlyCharges.mean())
print("monthly charges Max : ", d.MonthlyCharges.max())
print("monthly charges Min : ", d.MonthlyCharges.min())


#data visualisation

fig, ax = plt.subplots()
ax.hist(d['tenure'],10)
y1 = list(d.tenure)
y2=list(d.MonthlyCharges) 
plt.boxplot(y1)
plt.boxplot(y2)
fig, ax = plt.subplots()
ax.hist(d['tenure'],10)
plt.subplot(n,2,2)
fig, ax = plt.subplots()
ax.hist(d['MonthlyCharges'],10)
plt.show()
