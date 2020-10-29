#Boxplot
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Data Visualization
import statsmodels.api as sm
#%%

df=pd.read_csv(r"C:\Users\PC\.spyder-py3\Salary_Data.csv")

df.info()
df.describe()
df.sort_values("YearsExperience",inplace=True)
df.mean()
df.head()
df.tail()

df.isnull().sum()
#%%
x=df.iloc[:,:1].values
y=df.iloc[:,1].values


#%%
#Splitting the dataset into training  set and test set
from sklearn.model_selection import train_test_split  #For Machine Learning

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#%%
#Fitting simple linear reg to training set
from sklearn.linear_model import LinearRegression
#Create Model
reg=LinearRegression()
reg.fit(x_train,y_train)

#Predicting
y_pred=reg.predict(x_test)

#Training
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color="blue")

#Testing
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,reg.predict(x_test),color="blue")
#%%
#Predecting test results
mod=sm.OLS(y_train,x_train)
result=mod.fit()
result.summary()
reg.coef_
