import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\PC\.spyder-py3\nyc.wether.csv")
df
df.info()
df.describe()
df.min()
df.max()
df["CloudCover"].max()
df["CloudCover"].min()
df["CloudCover"].mean()

#%%

df.head()
df.tail()

df["Events"].describe()

df[["EST",'Events']][df["Events"]=='Rain']

df[["EST",'Temperature']][df['Temperature']>=50]
    
df[["EST",'Temperature']][df['Temperature']>=40]

#%%
#Data Cleansing
df.isnull().sum()

df["Events"].isnull().sum()

df['WindSpeedMPH'].isnull().sum()

df.info()

df['WindSpeedMPH'].fillna(df['WindSpeedMPH'].mean(),inplace=True)

df.isnull().sum()

df['Events'].mode()

df['Events'].fillna(df['Events'].mode()[0],inplace=True)

ndf=df.dropna()
ndf

df['PrecipitationIn'].unique()
#%%
my=pd.read_csv(r"C:\Users\PC\.spyder-py3\MyFile.csv")
my
my.describe()
my.info()
my.isnull().sum()

df['PrecipitationIn'].unique()
newdf=df.replace('T',55,inplace=True)

new_my=my.replace('-99999',value=np.NAN)

new_my.isnull().sum()

#%%
#Titanic
titanic=pd.read_csv(r"C:\Users\PC\.spyder-py3\Titanic_Survival_Train.csv")
titanic
titanic.describe()
titanic.isnull().sum()
xtit=titanic.iloc[:,:-10].values
#Titanic_Survival_Train.csv

#%%
