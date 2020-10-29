titanic["Cabin"].mode().value_counts()
titanic["Cabin"].mode()
titanic["Cabin"].dropna()
titanic["Cabin"].dropna().inplace()
titanic["Cabin"].dropna(inplace=True)
titanic.dropna("cabin",axis=1)
titanic.drop("cabin",axis=1)
titanic.drop("Cabin",axis=1)
titanic.drop("Cabin",axis=1,inplace=Ture)
titanic.drop("Cabin",axis=1).inplace())
titanic.drop("Cabin",axis=1).inplace()
titanic.drop("Cabin",axis=1).inplace
titanic.drop("Cabin",axis=1,inplace=True)
titanic.drop("Name",axis=1,inplace=True)
titanic.drop("PassengerId",axis=1,inplace=True)
titanic.drop("Ticket",axis=1,inplace=True)
runfile('C:/Users/PC/.spyder-py3/Titanic.py', wdir='C:/Users/PC/.spyder-py3')
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)
titanic["Embarked"].isnull().sum()
titanic["Embarked"].describe()
titanic["Embarked"].mode()
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

import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\PC\.spyder-py3\nyc.wether.csv")
df
import pandas as pd
import pandas as pd
import numpy as np
import pandas as pd

## ---(Tue Feb 12 10:54:42 2019)---
import pandas as pd
import numpy as np
#%%
#Titanic
titanic=pd.read_csv(r"C:\Users\PC\.spyder-py3\Titanic_Survival_Train.csv")
titanic
titanic.describe()
titanic.describe(include="all")  #Categorial data
titanic.info()
titanic.head()
titanic.tail()
titanic.isnull().sum()
titanic["Cabin"].describe()
titanic["Age"].describe()
titanic["Embarked"].describe()
xtit=titanic.iloc[:,:-10].values
#Titanic_Survival_Train.csv
#%%
#Subseting
myvar=titanic[["Sex","Pclass","Age"]]
type(myvar)
#%%
#Filtering
titanic_age_more_than=titanic[titanic["Age"]<1]
titanic_age_more_than.count()
titanic[["Age","Sex"]][titanic["Age"]>60][titanic["Sex"]=="male"]
#%%
titanic[["Name","Sex","Survived"]][titanic['Sex']=='male'][titanic['Survived']==1].count()
titanic[["Name","Sex","Survived"]][titanic['Sex']=='female'][titanic['Survived']==1].count()
titanic['Sex'][titanic['Survived']==1].value_counts()
titanic['Pclass'].value_counts()
#%%
pd.crosstab(titanic['Sex'],titanic['Survived'])
#%%
#Missing Lab
titanic.isnull().sum()
titanic["Age"].isnull().sum()
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)
titanic["Age"]
titanic["Embarked"].isnull().sum()
titanic["Embarked"].describe()
titanic["Embarked"].mode()
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)
#%%
#Removing useless
titanic.drop("Cabin",axis=1,inplace=True)
titanic.drop("Name",axis=1,inplace=True)
titanic.drop("PassengerId",axis=1,inplace=True)
titanic.drop("Ticket",axis=1,inplace=True)
#%%
from sklearn import preprocessing
colname=['Sex','Embarked']
le={}
for x in colname:
    titanic_train[x]=le[x].fit_transform(titanic_train[x])

for x in colname:
    titanic[x]=le[x].fit_transform(titanic[x])

for x in colname:
    le[x]=preprocessing.LabelEncoder()

for x in colname:
    titanic[x]=le[x].fit_transform(titanic[x])

x=titanic.iloc[:,1:].values    
y=titanic.iloc[:,0].values
from sklearn.model_selection import train_test_split  #For Machine Learning

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
model=classifier.fit(x_train,y_train)
y_pred=model.predict(x_test)
model.scatter(x_train,y_train,color='red')
model.plot(x_train,reg.predict(x_train),color="blue")
y_pred=model.predict(x_test)
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred)
cm