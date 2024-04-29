# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: A.ARUVI
RegisterNumber: 212222230014. 
*/


import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

### data.head():

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/6cfa4b8c-a292-4072-b06a-558824fd2444)

### data.info():

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/2308a3c8-1585-44a9-af88-21640eb820b1)

### data.isnull().sum():

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/405d3b97-1ac3-4736-bda0-b4ad58a5788f)

### data value count:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/5c88e728-e9ec-4697-a665-49e4eb0c61d8)

### data head() for salary:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/d08c96e8-b89c-4bf0-a62b-e5eccd491a92)

### x.head():

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/4a131079-ea59-4885-8e9c-449a668dafde)

### Accuracy value:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/462b5412-399f-41d7-8a0d-7879845e5151)

### Data prediction:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120443233/8e9866e7-9ce8-44a3-aec0-ff167d7450a0)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
