# EX-07 Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.  

## Program:
```
Developed by:SIVABALAN S
RegisterNumber:  212222240100
```
```py
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform (data["Position"])
data.head()

x=data[["Position", "Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=2)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score (y_test,y_pred)
r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```
## Output:
![1M](https://github.com/deepikasrinivasans/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393935/eb753a19-dac5-4e7b-9a8d-2f2865aff93d)
#### MSE value
![2M](https://github.com/deepikasrinivasans/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393935/96291726-5a85-4fb0-88e1-6a1b25f09c66)
#### R2 value
![3M](https://github.com/deepikasrinivasans/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393935/e72f8432-4549-4d43-a57f-1a670814c228)
#### Predicted value
![4M](https://github.com/deepikasrinivasans/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393935/5e68b7d3-4b1e-493b-afc1-bf1fa588bfa3)
#### Result Tree
![5M](https://github.com/deepikasrinivasans/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393935/bdc2ec17-7cd4-4e86-b4fe-ac355afe160c)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
