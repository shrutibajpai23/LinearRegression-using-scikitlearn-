import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 

#generate synthetic data 
np.random.seed(42)
x=np.random.rand(100,1)*10 
y=3*x+np.random.randn(100,1)*2

#splitting the dataset 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#fit linear regression model
model=LinearRegression()
model.fit(x_train,y_train)

#make predictions 
y_pred=model.predict(x_test)

#print coeffiecients 
print("slope: ",model.coef_[0][0])
print("intercept: ",model.intercept_[0])


#visualize 
plt.scatter(x_test,y_test,color='blue',label="testing data")
plt.plot(x_test,y_pred,color='red',label="predicted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("linear regression")

#evaluate model 
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("MSE: ", mse)
print("R^2: ", r2)
plt.legend()