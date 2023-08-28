#linear regression :

#Regression: 
#Enable us to determine the function or mathematical relation between two or more related variables.
#Using this relation we can estimate or pridict the unknown values of one variable from the known values of another variable. this process is known as Regression

#curve fitting by method of least square :
#for a set  of  n paired data  it can be observed that the points cluster around a curve which may be straight line parabola or any other curve. 
#we can find equation of curve which will be passing through most of the points.
#the formation or determining the equation of curve is called curve fitting.

# The fundamental difference between the problem of curve fitting and regression is that :-
# In curve fitting one variable is given as independent or other is dependent but in regression we can take any of the variable as dependent or independent

# two type of relation form independent and dependent variables in regression-
#  1) when x is independent and y is dependent
#      y= f(x)
#  2) when y is independent and x is dependent
#      x= fun(y)

# linear regression: If we fit a linear equation to the given data that it is called as linear regression
# to fit straight line(y=a+bx) to a given data is called as line of regression

# It is used to estimate real values (cost of houses, number of calls, total sales, etc.) based on a continuous variable(s).
# the relationship between independent and dependent variables by fitting the best line. 

# This best-fit line is known as the regression line and is represented by a linear equation Y= a*X + b.

# Y – Dependent Variable
# a – Slope
# X – Independent variable
# b – Intercept

# TYPES-
# 1) Simple linear regression : Simple Linear Regression is characterized by one independent variable.
# 2) Multiple linear regression : Multiple Linear Regression(as the name suggests) is characterized by multiple (more than 1) independent variables.

# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
df= pd.read_csv('/content/Linear Regression - Sheet1.csv')  
df.head(10)

x= df.iloc[:,:-1].values  
y= df.iloc[:, 1].values

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  

#Fitting the Simple Linear Regression model to the training dataset  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train) 

#Prediction of Test and Training set result  
y_pred= regressor.predict(x_test)  
x_pred= regressor.predict(x_train)  
fg=pd.DataFrame(y_pred)
fg.head()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
r2_score = regressor.score(x_test,y_test)
print("Accuracy using R-square",r2_score)

#visualizing the Training set results  
mtp.scatter(x_train, y_train, color="pink")   
mtp.plot(x_train, x_pred, color="black")    
mtp.title(" Linear Regression (training data)")  
mtp.xlabel("X")  
mtp.ylabel("Y")  
mtp.show()  

#visualizing the Test set results  
mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="yellow")    
mtp.title("Linear Regression (testing data)")  
mtp.xlabel("X")  
mtp.ylabel("Y")  
mtp.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

