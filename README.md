# Linear-Logistic-Regression
Linear-and-Logistic-Regression

# Motivation

In this work, I used two LIBSVM datasets which are pre-processed data originally from UCI data repository.

    Linear regression - Housing dataset (housing scale dataset). Predict housing values in suburbs of Boston. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale.
    Logistic regression - Adult dataset (I only use a3a training dataset). Predict whether income exceeds $50K/yr based on census data.  https: //www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a
 
# About the Project


# Linear Regression:

    Linear Regression is one of the most simple Machine learning algorithm that comes under Supervised Learning technique and used for solving regression problems. It is used for predicting the continuous dependent variable with the help of independent variables.

# Goals of the Project
1. The goal of the Linear regression is to find the best fit line that can accurately predict the output for the continuous dependent variable
2. If single independent variable is used for prediction then it is called Simple Linear Regression and if there are more than two independent variables then such regression is called as Multiple Linear Regression.
3. By finding the best fit line, algorithm establish the relationship between dependent variable and independent variable. And the relationship should be of linear nature.
4. The output for Linear regression should only be the continuous values such as price, age, salary, etc. The relationship between the dependent variable and independent variable can be shown in below image

![image](https://user-images.githubusercontent.com/96569665/233382533-c9c13612-438d-4232-b394-b82937b9a498.png)



# Logistic Regression:

    Logistic regression is one of the most popular Machine learning algorithm that comes under Supervised Learning techniques. It can be used for Classification as well as for Regression problems, but mainly used for Classification problems. Logistic regression is used to predict the categorical dependent variable with the help of independent variables.

The output of Logistic Regression problem can be only between the 0 and 1.

    Logistic regression can be used where the probabilities between two classes is required. Such as whether it will rain today or not, either 0 or 1, true or false etc.

    Logistic regression is based on the concept of Maximum Likelihood estimation. According to this estimation, the observed data should be most probable.

    In logistic regression, we pass the weighted sum of inputs through an activation function that can map values in between 0 and 1. Such activation function is known as sigmoid function and the curve obtained is called as sigmoid curve or S-curve. Consider the below image:

![image](https://user-images.githubusercontent.com/96569665/233382689-e411c1c2-088c-4506-baea-03cc6a4e1115.png)


It was difficult homework because we haven't covered any implementations of ML algorithms before. However it was very interesting to implement them myself. It took a lot of time to start homework because of setting up and getting used to environment and libraries. This homework helped me understand concepts we covered in class bette
Libraries Used


Esssential Libraries Installation

    Install pandas using pip command: import pandas as pd
    Install numpy using pip command: import numpy as np
    Install matplotlib using pip command: import matplotlib
    Install matplotlib.pyplot using pip command: import matplotlib.pyplot as plt
    Install load_svmlight_file using pip command: from sklearn.datasets import load_svmlight_file


# DATA SOURCE:
    Linear Regression Datasethttps://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale.
    Logistic regression Dataset -  https: //www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a


