# Linear Regression Project

This project aims to predict house prices using the linear regression algorithm. We use the Boston Housing dataset, which contains information about the housing values in Boston suburbs. We use 12 features as predictors and one target variable (the median value of owner-occupied homes in $1000s).
Requirements

Before running the code, please make sure you have the following packages installed:

    numpy
    pandas
    sklearn

You can install them using the following command:

pip install -r requirements.txt

Dataset

We use the Boston Housing dataset, which is a publicly available dataset that contains information about the housing values in Boston suburbs. The dataset contains 506 observations and 13 features. We use 12 features as predictors and one target variable (the median value of owner-occupied homes in $1000s). The dataset is available in the boston_housing.csv file.
Project Structure

The project structure is organized as follows:

    README.md (documentation of the project)
    linear_regression.py (the main code for the project)
    boston_housing.csv (the dataset)
    requirements.txt (list of required packages)

Running the Code

To run the code, simply execute the linear_regression.py file using the following command:

python linear_regression.py

The output will be the Root Mean Squared Error (RMSE) value, which is a measure of how well the model is performing.
Conclusion

In this project, we have used the linear regression algorithm to predict house prices based on their features. We have used the Boston Housing dataset for this purpose. We have split the dataset into training and testing sets, trained the model on the training data, and evaluated the model performance on the test data. The RMSE value is a measure of how well the model is performing. A lower RMSE value indicates better model performance.
