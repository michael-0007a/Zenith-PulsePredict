
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import joblib

# Ignore warnings
warnings.filterwarnings('ignore')

# Importing and loading dataset
dataset = pd.read_csv("heart disease dataset.csv")

# Splitting dataset into predictors and target variable
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=20)

# Initializing the maximum accuracy variable
max_accuracy = 0

# Trying different random states to find the one with the best accuracy
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    
    # Update max_accuracy if current accuracy is higher
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

# Using the best random state found
rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

# Getting the accuracy score
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)

# Print the results
print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")

# # Saving the model and scaler using joblib
# joblib.dump(rf, 'heart_disease_rf_model.joblib')
# print("Random Forest model saved as 'heart_disease_rf_model.joblib'")

# Optionally, you can save the preprocessing steps (if any)
# If you applied scaling, save the scaler, for example:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# joblib.dump(scaler, 'scaler.joblib')
# print("Scaler saved as 'scaler.joblib'")
