# Importing Required Dependencies & Libraries:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset to Pandas DataFrame:
credit_card_data = pd.read_csv('credit_card_dataset.csv')

# Printing First 5 Rows of Dataset:
credit_card_data.head()

# Printing Last 5 Rows of Dataset:
credit_card_data.tail()

# Extracting Dataset Information:
credit_card_data.info()

# Checking the number of Missing Values in each column:
credit_card_data.isnull().sum()

# Distribution of Legit Transactions & Fraudulent Transactions:
credit_card_data['Class'].value_counts()

# Separating the Data for Analysis:
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

# Statistical Measures of the Data:
legit.Amount.describe()
fraud.Amount.describe()

# Comparing the Values for Both Transactions:
credit_card_data.groupby('Class').mean()

# For Handling this Un-balanced Data, We're using Under-Sampling Technique.
## Building a sample dataset containing similar Distribution of Legit Transactions and Fraudulent Transactions
## Number of Fraudulent Transactions --> 492
legit_sample = legit.sample(n=492)

# Concatenating two DataFrames:
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

# Splitting the data into Features & Targets:
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)
print(Y)

# Split the data into Training data & Testing Data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training:
model = LogisticRegression()

## I'm using Logistic Regression over here, although we can also use KNN algorithm, SVM & Decision Tree Methods.
###############################################################################################################
## Logistic regression is commonly used in credit card fraud detection                                       ##
## because it is well-suited for binary classification tasks, efficiently distinguishing between legitimate  ##
## and fraudulent transactions based on historical data patterns.                                            ##
###############################################################################################################

# Training the Logistic Regression Model with Training Data:
model.fit(X_train, Y_train)

### Model Evaluation: ###
# Accuracy on Training Data:
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# Accuracy on Test Data:
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)