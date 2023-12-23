# Credit Card Fraud Detection

This repository contains code for building a credit card fraud detection model using machine learning techniques. The dataset used for this analysis is named `credit_card_dataset.csv`.

## Dataset Overview
- The dataset consists of credit card transactions, with information on various features such as amount, time, and class.
- The target variable is 'Class', where 0 represents legitimate transactions and 1 represents fraudulent transactions.

## Exploratory Data Analysis (EDA)

### Dataset Information
- Loaded the dataset using Pandas DataFrame.
- Displayed the first 5 and last 5 rows of the dataset.
- Extracted information about the dataset using `info()` method.

### Missing Values
- Checked and reported the number of missing values in each column.

### Class Distribution
- Explored the distribution of legitimate and fraudulent transactions.

### Statistical Measures
- Computed statistical measures (mean, count, etc.) for both legitimate and fraudulent transactions.

## Data Preprocessing

### Handling Unbalanced Data
- Utilized the under-sampling technique to create a balanced dataset.
- Created a new dataset containing a similar distribution of legitimate and fraudulent transactions.

## Model Training

### Features and Targets
- Separated the data into features (`X`) and targets (`Y`).

### Data Splitting
- Split the data into training and testing sets using the `train_test_split` method.

### Logistic Regression Model
- Chose Logistic Regression for the fraud detection model.
- Trained the Logistic Regression model on the training data.

## Model Evaluation

### Accuracy Scores
- Evaluated the model's accuracy on both training and testing datasets.

## Conclusion

Logistic Regression was chosen for its suitability in binary classification tasks, making it a good fit for distinguishing between legitimate and fraudulent credit card transactions based on historical data patterns.

**Note:** Other algorithms such as KNN, SVM, and Decision Trees can also be explored for this task.

Feel free to experiment with different models and parameters to enhance the fraud detection capabilities.

## Instructions to Run the Code

1. Ensure you have the necessary dependencies installed (NumPy, Pandas, and scikit-learn).
2. Download the `credit_card_dataset.csv` file from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).
3. Run the provided Python script to execute the code.

**Note:** The choice of algorithm and parameters can be modified for experimentation.
