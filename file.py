import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
#credit_card_data = pd.read_csv(r'D:\Documents\Study\AI_ML\SmartCreditCardFraudDetection\creditcard.csv')

# first 5 rows of the dataset
credit_card_data.head()

# last 5 rows of the dataset
credit_card_data.tail()

# dataset informations
credit_card_data.info()

# checking the number of missing values in each column
credit_card_data.isnull().sum()

# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)
#this will print the number of normal and fraudulent transactions along with the total number of columns

# statistical measures of the data(mean, standard deviation, minimum valie, maximum value, percentile)
legit.Amount.describe()

# similarly for fraudulent data
# statistical measures of the data(mean, standard deviation, minimum valie, maximum value, percentile)
fraud.Amount.describe()

# compare the values for both transactions
credit_card_data.groupby('Class').mean()

#random sampling
legit_sample = legit.sample(n=3)

new_dataset = pd.concat([legit_sample, fraud], axis=0)
#axis=0: data will be added row wise, axis=1: data will be added column wise

new_dataset.head()

new_dataset.tail()

# distribution of legit transactions & fraudulent transactions in the new data
new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()
# to show that the nature of the dataset is not changed(alrge extent)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# 20% of the data goes to the testing set and 80% goes to the training data
# stratify will evenly distribute class=0 and class=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression(max_iter=500)  # Increase from default 100

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

