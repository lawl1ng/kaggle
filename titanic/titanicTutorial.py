import numpy as np
import pandas as pd
import os

# Print all files in working directory
for dirname, _, filenames in os.walk('/home/cillian/repos/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Get training, test, and gender data
train_data = pd.read_csv('/home/cillian/repos/kaggle/titanic/data/train.csv')
train_data.head()

test_data = pd.read_csv("/home/cillian/repos/kaggle/titanic/data/test.csv")
test_data.head()

gender_submission_data = pd.read_csv("/home/cillian/repos/kaggle/titanic/data/gender_submission.csv")
gender_submission_data.head()

# Calculate percentage of women and men who survived (from the training data)
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

# First machine learning model
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('~/repos/kaggle/titanic/submissions/submission.csv', index=False)
print("Your submission was successfully saved!")
os.getcwd()
output.head()

# Second machine learning model
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(train_data.head())
print("The predictions are")
print(model.predict(train_data.head()))