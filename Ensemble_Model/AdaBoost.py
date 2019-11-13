"""
Created on  11/10/2019
@author: Jingchao Yang

References:
https://www.datacamp.com/community/tutorials/adaboost-classifier-python
https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464
"""
# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


def ada_use_DT(xtrain, ytrain, xtest):
    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=50,
                             learning_rate=1)
    # Train Adaboost Classifer
    model = abc.fit(xtrain, ytrain)

    # Predict the response for test dataset
    y_pred = model.predict(xtest)

    return y_pred


def ada_use_RF(xtrain, ytrain, xtest):
    # Create adaboost classifer object
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    abc = AdaBoostClassifier(n_estimators=50,
                             base_estimator=rf,
                             learning_rate=1)
    # Train Adaboost Classifer
    model = abc.fit(xtrain, ytrain)

    # Predict the response for test dataset
    y_pred = model.predict(xtest)

    return y_pred


# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

# predicting
DT_pre = ada_use_DT(X_train, y_train, X_test)
RF_pre = ada_use_RF(X_train, y_train, X_test)

print("Accuracy for DT:", metrics.accuracy_score(y_test, DT_pre))
print("Accuracy for RF:", metrics.accuracy_score(y_test, RF_pre))
