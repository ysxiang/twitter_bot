import PreprocessNumerical

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

X_train, y_train, X_test, y_test = PreprocessNumerical.PreprocessNumericalFeatures()


print("-------------------------------SVM--------------------------------")

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test), labels=[0, 1]))
accuracy = sum(clf.predict(X_test) == y_test) * \
    1.0 / X_test.shape[0]

print 'Accuracy with linear is ' + str(accuracy) + ', log loss is ' + str(logls)

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test), labels=[0, 1]))
accuracy = sum(clf.predict(X_test) == y_test) * \
    1.0 / X_test.shape[0]

print 'Accuracy with rbf is ' + str(accuracy) + ', log loss is ' + str(logls)

clf = SVC(kernel='sigmoid', probability=True)
clf.fit(X_train, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test), labels=[0, 1]))
accuracy = sum(clf.predict(X_test) == y_test) * \
    1.0 / X_test.shape[0]

print 'Accuracy with sigmoid is ' + str(accuracy) + ', log loss is ' + str(logls)

print("---------------------------Naive Bayes---------------------------")
gnb = GaussianNB()
# based on training data
y_train_pred = gnb.fit(X_train, y_train).predict(X_train)
print("Based on training data, number of mislabeled points out of a total %d points : %d" % (len(X_train),(y_train != y_train_pred).sum()))

# based on the test data
y_test_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Based on testing data, number of mislabeled points out of a total %d points : %d" % (len(X_test),(y_test != y_test_pred).sum()))

print("---------------------------Random Forest---------------------------")
clf = RandomForestClassifier()
param_grid = {'n_estimators': np.arange(10, 1000, 200),
              'max_features': ['sqrt', 'log2', None],
              'criterion': ['gini', 'entropy'],
              }

'''
All Parameters in a RFC
            'max_depth':[None],
            'min_samples_split':[2],
            'min_samples_leaf':[1],
            'min_weight_fraction_leaf':[0.0],
            'max_leaf_nodes':[None],
            'min_impurity_split':[1e-07],
            'bootstrap':[True],
            'oob_score':[False],
            'n_jobs':[1],
            'random_state':[None],
            'verbose':[0],
            'warm_start':[False],
            'class_weight':[None]
'''
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# CV_rfc.fit(X, Y)

# print('CROSSVALIDATED RF')
# print(CV_rfc.best_params_)
# print(CV_rfc.cv_results_)


## RUN after first calculating the best parameters from the code above.
## USE THE BEST ONES IN THE CLASSIFIER'S ARGUMENTS
best_rfc = RandomForestClassifier(max_features='sqrt', n_estimators=200, criterion='entropy')
best_rfc.fit(X_train, y_train)
print('Random Forest accuracy on training data: {}'.format(best_rfc.score(X_train, y_train)))
print('Random Forest accuracy on test data: {}'.format(best_rfc.score(X_test, y_test)))