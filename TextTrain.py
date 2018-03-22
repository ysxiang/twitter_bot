import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from nltk.tokenize import TweetTokenizer, word_tokenize
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


rawdata_bots = pd.read_csv('bots_data.csv', sep=",", encoding='latin1')
rawdata_nonbots = pd.read_csv('nonbots_data.csv', sep=",", encoding='latin1')
rawdata = pd.concat([rawdata_bots, rawdata_nonbots], ignore_index=True)
rawdata.fillna('', inplace=True)
print('total: {}'.format(rawdata.shape))


rawdata.drop(['id'], axis=1, inplace=True)
rawdata.drop(['id_str'], axis=1, inplace=True)
rawdata.drop(['screen_name'], axis=1, inplace=True)
rawdata.drop(['location'], axis=1, inplace=True)
rawdata.drop(['lang'], axis=1, inplace=True)
rawdata.drop(['created_at'], axis=1, inplace=True)
rawdata.drop(['status'], axis=1, inplace=True)
rawdata.drop(['default_profile'], axis=1, inplace=True)
rawdata.drop(['default_profile_image'], axis=1, inplace=True)
rawdata.drop(['has_extended_profile'], axis=1, inplace=True)
rawdata.drop(['name'], axis=1, inplace=True)
rawdata.drop(['followers_count'], axis=1, inplace=True)
rawdata.drop(['verified'], axis=1, inplace=True)
rawdata.drop(['friends_count'], axis=1, inplace=True)
rawdata.drop(['url'], axis=1, inplace=True)
rawdata.drop(['listedcount'], axis=1, inplace=True)
rawdata.drop(['favourites_count'], axis=1, inplace=True)
rawdata.drop(['statuses_count'], axis=1, inplace=True)


description = rawdata['description']

drop_row = []
for i in range(len(description)):
    if not description[i]:
        drop_row.append(i)


# Need to delete the rows with which 'bot' is in its description
rawdata.drop(drop_row, inplace=True)
rawdata = rawdata.reset_index(drop=True)

split = np.random.rand(len(rawdata)) < 0.8
train_data = rawdata[split]
X_train, y_train = train_data['description'], train_data['bot']

test_data = rawdata[~split]
X_test, y_test = test_data['description'], test_data['bot']

X_train2, y_train2 = X_train, y_train
X_test2, y_test2 = X_test, y_test


print("--------------------CountVectorizer --------------------")
# CountVectorizer
count_vec = CountVectorizer(decode_error='ignore', stop_words=stopwords.words("english"))
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)

# TF-IDF
tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_count)
X_test_tfidf = tfidf_trans.transform((X_test_count))

# PCA
pca = PCA()
X_train_tfidf_rd = pca.fit_transform(X_train_tfidf.toarray())
X_test_tfidf_rd = pca.transform(X_test_tfidf.toarray())

print("-------------------------------SVM--------------------------------")

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_tfidf_rd, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
accuracy = sum(clf.predict(X_test_tfidf_rd) == y_test) * \
    1.0 / X_test_tfidf_rd.shape[0]

print 'Accuracy with linear is ' + str(accuracy) + ', log loss is ' + str(logls)

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train_tfidf_rd, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
accuracy = sum(clf.predict(X_test_tfidf_rd) == y_test) * \
    1.0 / X_test_tfidf_rd.shape[0]

print 'Accuracy with rbf is ' + str(accuracy) + ', log loss is ' + str(logls)

clf = SVC(kernel='sigmoid', probability=True)
clf.fit(X_train_tfidf_rd, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
accuracy = sum(clf.predict(X_test_tfidf_rd) == y_test) * \
    1.0 / X_test_tfidf_rd.shape[0]

print 'Accuracy with sigmoid is ' + str(accuracy) + ', log loss is ' + str(logls)

print("---------------------------Naive Bayes---------------------------")
gnb = GaussianNB()
y_pred = gnb.fit(X_train_tfidf_rd, y_train).predict(X_train_tfidf_rd)
print("Based on training data, number of mislabeled points out of a total %d points : %d" % (len(X_train_tfidf_rd),(y_train != y_pred).sum()))
print y_pred

gnb = GaussianNB()
y_pred = gnb.fit(X_train_tfidf_rd, y_train).predict(X_test_tfidf_rd)
print("Based on testing data, number of mislabeled points out of a total %d points : %d" % (len(X_test_tfidf_rd),(y_test != y_pred).sum()))
print y_pred


print("---------------------------Random Forest---------------------------")

clf = RandomForestClassifier()
param_grid = {'n_estimators': np.arange(10, 1000, 200),
              'max_features': ['sqrt', 'log2', None],
              'criterion': ['gini', 'entropy'],
              }

# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# CV_rfc.fit(X, Y)

# print('CROSSVALIDATED RF')
# print(CV_rfc.best_params_)
# print(CV_rfc.cv_results_)


## RUN after first calculating the best parameters from the code above.
## USE THE BEST ONES IN THE CLASSIFIER'S ARGUMENTS
best_rfc = RandomForestClassifier(max_features='sqrt', n_estimators=200, criterion='entropy')
best_rfc.fit(X_train_tfidf_rd, y_train)
print('accuracy on training data: {}'.format(best_rfc.score(X_train_tfidf_rd, y_train)))
print('accuracy on test data: {}'.format(best_rfc.score(X_test_tfidf_rd, y_test)))


print("--------------------TweetTokenizer --------------------")

def tokenize(tweet):
    tknzr = TweetTokenizer()
    try:
        tweet = tweet.lower()
        tokens = tknzr.tokenize(tweet)
        tokens = map(lambda t: t if not t.startswith(
            'http') else '<url>', tokens)
        return tokens
    except:
        return 'NC'

# CountVectorizer
count_vec = CountVectorizer(decode_error='ignore', tokenizer=tokenize, stop_words=stopwords.words("english"))
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)

# TF-IDF
tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_count)
X_test_tfidf = tfidf_trans.transform((X_test_count))

# PCA
pca = PCA()
X_train_tfidf_rd = pca.fit_transform(X_train_tfidf.toarray())
X_test_tfidf_rd = pca.transform(X_test_tfidf.toarray())

# built_in_tokenizer = count_vec.build_tokenizer()
# tokens = built_in_tokenizer(X_train[1])
# print (tokens)
# tokens = built_in_tokenizer(X_train[2])
# print (tokens)
#
# tknzr = TweetTokenizer()
# tokens = tknzr.tokenize(X_train[1])
# print (tokens)
# tokens = tknzr.tokenize(X_train[2])
# print (tokens)

print("-------------------------------SVM--------------------------------")

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_tfidf_rd, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
accuracy = sum(clf.predict(X_test_tfidf_rd) == y_test) * \
    1.0 / X_test_tfidf_rd.shape[0]

print 'Accuracy with linear is ' + str(accuracy) + ', log loss is ' + str(logls)

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train_tfidf_rd, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
accuracy = sum(clf.predict(X_test_tfidf_rd) == y_test) * \
    1.0 / X_test_tfidf_rd.shape[0]

print 'Accuracy with rbf is ' + str(accuracy) + ', log loss is ' + str(logls)

clf = SVC(kernel='sigmoid', probability=True)
clf.fit(X_train_tfidf_rd, y_train)
logls = (log_loss(y_test, clf.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
accuracy = sum(clf.predict(X_test_tfidf_rd) == y_test) * \
    1.0 / X_test_tfidf_rd.shape[0]

print 'Accuracy with sigmoid is ' + str(accuracy) + ', log loss is ' + str(logls)


print("---------------------------Naive Bayes---------------------------")
gnb = GaussianNB()
y_pred = gnb.fit(X_train_tfidf_rd, y_train).predict(X_train_tfidf_rd)
print("Based on training data, number of mislabeled points out of a total %d points : %d" % (len(X_train_tfidf_rd),(y_train != y_pred).sum()))
print y_pred

gnb = GaussianNB()
y_pred = gnb.fit(X_train_tfidf_rd, y_train).predict(X_test_tfidf_rd)
print("Based on testing data, number of mislabeled points out of a total %d points : %d" % (len(X_test_tfidf_rd),(y_test != y_pred).sum()))
print y_pred


print("---------------------------Random Forest---------------------------")

clf = RandomForestClassifier()
param_grid = {'n_estimators': np.arange(10, 1000, 200),
              'max_features': ['sqrt', 'log2', None],
              'criterion': ['gini', 'entropy'],
              }

# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# CV_rfc.fit(X, Y)

# print('CROSSVALIDATED RF')
# print(CV_rfc.best_params_)
# print(CV_rfc.cv_results_)


## RUN after first calculating the best parameters from the code above.
## USE THE BEST ONES IN THE CLASSIFIER'S ARGUMENTS
best_rfc = RandomForestClassifier(max_features='sqrt', n_estimators=200, criterion='entropy')
best_rfc.fit(X_train_tfidf_rd, y_train)
print('accuracy on training data: {}'.format(best_rfc.score(X_train_tfidf_rd, y_train)))
print('accuracy on test data: {}'.format(best_rfc.score(X_test_tfidf_rd, y_test)))


# print("--------------------logistic regression--------------------")
# lr = LogisticRegressionCV()
# lr.fit(X_train_tfidf_rd, y_train)
# logls = (log_loss(y_test, lr.predict_proba(X_test_tfidf_rd), labels=[0, 1]))
# accuracy = sum(lr.predict(X_test_tfidf_rd) == y_test) * \
#     1.0 / X_test_tfidf_rd.shape[0]
#
# print 'Accuracy is ' + str(accuracy) + ', log loss is ' + str(logls)
