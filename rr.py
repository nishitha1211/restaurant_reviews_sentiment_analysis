"""
Author: Nishitha 
Collaborators: Pavan, Venkatesh
Created: 2019

Restaurant reviews sentiment analysis
"""
import pandas as pd
import numpy  as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
#import sklearn.cross_validation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import linear_model
from es import send_tag_to_es

cv = CountVectorizer(max_features=1500)
classifier = MultinomialNB(alpha=0.1)
# classifier = linear_model.LogisticRegression(C=1.5)


def get_corpus(dataset, size=1000):
    corpus = []
    for i in range(0, size):
        review = re.sub('[^a-zA-Z]', ' ', dataset[i])

        review = review.lower()
        review = review.split()

        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        send_tag_to_es(review)
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def get_vector():
    return cv


def get_model():
    return classifier


def fit_transform(corpus):
    return get_vector().fit_transform(corpus).toarray()


def set_model(X_train, y_train):
    classifier.fit(X_train, y_train)


def predict(data):
    corp = get_corpus(data, len(data))
    test = get_vector().transform(corp).toarray()
    return get_model().predict(test)


def start_build():
    dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
    corpus = get_corpus(dataset['Review'], len(dataset))
    X = fit_transform(corpus)
    y = dataset.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    set_model(X_train, y_train)
    # Predicting the Test set results
    y_pred = get_model().predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Accuracy, Precision and Recall

    score1 = accuracy_score(y_test, y_pred)
    score2 = precision_score(y_test, y_pred)
    score3 = recall_score(y_test, y_pred)
    print("\n")
    print("Accuracy is ", round(score1 * 100, 2), "%")
    print("Precision is ", round(score2, 2))
    print("Recall is ", round(score3, 2))


#bernoulli

# classifier = BernoulliNB(alpha=0.8)
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print ("Confusion Matrix:\n",cm)
#
# # Accuracy, Precision and Recall
#
# score1 = accuracy_score(y_test,y_pred)
# score2 = precision_score(y_test,y_pred)
# score3= recall_score(y_test,y_pred)
# print("\n")
# print("Accuracy is ",round(score1*100,2),"%")
# print("Precision is ",round(score2,2))
# print("Recall is ",round(score3,2))
#
# #logistic regression
#
# classifier = linear_model.LogisticRegression(C=1.5)
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print ("Confusion Matrix:\n",cm)
#
# # Accuracy, Precision and Recall
#
# score1 = accuracy_score(y_test,y_pred)
# score2 = precision_score(y_test,y_pred)
# score3= recall_score(y_test,y_pred)
# print("\n")
# print("Accuracy is ",round(score1*100,2),"%")
# print("Precision is ",round(score2,2))
# print("Recall is ",round(score3,2))

