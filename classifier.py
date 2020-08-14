# This file contains a simple implementation of Naive Bayes to classify data within a dataset as "spam" or "not spam".

import numpy as np
import pandas as pd
import re
from sklearn import naive_bayes, feature_extraction, metrics, model_selection

# Class object for processing data sets.
class SpamDataProcessor:
    def __init__(self):
        self.features = feature_extraction.text.CountVectorizer(stop_words="english")

    # Use regular expressions to remove non-english characters, 
    def preprocess_data(self, dataset):
        dataset['email'] = dataset['email'].apply(lambda x: re.sub(r'([^a-z A-Z]|NUMBER|URL)', "", str(x)))
        dataset['email'] = dataset['email'].apply(lambda x: re.sub(r'\b\w{1,2}\b', "", str(x)))
        dataset['email'] = dataset['email'].apply(lambda x: re.sub(r'\ +', " ", str(x)))

    def process_real_data(self, dataset, split_percentage=0.33, rand_state=99, shuffle=True):
        self.preprocess_data(dataset)
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataset['email'], dataset['label'], test_size=split_percentage, random_state=rand_state, shuffle=shuffle)
        self.features.fit(X_train)
        X_train_trans, X_test_trans = self.features.transform(X_train), self.features.transform(X_test)
        return X_train_trans, X_test_trans, Y_train, Y_test

    def process_fake_data(self, fake_data):
        self.preprocess_data(fake_data)
        fake_vectors = self.features.transform(fake_data['email'])
        return fake_vectors, fake_data['label']


# Class object for creating models and analyzing performance

class BayesModelEvaluator:
    def __init__(self):
        self.model_performance_metrics = pd.DataFrame(
            columns=["alpha value", "train accuracy", "test accuracy", "test precision", "test recall"]
        )
    
    def run_model(self, X_train, X_test, Y_train, Y_test, alpha):
        bayes = naive_bayes.MultinomialNB(alpha=alpha)
        bayes.fit(X_train, Y_train)
        train_score = bayes.score(X_train, Y_train)
        test_score = bayes.score(X_test, Y_test)
        precision_score = metrics.precision_score(Y_test, bayes.predict(X_test))
        recall_score = metrics.recall_score(Y_test, bayes.predict(X_test))
        data_row = {
            "alpha value": alpha,
            "train accuracy": train_score,
            "test accuracy": test_score,
            "test precision": precision_score,
            "test recall": recall_score
        }
        self.model_performance_metrics = self.model_performance_metrics.append(data_row, ignore_index=True)

# Load, process, split, and evaluate "spam or not spam" dataset.

from random import randint

random_state_value = randint(1, 1000)
print("Using random state value: ", random_state_value)

SDP = SpamDataProcessor()
RealDataEval = BayesModelEvaluator()

raw_data = pd.read_csv("spam_or_not_spam.csv")
X_train, X_test, Y_train, Y_test = SDP.process_real_data(raw_data, rand_state=random_state_value)

alpha_list = np.arange(1/1000, 1, 0.1)

for alpha in alpha_list:
    RealDataEval.run_model(X_train, X_test, Y_train, Y_test, alpha=alpha)

print("Results for \"Spam or Not Spam\" dataset: ")
print(RealDataEval.model_performance_metrics)

# Load, process, and evaluate on fake dataset:

FakeDataEval = BayesModelEvaluator()
fake_data = pd.read_csv("fake-spam-data.csv")
X_fake, Y_fake = SDP.process_fake_data(fake_data)

for alpha in alpha_list:
    FakeDataEval.run_model(X_train, X_fake, Y_train, Y_fake, alpha=alpha)

print("\nResults for using Fake Data as test set: ")
print(FakeDataEval.model_performance_metrics)
