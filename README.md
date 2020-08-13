# naive-bayes-spam-detector
Python program that classifies email messages as either spam or not spam using the Scikit-learn machine learning library.

## Prerequisites
This program is written in Python 3 and makes use of the Numpy, Pandas, and Scikit-learn libraries.

## Datasets
This program makes use of two datasets stored in csv format.

The primary dataset "Spam or Not Spam" [originally posted by user Hakan Ozler on Kaggle.com](https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset). This set is a combination of of the files '20030228easyham.tar.bz2' and '20030228_spam.tar.bz2' from [SpamAssassin's public datasets](https://spamassassin.apache.org/old/publiccorpus/). This set contains 2500 examples of ham email messages and 500 examples of spam email messages. The set contains two columns: email and label. Elements in the email column are text strings with numeric values and urls replaced with the words 'NUMBER' and 'URL' respectively. Elements in the label column can have two possible values: 0 if an email is ham(not spam) and 1 if an email is spam.

The secondary dataset contains fake data. This set follows the style of the primary set; two columns for email and label. Data examples within this set are completely synthesized to see how a model classifies completely new data. The current number of examples remains low. More examples will be added to this set in the future.

## Implementation
