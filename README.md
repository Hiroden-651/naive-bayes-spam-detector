# naive-bayes-spam-detector
Python program that classifies email messages as either spam or not spam using the Scikit-learn machine learning library.

## Prerequisites
This program is written in Python 3 and makes use of the Numpy, Pandas, and Scikit-learn libraries.

## Datasets
This program makes use of two datasets stored in csv format.

The primary dataset "Spam or Not Spam" [originally posted by user Hakan Ozler on Kaggle.com](https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset). This set is a combination of of the files '20030228easyham.tar.bz2' and '20030228_spam.tar.bz2' from [SpamAssassin's public datasets](https://spamassassin.apache.org/old/publiccorpus/). This set contains 2500 examples of ham email messages and 500 examples of spam email messages. The set contains two columns: email and label. Elements in the email column are text strings with numeric values and urls replaced with the words 'NUMBER' and 'URL' respectively. Elements in the label column can have two possible values: 0 if an email is ham(not spam) and 1 if an email is spam.

The secondary dataset contains fake data. This set follows the style of the primary set; two columns for email and label. Data examples within this set are completely synthesized to see how a model classifies completely new data. The current number of examples remains low. More examples will be added to this set in the future.

## Implementation
This program makes use of two class objects for processing datasets and creating/evaluating naive bayes models.

The first class, "SpamDataProcessor" preprocesses data by removing placeholders, stop-words, and non-English characters using regular expressions before shuffling and splitting the "real" data into training and testing sets and labels. The default values for test set size and random state are 33% and 99 respectively. This class object preprocesses the "fake" data the same way, but does not split it into multiple sets. This object makes use of Scikit-learns 

The second class object, BayesModelEvaluator creates a Naive Bayes model from training and test datasets and alpha values. Whenever an instance of an object creates a model, it stores metrics(accuracy, precision, recall) in a pandas dataframe data member.

## Usage
To use this program, clone this repository, navigate inside, and simply run the following command:
```bash
    python3 classifier.py
```

Below is an example of what the program should output:

```bash
Using random state value:  784
Results for "Spam or Not Spam" dataset: 
   alpha value  train accuracy  test accuracy  test precision  test recall
0        0.001        0.999502       0.976768        0.966667     0.889571
1        0.101        0.999005       0.982828        0.956250     0.938650
2        0.201        0.999005       0.984848        0.962500     0.944785
3        0.301        0.999005       0.986869        0.962963     0.957055
4        0.401        0.999005       0.986869        0.962963     0.957055
5        0.501        0.999005       0.986869        0.962963     0.957055
6        0.601        0.999005       0.985859        0.962733     0.950920
7        0.701        0.998507       0.984848        0.962500     0.944785
8        0.801        0.998507       0.986869        0.974684     0.944785
9        0.901        0.998507       0.986869        0.974684     0.944785

Results for using Fake Data as test set: 
   alpha value  train accuracy  test accuracy  test precision  test recall
0        0.001        0.999502           0.75        0.666667          1.0
1        0.101        0.999005           0.75        0.666667          1.0
2        0.201        0.999005           0.75        0.666667          1.0
3        0.301        0.999005           0.75        0.666667          1.0
4        0.401        0.999005           1.00        1.000000          1.0
5        0.501        0.999005           1.00        1.000000          1.0
6        0.601        0.999005           1.00        1.000000          1.0
7        0.701        0.998507           1.00        1.000000          1.0
8        0.801        0.998507           1.00        1.000000          1.0
9        0.901        0.998507           1.00        1.000000          1.0
```