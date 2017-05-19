# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:22:25 2017

@author: tensorflow
"""

import tensorflow as tf
import pandas as pd
import urllib
import os
import itertools

BOSTON_TRAINING = './boston/boston_train.csv'
BOSTON_TRAINING_URL = 'http://download.tensorflow.org/data/boston_train.csv'

BOSTON_TEST = './boston/boston_test.csv'
BOSTON_TEST_URL = 'http://download.tensorflow.org/data/boston_test.csv'

BOSTON_PREDICT = './boston/boston_predict.csv'
BOSTON_PREDICT_URL = 'http://download.tensorflow.org/data/boston_predict.csv'

def main():
    if not os.path.exists(BOSTON_TRAINING):
        raw = urllib.urlopen(BOSTON_TRAINING_URL).read()
        with open(BOSTON_TRAINING, 'w') as f:
            f.write(raw)
    if not os.path.exists(BOSTON_TEST):
        raw = urllib.urlopen(BOSTON_TEST_URL).read()
        with open(BOSTON_TEST, 'w') as f:
            f.write(raw)
    if not os.path.exists(BOSTON_PREDICT):
        raw = urllib.urlopen(BOSTON_PREDICT_URL).read()
        with open(BOSTON_PREDICT, 'w') as f:
            f.write(raw)

    COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'mdev']
    FEATURES = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio']
    LABELS = 'mdev'

    training_set = pd.read_csv(BOSTON_TRAINING, skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv(BOSTON_TEST, skipinitialspace=True, skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv(BOSTON_PREDICT, skipinitialspace=True, skiprows=1, names=COLUMNS)
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[10, 10])

    def input_fn(data_set):
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
#        print ({k: data_set[k].values for k in FEATURES})
        labels = tf.constant(data_set[LABELS].values)
        return feature_cols, labels

    regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
    ev = regressor.evaluate(input_fn=lambda: input_fn(prediction_set), steps=1)
    print("evaluate loss:", ev['loss'])

    y = regressor.predict(input_fn=lambda: input_fn(test_set))
    predictions = list(itertools.islice(y, 6))
    print("predictions:", predictions)

if __name__ == "__main__":
    main()