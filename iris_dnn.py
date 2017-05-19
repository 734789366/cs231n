# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:55:08 2017

@author: tensorflow
"""
'''
The Iris data set contains 150 rows of data, comprising 50 samples from each 
of three related Iris species: Iris setosa, Iris virginica, and Iris versicolor.
Each row contains the data for each flower sample: sepal length, sepal width,
petal length, petal width, and flower species.
Flower species are represented as integers, with 0 denoting Iris setosa,
1 denoting Iris versicolor, and 2 denoting Iris virginica.
A training set of 120 samples
A test set of 30 samples
'''
'''
For the iris neural network classifier, while logging training loss, you might
also want to simultaneously evaluate against test data to see how well the
model is generalizing. You can accomplish this by configuring a
ValidationMonitor with the test data (test_set.data and test_set.target),
and setting how often to evaluate with every_n_steps.
ValidationMonitors rely on saved checkpoints to perform evaluation operations,
so you'll want to modify instantiation of the classifier to add a
tf.contrib.learn.RunConfig that includes save_checkpoints_secs, which
specifies how many seconds should elapse between checkpoint saves during training
'''
import numpy as np
import tensorflow as tf
import os
import urllib

IRIS_TRAINING = './iris/iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = './iris/iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

def main():
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'w') as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, 'w') as f:
            f.write(raw)

#load the training and test sets into Datasets using the load_csv_with_header()
#method in learn.datasets.base. The load_csv_with_header() method takes three
#required arguments:

#    filename, which takes the filepath to the CSV file
#    target_dtype, which takes the numpy datatype of the dataset's target value.
#    features_dtype, which takes the numpy datatype of the dataset's feature values.

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
                    filename=IRIS_TRAINING,
                    target_dtype=np.int,
                    features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
                    filename=IRIS_TEST,
                    target_dtype=np.int,
                    features_dtype=np.float32)

#The code first defines the model's feature columns, which specify the data
#type for the features in the data set. All the feature data is continuous,
#so tf.contrib.layers.real_valued_column is the appropriate function to use
#to construct the feature columns. There are four features in the data set
#(sepal width, sepal height, petal width, and petal height), so accordingly
#dimension must be set to 4 to hold all the data.

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
#By default, if no evaluation metrics are specified, ValidationMonitor will
#log both loss and accuracy, but you can customize the list of metrics that
#will be run every 50 steps. To specify the exact metrics you'd like to run
#in each evaluation pass, you can add a metrics param to the ValidationMonitor
#constructor. metrics takes a dict of key/value pairs, where each key is the
#name you'd like logged for the metric, and the corresponding value is a
#MetricSpec object.
# metric_fn. The function that calculates and returns the value of a metric.
# prediction_key. The key of the tensor containing the predictions returned by the model
# label_key. The key of the tensor containing the labels returned by the model,
# as specified by the model's input_fn
# weights_key. Optional. The key of the tensor (returned by the input_fn)
# containing weights inputs for the metric_fn
    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
    }

#Note that in the above log output, by step 600, the model has already
#achieved precision and recall rates of 1.0. This raises the question
#as to whether model training could benefit from early stopping.
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                            test_set.data,
                            test_set.target,
                            every_n_steps=50,
                            metrics=validation_metrics,
                            early_stopping_metric='loss',
                            early_stopping_metric_minimize=True,
                            early_stopping_rounds=200)
#tf.contrib.learn offers a variety of predefined models, called Estimators,
#which you can use "out of the box" to run training and evaluation operations
#on your data. Here, you'll configure a Deep Neural Network Classifier model
#to fit the Iris data. Using tf.contrib.learn, you can instantiate your
#tf.contrib.learn.DNNClassifier with just a couple lines of code

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=(10, 20, 10),
                                                n_classes=3,
                                                model_dir="./iris_model",
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))


#Datasets in tf.contrib.learn are named tuples; you can access feature data and
#target values via the data and target fields. Here, training_set.data and
#training_set.target contain the feature data and target values for the training
#set, respectively, and test_set.data and test_set.target contain feature data
#and target values for the test set.

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

#Now that you've configured your DNN classifier model, you can fit it to the
#Iris training data using the fit method. Pass get_train_inputs as the input_fn,
#and the number of steps to train (here, 2000):

    classifier.fit(input_fn=get_train_inputs, steps=2000, monitors=[validation_monitor])
    accuracy = classifier.evaluate(input_fn=get_test_inputs, steps=1)['accuracy']
    print ("Evaluate accuracy: %g" % accuracy)

    def new_samples():
        return np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predictions = list(classifier.predict(input_fn=new_samples))
    print("predictions for the new classes:%d", predictions)

if __name__ == '__main__':
    main()
