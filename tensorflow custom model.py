#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:35:18 2017

@author: tensorflow
"""
"""
tf.contrib.learn does not lock you into its predefined models.
Suppose we wanted to create a custom model that is not built into TensorFlow.
We can still retain the high level abstraction of data set, feeding, training,
etc. of tf.contrib.learn. For illustration, we will show how to implement our
own equivalent model to LinearRegressor using our knowledge of the lower level
TensorFlow API.

To define a custom model that works with tf.contrib.learn, we need to use
tf.contrib.learn.Estimator.
tf.contrib.learn.LinearRegressor is actually a sub-class of
tf.contrib.learn.Estimator. Instead of sub-classing Estimator, we simply
provide Estimator a function model_fn that tells tf.contrib.learn how it can
evaluate predictions, training steps, and loss.
"""

import tensorflow as tf
import numpy as np

# Declare list of features, we only have one real-valued feature
def model(feature, label, mode):
    # Build linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*feature['x'] + b

    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - label))

    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the appropriate functionality
    return tf.contrib.learn.ModelFnOps(
            mode=mode, predictions=y,
            loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

# Define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)

#evaluate our model
print(estimator.evaluate(input_fn=input_fn, step=10))
