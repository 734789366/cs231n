#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:15:53 2017

@author: tensorflow
"""
import tensorflow as tf
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many other types
# of columns that are more complicated and useful
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Tensorflow provides many helper methods to read and setup data sets.
# Here we use 'numpy_input_fn'. We have to tell the function how many batches of
# data (num_epoches) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

print(estimator.evaluate(input_fn=input_fn))