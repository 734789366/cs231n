# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:17:00 2017

@author: Administrator
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn

def cnn_model_fn(features, labels, mode):
    #input layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
    
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
    
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(dense, rate=0.4, training=mode==tf.contrib.learn.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(dropout, units=10)
    
    loss = None
    train_op = None
    
    if mode != tf.contrib.learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")
    
    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name='softmax_tensor')
            }
    return model_fn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main(ununsed_arg):
    mnist = input_data.read_data_sets('./MNIST')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    evel_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.contrib.learn.Estimator(model_fn=cnn_model_fn, model_dir="./cnn_mnist")
    
    tensor_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)
    
    mnist_classifier.fit(
            x=train_data,
            y=train_labels,
            batch_size=100,
            steps=20000,
            monitors=[logging_hook])
    
    metrics = {
            'accuracy':tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.CLASSES)}
    eval_results = mnist_classifier.evaluate(evel_data, eval_labels, metrics=metrics)
    print(eval_results)
    
if __name__ == '__main__':
    tf.app.run()