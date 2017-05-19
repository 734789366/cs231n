# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:24:14 2017

@author: tensorflow
"""
"""
The MNIST data is split into three parts: 55,000 data points of training data
(mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of
validation data (mnist.validation). This split is very important: it's
essential in machine learning that we have separate data which we don't learn
from so that we can make sure that what we've learned actually generalizes!

As mentioned earlier, every MNIST data point has two parts: an image of a
handwritten digit and a corresponding label. We'll call the images "x" and the
labels "y". Both the training set and test set contain images and their
corresponding labels; for example the training images are mnist.train.images
and the training labels are mnist.train.labels.

We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't
matter how we flatten the array, as long as we're consistent between images.
From this perspective, the MNIST images are just a bunch of points in a
784-dimensional vector space, with a very rich structure
(warning: computationally intensive visualizations)

The result is that mnist.train.images is a tensor (an n-dimensional array)
with a shape of [55000, 784]. The first dimension is an index into the list of
images and the second dimension is the index for each pixel in each image.
Each entry in the tensor is a pixel intensity between 0 and 1, for a particular
pixel in a particular image.

Each image in MNIST has a corresponding label, a number between 0 and 9
representing the digit drawn in the image.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import data
mnist = input_data.read_data_sets('./MNIST', one_hot=True)

# create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# define loss and optimizer
# The raw formulation of cross-entropy if:
# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),reduction_indices=[1]))
y_ = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar("loss", cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
train_writer = tf.summary.FileWriter("./train_log",sess.graph)
merged = tf.summary.merge_all()

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, loss, _ = sess.run((merged, cross_entropy, train_step), feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        train_writer.add_summary(summary, i)
    print("loss=", loss)

# test the trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

