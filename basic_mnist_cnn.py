# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:40:01 2017

@author: tensorflow
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
One should generally initialize weights with a small amount of noise for 
symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons,
it is also good practice to initialize them with a slightly positive initial 
bias to avoid "dead neurons".
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''
Our convolutions uses a stride of one and are zero padded so that the output 
is the same size as the input. 
Our pooling is plain old max pooling over 2x2 blocks.
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets('./MNIST', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
'''
To apply the layer, we first reshape x to a 4d tensor, with the second and
third dimensions corresponding to image width and height 28*28=784, and the final
dimension corresponding to the number of color channels.
'''
x_image = tf.reshape(x, [-1, 28, 28, 1])

'''
First Convolutional Layer
The convolution will compute 32 features for each 5x5 patch. Its weight tensor
will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size,
the next is the number of input channels, and the last is the number of output
channel. So input is one 28*28 image, and outputs are 32 28*28 images
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
'''
We then convolve x_image with the weight tensor, add the bias, apply the ReLU 
function, and finally max pool. The max_pool_2x2 method will reduce the image
size to 14x14.
'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
Second Convolutional Layer
The convolution will compute 64 features for each 5x5 patch.
So the inputs are 32 14*14 images and the outputs are 64 14*14 images.
The max_pool_2x2 meduce will again reduce the size to 7*7
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
The first Fully connected Layer
Now that the image size has been reduced to 7x7, we add a fully-connected layer
with 1024 neurons to allow processing on the entire image. We reshape the tensor
from the pooling layer into a batch of vectors, multiply by a weight matrix,
add a bias, and apply a ReLU.
So the input is a reshaped tensor with size 7*7*64 and the output is a vector 
with size 1024.
'''
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_drop = tf.placeholder(tf.float32)
h_fc_drop = tf.nn.dropout(h_fc1, keep_drop)

'''
The second Fully connected Layer
The input is a vector with size 1024 and the output is a vector with size 10.
'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_drop:1.0})
        print("step: %d, train_accuracy: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_drop:0.5})
    
    if i%1000 == 0 and i != 0:
        print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_drop: 1.0}))
    
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_drop: 1.0}))