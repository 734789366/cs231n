# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:12:12 2017

@author: tensorflow
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data, mnist
import time

'''
Build the network structure.
Two hidden layers, and one output layer.
IMAGE_PIXELS -> hidden1_units -> hidden2_units -> NUM_CLASSES 
Output: tensor to compare with labels
'''
def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([mnist.IMAGE_PIXELS, hidden1_units],
                        stddev=2.0/float(mnist.IMAGE_PIXELS)), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                        stddev=2.0/float(hidden1_units)), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, mnist.NUM_CLASSES],
                        stddev=2.0/float(hidden2_units)), name='weights')
        biases = tf.Variable(tf.zeros([mnist.NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    
    return logits

'''
Compare logits and labels.
Calculate the loss using softmax
'''
def loss_func(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                        logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

'''
Define the training ops, and minimize the loss.
'''
def train(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_steps', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step) 
    return train_op

def evaluation(logits, labels):
    '''
    For a classifier model, we can use the in_top_k op.
    It returns a bool tensor with shape [batch_size] that is true for
    the examples where the label is in the top k (here k=1) of all logits
    '''
    correct = tf.nn.in_top_k(logits, labels, 1)
    print("correct", correct)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
    
'''
Define the placeholder for images and labels.
It has batch_size data
'''
def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

'''
Fill in the placeholder with batch_size datas from data_set
'''
def fill_feed_dict(data_set, batch_size, image_pl, label_pl):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
        image_pl: images_feed,
        label_pl: labels_feed
    }
    return feed_dict

def do_eval(sess, eval_correct, batch_size,
            images_placeholder, labels_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples/50
    number_of_examples = steps_per_epoch * 50
    for steps in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, batch_size, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count)/number_of_examples
    print('Num examples: %d, Num correct: %d, precision: %0.04f'%
            (number_of_examples, true_count, precision))        

'''
Do the training process
'''
def run_training():    
    mnist_data = input_data.read_data_sets('./MNIST', one_hot=False)
    batch_size = 50
    hidden1 = 128
    hidden2 = 32
    learning_rate = 0.01
    steps = 5000
    max_steps = 5000
    with tf.Graph().as_default():
        # initialize the placeholder
        img_placeholder, lb_placeholder = placeholder_inputs(batch_size)
        # build the network
        network_logits = inference(img_placeholder, hidden1, hidden2)
        # define the loss 
        image_loss = loss_func(network_logits, lb_placeholder)
        # define the training ops
        train_ops = train(image_loss, learning_rate)
        # Add the op to compare the logists to the labels during evaluation
        eval_correct = evaluation(network_logits, lb_placeholder)
        
        sess = tf.Session()
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./mnist_summary', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for step in xrange(steps):
            start_time = time.time()
            # fill the placeholder with real images and labels
            input_feed_dict = fill_feed_dict(mnist_data.train, batch_size,
                                             img_placeholder, lb_placeholder)

            # calculate the loss and run the training ops to minimize the loss
            _, loss = sess.run([train_ops, image_loss], feed_dict = input_feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print("step: %d, loss=%.2f, duration=%.3f sec"%(step, loss, duration))
                summary_str = sess.run(summary, feed_dict=input_feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_file = './mnist_summary/mnist.ckpt'
                saver.save(sess, checkpoint_file, global_step= step)
                
                print("Training Data Eval:")
                do_eval(sess, eval_correct, batch_size, img_placeholder,
                        lb_placeholder, mnist_data.train)
                        
                print("Validation Data Eval:")
                do_eval(sess, eval_correct, batch_size, img_placeholder,
                        lb_placeholder, mnist_data.validation)
                
                print("Test Data Eval:")
                do_eval(sess, eval_correct, batch_size, img_placeholder,
                        lb_placeholder, mnist_data.test)
                        
if __name__ == '__main__':
    run_training()
