# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:07:49 2017

@author: Administrator
"""

'''Build cfar10 network
Summary of available functions:
# compute input images and labels for training. If you would like to run evaluations,
# use inputs() instead.
inputs, labels = distorted_inputs()
# compute inference on the model inputs to make a prediction
predictions = inference(inputs)
# compute the total loss of the predictions with respect to the label
loss = loss(predictions, labels)
# Create a Graph to run a step of training with respect to the loss\
train_op = train(loss, global_steps)
'''
import tensorflow as tf
import cfar10_input
import re, os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("batch_size", 128, '''number of images to process in one batch''')
tf.app.flags.DEFINE_string("data_dir", '.', '''data_dir''')
tf.app.flags.DEFINE_bool('use_fp16', False, '''use tf.float16''')

IMAGE_SIZE = cfar10_input.IMAGE_SIZE
NUM_CLASSES = cfar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cfar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cfar10_input.NUM_EXAMOLES_PER_EPOCH_FOR_EVAL

# constants describing training process
MOVING_AVERAGE_DECAY = 0.9999 # the decay to use for the moving average
NUM_EPOCHS_PER_DECAY = 350.0 # epochs after which learning rate decays
LEARNING_RATE_DECAY_FACTOR = 0.1 # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.1 # Initial learning rate

# if a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the names
# of the summaries when visualizing a model.
TOWER_NAME = 'tower'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cfar-10-binary.tar.gz'

def _activation_summary(x):
    '''Helper to create summaries for activations.
    Create a summary that provides a histogram of activations.
    Create a summary that measures the sparsity of activations.
    
    Args:
        x: tensor
    Returns:
        nothing
    '''
    # Remove 'tower_[0-9] from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/'%TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name+'/activations', x)
    tf.summary.scalar(tensor_name+'/sparsity', tf.nn.zero_fraction(x))
    
def _variable_on_cpu(name, shape, initializer):
    '''Helper to create a Variable stored on CPU memory.
    
    Args:
        name: name of the variable
        shape: list of ints
        initializer = initializer of the Variable
    Returns:
        Variable tensor
    '''
    with tf.device('/cpu0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    '''Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
        name: name of the variables
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable
    Returns:
        Variable Tensor
    '''
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape,
                            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('loss', weight_decay)
    return var

def distorted_inputs():
    '''Construct distorted input for cfar training using the Reader ops.
    
    Returns:
        images: Images. 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
        label: Label. 1-D tensor of [batch_size]
    Raises:
        ValueError: if no data_dir
    '''
    if not FLAGS.data_dir:
        raise ValueError('please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distored_inputs(data_dir,
                                                  batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def input(eval_data):
    '''Construct input for CIFAR evaluation using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, LABEL_SIZE, 3]
        labels: Labels. 1D tensor of [batch_size]
    Raises:
        ValueError: if no data_dir
    '''
    if not FLAGS.data_dir:
        raise ValueError('please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inference(images):
    '''Build cifar-10 model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Return logits.
    '''
    # We instantiate all variables using tf.get_variable() instead of tf.Variable()
    # in order to share variables across multiple GPU training runs. If we only
    # ran this model on a single GPU, we could simplifu this function by replacing
    # all instances of tf.get_variable() with tf.Variable()
    #
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64], 
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='same')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_avtivation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_avtivation, name=scope.name)
        _activation_summary(conv1)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='same', name='pool1')
    
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    
    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64], 
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='same')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_avtivation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_avtivation, name=scope.name)
        _activation_summary(conv2)
        
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='same', name='pool2')
    
    # local3
    with tf.name_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name = scope.name)
        _activation_summary(local3)

    # local4
    with tf.name_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name = scope.name)
        _activation_summary(local4)

    # linear layer(Wx + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and perform the softmax internally for efficiency
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weight', [192, NUM_CLASSES],
                                             stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmx_linear = tf.add(tf.matmul(local4, weights), biases, name = scope.name)
        _activation_summary(softmx_linear)
    
    return softmx_linear

def loss(logits, labels):
    '''Add L2loss to all the trainable variables
    Add summary for 'Loss' and 'Loss/avg'
    Args:
        logits: logits from inference()
        labels: Labels from distored_inputs or inputs(). 1-D tensor of shape [batch_size]
        
    Returns:
        Loss tensor of type float
    '''
    #Calculate the average cross entropy loss across the batch
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')