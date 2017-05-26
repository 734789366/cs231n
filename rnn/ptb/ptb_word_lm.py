# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:45:50 2017

@author: tensorflow
"""

'''
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
'''
import tensorflow as tf
import numpy as np
import reader
import time
import inspect

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model', 'small', 'a type of model. Possible options are: small, medium, large')
flags.DEFINE_string('data_path', None, 'Where the training/test data is stored')
flags.DEFINE_string('save_path', None, 'Model output directory')
flags.DEFINE_bool('use_fp16', False, 'Train using 16-bit floats instead of 32bit floats')

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
    '''The input data'''
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data)//batch_size) - 1) // num_steps # How many batches per epoch
        self.inp_data, self.target = reader.ptb_producer(data, batch_size, num_steps, name=name)
        
class PTBModel(object):
    '''The PTB model'''
    def __init__(self, is_training, config, input_):
        self.__input = input_
        
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size # number of units
        vocab_size = config.vocab_size
        
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1, but the hyperparameters of the model would need to
        # be different than reported in the paper
        def lstm_cell():
            # with the latest tensorflow source code, the BasicLSTMCell will need a
            # parameter which is unfortunately not defined in Tensorflow 1.0. So
            # to maintain backwards compatbility, we add an argument check here:
            if 'resume' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, data_type())
        
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=data_type())
            input = tf.nn.embedding_lookup()