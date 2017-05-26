'''
embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。
比如说，ids=[1,3,2],就是返回params中第1,3,2行。返回结果为由params的1,3,2行组成的tensor.
'''

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

embedding = tf.Variable(np.identity(5, dtype=np.int32))
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess.run(tf.global_variables_initializer())

print('embedding: ', sess.run(embedding))
print('input_embedding: ', sess.run(input_embedding, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))