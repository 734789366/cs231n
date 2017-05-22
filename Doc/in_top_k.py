# coding=utf-8
'''
in_top_k 这个函数的作用，官方文档介绍中：
tf.nn.in_top_k(predictions, targets, k, name=None)这个函数返回一个batch_size
大小的布尔矩阵array.
predictions 是一个 batch_size*classes 大小的矩阵;
targets 是一个 batch_size 大小的类别 index 矩阵;
这个函数的作用是，如果 targets[i] 是 predictions[i][:] 的前 k 个最大值,
则返回的 array[i] = True， 否则，返回的 array[i] = False
'''

import tensorflow as tf

logits = tf.Variable(tf.truncated_normal([10, 5], mean=0.0, stddev=1.0, dtype=tf.float32))
labels = tf.Variable([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

eval_correct = tf.nn.in_top_k(logits, labels, 1)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(logits))
print(sess.run(labels))
print(sess.run(eval_correct))
sess.close()
