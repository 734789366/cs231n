# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:38:57 2017

@author: tensorflow
"""

'''Evaluate CIFAR-10'''

import tensorflow as tf
import cfar10_main
import datetime, time
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './eval_dir', '''Directory where to write event logs''')
tf.app.flags.DEFINE_string('eval_data', 'test', '''Either test or train_eval''')
tf.app.flags.DEFINE_string('checkpoint_dir', './train_log', '''Directory where to read model checkpoint''')
tf.app.flags.DEFINE_integer('eval_interval_sec', 60*5, '''How often to run the eval''')
tf.app.flags.DEFINE_integer('num_examples', 10000, '''Number of examples to run''')
tf.app.flags.DEFINE_boolean('run_once', False, '''Whether to run eval only once''')

def eval_once(saver, summary_writer, top_k_op, summary_op):
    '''Run eval once
    
    Args:
        saver: Saver
        summary_writer: Summary Writer
        top_k_op: Top K op
        summary_op: Summary op
    '''
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restore from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checlpoint_path looks something like:
            # /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print("No checkpoint find out")
            return
        
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0 # Count the number of correct predictions
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                
                precision = true_count/total_sample_count
                print('%s: precision @1 = %.3f' % (datetime.now(), precision))
                
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
            
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    
def evaluete():
    '''Eval CIFAR-10 for a number of steps'''
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cfar10_main.input(eval_data=eval_data)
        
        # Build a Graph that computes the logits predictions from the inference model
        logits = cfar10_main.inference(images)
        
        # Calculate predictions
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        
        # Restore the moving average version of the learned variables for eval
        variable_averages = tf.train.ExponentialMovingAverage(cfar10_main.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of summaries
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
            print("sleep", FLAGS.eval_interval_secs)
    
def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluete()
    
    
if __name__ == '__main__':
    tf.app.run()