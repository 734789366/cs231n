# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:41:00 2017

@author: tensorflow
"""

import tensorflow as tf
import numpy as np
import math
import random
import os
import zipfile
import collections
from six.moves import urllib
from six.moves import xrange
import matplotlib.pyplot as plt

url = 'http://mattmahoney.net/dc/'

# Download the data
def maybe_download(filename, expected_bytes):
    '''Download a file if not present, and make sure it's the right size'''
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified ', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Fail to verify ',filename)
    return filename

# read the data into a list of strings
def read_data(filename):
    '''Extract the first file enclosed in a zip file as a list of words'''
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def build_dataset(words, vocabulary_size):
    ''' Build the word sets for training. This function will review the words read
    from a file and decode each word in it. It will create a dictionary which looks
    like a password translator. It will assign an indepent number representing each word.
    Then it will translate the whole sequence of words into a sequence of numbers.
    
    Args:
        words: all the input data from the file
        vocabulary_size: the word size to extract from data set
    Returns:
        idx_summary: the number representation of the input words
        word_dict: a dict which tells the count of each word. {'the': 245, ...}
        word_idx: a password translator book which is used to translate the words
                to a sequence of numbers
        idx_word: the reverse version of word_idx
    '''
    word_dict = [['UNK', -1]]
    word_dict.extend(collections.Counter(words).most_common(vocabulary_size-1))
    word_idx = dict()
    for word, _ in word_dict:
        word_idx[word] = len(word_idx) # for each word, we give it an index representation
    idx_summary = list() # translate the words into a number sequence, '1 2 5 0 ...'
    unk_count = 0
    for word in words:
        if word in word_dict:
            index = word_idx[word] # for each word in words, we get its index
        else:
            index = 0
            unk_count += 1
        idx_summary.append(index) # This summary will record the position of each word
    word_dict[0][1] = unk_count
    idx_word = dict(zip(word_idx.values(), word_idx.keys()))
    return idx_summary, word_dict, word_idx, idx_word

filename = maybe_download('text8.zip', 31344016)
print (filename)
words = read_data(filename)
print(len(words))
vocabulary_size = 5000
idx_words, word_dict, translator, reverse_translator = build_dataset(words, vocabulary_size)
del words
print('Most common word (+UNK)', word_dict[:5])
print('Sample data', idx_words[:10], [reverse_translator[i] for i in idx_words[:10]])
print(len(idx_words))

#========================================================================================

data_cursor = 0
# Function to generate a training batch for the skip-gram model
def generate_batch(batch_size, num_skips, skip_window):
    global data_cursor
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window
    batch = np.ndarray(batch_size, dtype=np.int32)
    labels = np.ndarray([batch_size, 1], dtype=np.int32)
    span = 2 * skip_window + 1 #[Skip_window target skip_window]
    batch_buffer = collections.deque(maxlen=span)

    for _ in range(span):
        batch_buffer.append(idx_words[data_cursor])
        data_cursor  = (data_cursor + 1) % len(idx_words)

    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = batch_buffer[skip_window]
            labels[i * num_skips + j, 0] = batch_buffer[target]
        batch_buffer.append(idx_words[data_cursor])
        data_cursor = (data_cursor + len(idx_words) - span) % len(idx_words)
        return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_translator[batch[i]], '->', labels[i, 0], reverse_translator[labels[i, 0]])

# Build and train skip-gram model
batch_size = 128
embedding_size = 128 # dimenssion of the embedding vercor
skip_window = 1 # How manu words to consider left and right
num_skips = 2 # How many times to reuse an input to generate a label

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the wors that have a low numeric ID, which by construction
# are also the most frequent
valid_size = 16 # Random set of words to evaluate similarity on
valid_window = 100 # Only pick dev samples in the head of the distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # Number of negative examples to sample

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # Look up embeddings for inputs
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
    loss = tf.reduce_mean(tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))
                        
    # Construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    # Compute the cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # Add variable initializer
    init = tf.global_variables_initializer()

# Begin training
num_steps = 2000
with tf.Session(graph=graph) as sess:
    init.run()
    print('Initialized')
    
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        
        if step % 200 == 0:
            if step > 0:
                average_loss /= 200
                print('average loss at step', step, ':', average_loss)
                average_loss = 0
        if step % 1000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_translator[valid_examples[i]]
                top_k = 8 # Number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_translator[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'more labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
    
try:
    from sklearn.manifold import TSNE
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_translator[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
except ImportError:
     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")