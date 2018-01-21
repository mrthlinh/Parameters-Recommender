'''
Evaluation LSTM model to predict the next token
author: Linh Hoang Truong
'''
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle

# Load Test set
test_file = "./data/train_set_beta.txt" # Evaluate Result of Training Set
# test_file = "./data/test_set_beta.txt" # Evaluate Result of Test Set

# ===============
# Helper Function
# ===============
def build_test_set(file,max_seq):
    sequence = []
    next_token = []
    min_len = 1000000000
    max_len = 0
    with open(file) as f:
        content = f.readlines()
    for i in range(len(content)):
        token = content[i].split()
        token_count = 0
        sen = []
        for t in token:
            if "|||" in t:
                next_token.append(t)
                sequence.append(sen[:])
            sen.append(t)
            token_count += 1
            if (len(sen)>max_seq):
                sen = sen[1:]
            if (len(sen) > max_len):
                max_len = token_count
            if (token_count < min_len):
                min_len = token_count
    print("Maximum sequence length: ", max_len)
    print("Minimum sequence length: ", min_len)
    dict = {'sequence':sequence, 'next_token':next_token,'max_len':max_len}
    return dict

# ===============
# Restore data
# ===============

# Load Pickle file
with open('./pickle/dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
with open('./pickle/reverse_dict.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
with open('./pickle/parameter.pickle', 'rb') as handle:
    parameter = pickle.load(handle)
# with open('./pickle/test_set.pickle', 'rb') as handle:
#     test_set = pickle.load(handle)
vocab_size = len(dictionary)


# Parameters Restore
layer_num = parameter['layer_num']
n_input = parameter['n_input']
max_seq = parameter['max_seq']
n_hidden = parameter['n_hidden'] # number of units in RNN cell

# Load Test set
# # test_file = "./data/train_set_beta.txt"
# test_file = "./data/test_set_beta_3.txt"
test_set = build_test_set(test_file,max_seq)
sequence= test_set['sequence']
next_token= test_set['next_token']

# =============
#    Model
# =============

# tf Graph input
x = tf.placeholder("float", [None, max_seq, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]),name='w')

biases = tf.Variable(tf.random_normal([vocab_size]),name = 'b')
def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, max_seq])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,max_seq,1)

    # n-layer LSTM, each layer has n_hidden units.
    RNN_layer = []
    for i in range(layer_num):
        RNN_layer.append(rnn.BasicLSTMCell(n_hidden))
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    rnn_cell = rnn.MultiRNNCell(RNN_layer)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights) + biases


pred = RNN(x, weights, biases)

# Loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as session:

    saver.restore(session, tf.train.latest_checkpoint('./model_checkpoint/'))

    acc = 0
    # print('len sequence ',len(sequence))
    for i in range(len(sequence)):
        words = sequence[i]
        # if cannot find key in dictionary, change it to 0
        symbols_in_keys = []
        for j in range(len(words)):
            try:
                key = [dictionary[str(words[j])]]
            except:
                key = [0]
            symbols_in_keys.append(key)


        # symbols_in_keys = [[dictionary[str(words[i])]] for i in range(len(words))]
        zero_pad = [[0]] * (max_seq - len(words))
        symbols_in_keys = zero_pad + symbols_in_keys
        keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])

        onehot_pred = session.run(pred, feed_dict={x: keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        # sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index + 1])
        # symbols_in_keys = symbols_in_keys[1:]
        # symbols_in_keys.append(onehot_pred_index)
        print(reverse_dictionary[onehot_pred_index+1], ' -- ', next_token[i])

        if reverse_dictionary[onehot_pred_index+1] == next_token[i]:
            acc += 1
            print('Correct')
    print("Accuracy: ","{:.2f}%".format(100*acc/len(sequence)))

    session.close()
