'''
LSTM model to predict the next token
author: Linh Hoang Truong
'''
# Tips for training
# https://danijar.com/tips-for-training-recurrent-neural-networks/

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import collections
import time
import sys
import random

# ===============
# Helper Function
# ===============
start_time = time.time()
model_name = random.randint(1,1000)
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

def read_data(fname):
    output = []
    # max_len = 0
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for i in range(len(content)):
        split = content[i].split()
        output = output + split
        # if len(content[i]) > max_len:
        #     max_len = len(content[i])
    # content = [content[i].split() for i in range(len(content))]
    output = np.array(output)
    output = np.reshape(output, [-1, ])
    return output

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) + 1
        # dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

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
# ===========
# Parameters
# ===========
training_file = './data/train_set_beta.txt'
learning_rate = 0.001
layer_num = 3
n_input = 3
max_seq = 20
epoch = 10
n_hidden = 512 # number of units in RNN cell

# ===========
# Build data
# ===========

# Text file containing words for training
training_data = read_data(training_file)
print("Loaded training data...")

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)
print("Build Data Set...")

test_set = build_test_set(training_file,max_seq)
print("Build Data Set...")


# =============
#    Model
# =============

# tf Graph input
x = tf.placeholder("float", [None, max_seq, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]),name='w')
biases = tf.Variable(tf.random_normal([vocab_size]),name = 'b')

# LSTM model
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

# Predict next token
pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    display_step = int(len(training_data) / 100)
    print("Training....")
    for i in range(epoch):
        last_time = time.time()
        print("Epoch ",i+1)
        # print("[ ",end='',flush=True)
        num_training = len(training_data)
        end_offset = n_input + 1
        step = 0
        acc_total = 0
        loss_total = 0
        progress = 0
        while step < (num_training - n_input):


            symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(step, step+n_input) ]
            zero_pad = [[0]] * (max_seq - n_input)
            symbols_in_keys = zero_pad + symbols_in_keys
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[step+n_input])]-1] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            loss_total += loss
            acc_total += acc
            step += 1

            # Display Progress
            progress = 100 * step / (num_training - n_input)
            sys.stdout.write("\r%d%%" % progress)
            sys.stdout.flush()


        print("")
        print("Average Loss= " + \
              "{:.6f}".format(loss_total/num_training) + ", Average Accuracy= " + \
              "{:.2f}%".format(100*acc_total/num_training) + ", Time: " + \
              "{:.2f}s".format(time.time() - last_time))
        # Save model for each epoch
        saver = tf.train.Saver()
        save_path = saver.save(session, "./model_checkpoint/LSTM_model_epoch_" + str(i+1) + ".ckpt")


    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))



    # while True:
    #     prompt = "%s words: " % n_input
    #     sentence = input(prompt)
    #     sentence = sentence.strip()
    #     words = sentence.split(' ')
    #     try:
    #         symbols_in_keys = [[dictionary[str(words[i])]] for i in range(len(words))]
    #         zero_pad = [[0]] * (max_seq - len(words))
    #         symbols_in_keys = zero_pad + symbols_in_keys
    #         keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
    #
    #         num_next_word = 1
    #         for i in range(num_next_word):
    #             onehot_pred = session.run(pred, feed_dict={x: keys})
    #             onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
    #             sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index+1])
    #             symbols_in_keys = symbols_in_keys[1:]
    #             symbols_in_keys.append(onehot_pred_index)
    #         print(sentence)
    #     except:
    #         print("Word not in dictionary")

    session.close()


# Store data
print("Store pickle data...")
with open('./pickle/dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./pickle/reverse_dict.pickle', 'wb') as handle:
    pickle.dump(reverse_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store parameter
parameter = {'learning_rate':learning_rate, 'layer_num':layer_num,
             'n_input': n_input, 'max_seq':max_seq,
             'n_hidden':n_hidden}
with open('./pickle/parameter.pickle','wb') as handle:
    pickle.dump(parameter, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store test set
with open('./pickle/test_set.pickle','wb') as handle:
    pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

