

import numpy as np
import os
import tensorflow as tf
import sklearn as sk

###### Do not modify here ###### 

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

# training on MNIST but only on digits 0 to 4
X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]
X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]

###### Do not modify here ###### 

###### functions start #####
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
###### functions end #####

##### main program #####
BATCH_SIZE = 100
LR = 1e-4

xs = tf.placeholder(tf.float32, [None,784])/255.
ys = tf.placeholder(tf.float32, [None,5])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])

#build convolut ion neural network
#layer 1 - conv1 + pool1
layer1_outputsz = 32
W_conv1 = weight_variable([5,5,1,layer1_outputsz]) #patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([layer1_outputsz])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1) #output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1) #output size 14x14x32

#layer 2 - conv2 + pool2
layer2_outputsz = 64
W_conv2 = weight_variable([5,5,layer1_outputsz,layer2_outputsz]) #patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([layer2_outputsz])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2) #output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2) #output size 7x7x64

#layer 3 - fc1
layer3_outputsz = 1024
W_fc1 = weight_variable([7*7*layer2_outputsz, layer3_outputsz])
b_fc1 = bias_variable([layer3_outputsz])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*layer2_outputsz])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#layer 4 - dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer - fc2
classes = 5
W_fc2 = weight_variable([layer3_outputsz,classes])
b_fc2 = bias_variable([classes])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#calculate error rate
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys, 1),logits=prediction)
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)



# mnist_mv = input_data.read_data_sets('MINST_data',one_hot=True)
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0])<1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
        
    sess.run(init)
    y_tr_one_eval = tf.one_hot(y_train1,classes,dtype=tf.int8).eval()

    for i in range(0,len(X_train1),BATCH_SIZE):
        sess.run(train_step, feed_dict={xs:X_train1[i:i+BATCH_SIZE],ys:y_tr_one_eval[i:i+BATCH_SIZE], keep_prob:0.5})
        
        #validation
        if i%2500==0:
            y_val_one_eval = tf.one_hot(y_valid1,classes,dtype=tf.int8).eval()
            print(compute_accuracy(X_valid1, y_val_one_eval))
    print("end of training")

#test
    print("testing!!!")
    y_tst_one_eval = tf.one_hot(y_test1,classes,dtype=tf.int8).eval()
    print(compute_accuracy(X_test1,y_tst_one_eval))
    print("end of program")
   # print ("Precision", sk.metrics.precision_score(y_test1, y_tst_one_eval))
    #print ("Recall", sk.metrics.recall_score(y_test1, y_tst_one_eval))
    #print ("f1_score", sk.metrics.f1_score(y_test1, y_tst_one_eval))
    
#0.969838

