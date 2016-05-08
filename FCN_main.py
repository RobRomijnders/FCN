# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: rob
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import matplotlib.pyplot as plt
from bn_class import *
from util_RCNN import *

"""Hyperparameters"""
# The graph is build with conv-relu blocks. One list as below denotes the settings
# for a conv-relu block as in [number_filters, kernel_size]
filt_1 = [32,5]       #Configuration for conv1 in [num_filt,kern_size]
filt_2 = [16,5]       #Configuration for conv1 in [num_filt,kern_size]
num_fc = 40           #How many neurons for the final fc layer?
num_classes = 10        #How many classes are we targetting?
learning_rate = 8e-4
batch_size = 100
max_iterations = 800
test_size = 200


#%For quick debugging, we use a two-class version of the MNIST,
#where the targets are encoded one-hot.
# You can use any variation of MNIST. As long as you make sure
#that y_train and y_test are one-hot and X_train and X_test have
#the samples ordered in rows
X_test = np.loadtxt('MNIST_data/X_test.csv', delimiter=',')
X_test_tiled = np.loadtxt('MNIST_data/X_test_tiled.csv', delimiter=',')
y_test = np.loadtxt('MNIST_data/y_test.csv', delimiter=',')
X_train = np.loadtxt('MNIST_data/X_train.csv', delimiter=',')
y_train = np.loadtxt('MNIST_data/y_train.csv', delimiter=',')

#Obtain sizes
N,D = X_train.shape
Ntest = X_test.shape[0]
print('We have %s observations with %s dimensions'%(N,D))

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

#Check for the input sizes
assert (N>X_train.shape[1]), 'You are feeding a fat matrix for training, are you sure?'
assert (Ntest>X_test.shape[1]), 'You are feeding a fat matrix for testing, are you sure?'

# Nodes for the input variables
x = tf.placeholder("float",shape = [None,None], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
keep_prob = tf.placeholder("float")
#bn_train = tf.placeholder(tf.bool)
test_large = tf.placeholder(tf.bool)


# Define functions for initializing variables and standard layers
#For now, this seems superfluous, but in extending the code
#to many more layers, this will keep our code
#read-able
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("Reshaping_data") as scope:
  size_x = tf.shape(x)[1]
  #Use tf.cond() in the line below. We want the path to change depending on 
  # test_large evaluated at session-time. A normal python-like if-statement would
  # be changing at graph-time.
  x_image = tf.cond(test_large, lambda: tf.reshape(x, [-1,110,110,1]), lambda: tf.reshape(x, [-1,28,28,1]))
  image_summ = tf.image_summary("Example_images", x_image)

with tf.name_scope("Conv1") as scope:
  W_conv1 = weight_variable([filt_1[1], filt_1[1], 1, filt_1[0]], 'Conv_Layer_1')
  b_conv1 = bias_variable([filt_1[0]], 'bias_for_Conv_Layer_1')
  a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  #Toggle between the two below lines is using batchnorm or not
  #h_conv1 = tf.nn.relu(a_conv1)
  h_conv1 = a_conv1
  h_pool1 = max_pool_2x2(h_conv1)
  width1 = int(np.floor((28-2)/2))+1 


# ewma is the decay for which we update the moving average of the 
# mean and variance in the batch-norm layers
# The placeholder bn_train denotes wether we are in train or testtime. 
# - In traintime, we update the mean and variance according to the statistics
#    of the batch
#  - In testtime, we use the moving average of the mean and variance. We do NOT
#     update 
#with tf.name_scope('Batch_norm1') as scope:
#  ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
#  bn_conv1 = ConvolutionalBatchNormalizer(filt_1[0], 0.001, ewma, True)           
#  update_assignments = bn_conv1.get_assigner() 
#  a_bn1 = bn_conv1.normalize(h_pool1, train=bn_train) 
#  h_bn1 = tf.nn.relu(a_bn1) 

with tf.name_scope("Conv2") as scope:
  W_conv2 = weight_variable([filt_2[1], filt_2[1], filt_1[0], filt_2[0]], 'Conv_Layer_2')
  b_conv2 = bias_variable([filt_2[0]], 'bias_for_Conv_Layer_2')
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  width2 = int(np.floor((width1-2)/2))+1 

with tf.name_scope("FC_conv") as scope:
  W_fcc1 = weight_variable([width2,width2,filt_2[0],num_fc],'FC_conv_1')
  b_fcc1 = bias_variable([num_fc],'bias_for_FC_conv_1')
  h_fcc1 = tf.nn.relu(tf.add(tf.nn.conv2d(h_pool2, W_fcc1, strides=[1, 1, 1, 1], padding='VALID'),b_fcc1))
  h_fcc1 = tf.nn.dropout(h_fcc1,keep_prob)

with tf.name_scope("Output") as scope:
  W_fcc2 = weight_variable([1,1,num_fc,num_classes],'FC_conv_2')
  b_fcc2 = bias_variable([num_classes],'bias_for_FC_conv_2')
  h_fcc2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_fcc1, W_fcc2, strides=[1, 1, 1, 1], padding='SAME'),b_fcc2))
  h_fcc2_strip = tf.squeeze(h_fcc2)


with tf.name_scope("Softmax") as scope:
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fcc2_strip,y_)
  cost = tf.reduce_sum(loss)
  loss_summ = tf.scalar_summary("cross entropy_loss", cost)

with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)
    # The following block plots for every trainable variable
    #  - Histogram of the entries of the Tensor
    #  - Histogram of the gradient over the Tensor
    #  - Histogram of the grradient-norm over the Tensor
    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      
      numel +=tf.reduce_sum(tf.size(variable))  
        
      h1 = tf.histogram_summary(variable.name, variable)
      h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
      h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
with tf.name_scope("Evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fcc2_strip,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
   
merged = tf.merge_all_summaries()

# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))+1))


# Toggle between either the next line or the two subsequent lines
# Using "with tf.Session() as sess:" closes the tf session automatically
# For normal use, this is convenient. But now, we want to play with the
# tiled images afterwards. Therefore we define a session. We close it
# manually after we finish
#with tf.Session() as sess:
sess = tf.Session()
if True:
  with tf.device("/cpu:0"):
    print('Start session')
    writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/RCNN/log_tb", sess.graph_def)
    step = 0
    sess.run(tf.initialize_all_variables())
    
#    #Debugging purposes:
#    batch_ind = np.random.choice(N,batch_size,replace=False)
#    result = sess.run([size_x],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5,test_large: False})
#    print(result[0])
    
    
    for i in range(max_iterations):
      batch_ind = np.random.choice(N,batch_size,replace=False)
      if i%100 == 1:
        #Measure train performance
        result = sess.run([cost,accuracy,train_step],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5,test_large: False})
        perf_collect[0,step] = result[0]
        perf_collect[2,step] = result[1]
        
        
        #Measure test performance
        test_ind = np.random.choice(Ntest,test_size,replace=False)
        result = sess.run([cost,accuracy,merged], feed_dict={ x: X_test[test_ind], y_: y_test[test_ind], keep_prob: 1.0,test_large: False})
        perf_collect[1,step] = result[0]
        perf_collect[3,step] = result[1]
      
        #Write information for Tensorboard
        summary_str = result[2]
        writer.add_summary(summary_str, i)
        writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
        
        #Print intermediate numbers to terminal
        acc = result[1]
        print("Estimated accuracy at iteration %s of %s: %s" % (i,max_iterations, acc))
        step += 1
      else:
        sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5,test_large: False})
    #Feed and fetch a tiled image
    # Note that we do not feed y_train.
    result = sess.run([h_fcc2],feed_dict = {x:X_test_tiled[0:20], keep_prob: 1.0,test_large: True})
    # Result is a list, so we feed result[0]
    plot_heat(result[0],X_test_tiled[0:20])     #Function to plot heatmaps. Change this line for your own application
 
"""Additional plots"""
plt.figure()
plt.plot(perf_collect[2],label = 'Train accuracy')
plt.plot(perf_collect[3],label = 'Test accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(perf_collect[0],label = 'Train cost')
plt.plot(perf_collect[1],label = 'Test cost')
plt.legend()
plt.show()

# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=/home/rob/Dropbox/ConvNets/tf/log_tb

#Use this line to close the session:
if False:
  sess.close()