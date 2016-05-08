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


"""Hyperparameters"""
# The graph is build with conv-relu blocks. One list as below denotes the settings
# for a conv-relu block as in [number_filters, kernel_size]
filt_1 = [32,5]       #Configuration for conv1 in [num_filt,kern_size]
filt_2 = [64,5]       #Configuration for conv1 in [num_filt,kern_size]
num_fc = 1024           #How many neurons for the final fc layer?
num_classes = 10        #How many classes are we targetting?
learning_rate = 1e-4
batch_size = 100
max_iterations = 1000
test_size = 200


#%For quick debugging, we use a two-class version of the MNIST,
#where the targets are encoded one-hot.
# You can use any variation of MNIST. As long as you make sure
#that y_train and y_test are one-hot and X_train and X_test have
#the samples ordered in rows
X_test = np.loadtxt('MNIST_data/X_test.csv', delimiter=',')
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
x = tf.placeholder("float", shape=[None, 784], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
keep_prob = tf.placeholder("float")
bn_train = tf.placeholder(tf.bool)


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
  x_image = tf.reshape(x, [-1,28,28,1])
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
with tf.name_scope('Batch_norm1') as scope:
  ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
  bn_conv1 = ConvolutionalBatchNormalizer(filt_1[0], 0.001, ewma, True)           
  update_assignments = bn_conv1.get_assigner() 
  a_bn1 = bn_conv1.normalize(h_pool1, train=bn_train) 
  h_bn1 = tf.nn.relu(a_bn1) 

with tf.name_scope("Conv2") as scope:
  W_conv2 = weight_variable([filt_2[1], filt_2[1], filt_1[0], filt_2[0]], 'Conv_Layer_2')
  b_conv2 = bias_variable([filt_2[0]], 'bias_for_Conv_Layer_2')
  h_conv2 = tf.nn.relu(conv2d(h_bn1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  width2 = int(np.floor((width1-2)/2))+1 
  size1 = tf.shape(h_pool2)

#  W_fc1 = weight_variable([width2**2 * filt_2[0], num_fc], 'Fully_Connected_layer_1')
#  b_fc1 = bias_variable([num_fc], 'bias_for_Fully_Connected_Layer_1')
#  h_pool2_flat = tf.reshape(h_pool2, [-1, width2**2*filt_2[0]])
#  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope("FC_conv") as scope:
  W_fcc1 = weight_variable([width2,width2,filt_2[0],num_fc],'FC_conv_1')
  b_fcc1 = bias_variable([num_fc],'bias_for_FC_conv_1')
  #h_fcc1 = tf.nn.relu(conv2d(h_pool2,W_fcc1)+b_fcc1)
  h_fcc1 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_fcc1, strides=[1, 1, 1, 1], padding='VALID')+b_fcc1)
  size2 = tf.shape(h_fcc1)

##################
#  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#  W_fc2 = tf.Variable(tf.truncated_normal([num_fc, num_classes], stddev=0.1),name = 'W_fc2')
#  b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
#  h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
with tf.name_scope("Output") as scope:
  W_fcc2 = weight_variable([1,1,num_fc,num_classes],'FC_conv_2')
  b_fcc2 = bias_variable([num_classes],'bias_for_FC_conv_2')
  #h_fcc2 = tf.nn.relu(conv2d(h_fcc1,W_fcc2)+b_fcc2) 
  h_fcc2 = tf.nn.relu(tf.nn.conv2d(h_fcc1, W_fcc2, strides=[1, 1, 1, 1], padding='VALID')+b_fcc2)
  size3 = tf.shape(h_fcc2)
  h_fcc2_strip = tf.squeeze(h_fcc2)
  size4 = tf.shape(h_fcc2_strip)

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
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))

with tf.Session() as sess:
  print('Start session')
  writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/RCNN/log_tb", sess.graph_def)
 

  step = 0
  sess.run(tf.initialize_all_variables())
  for i in range(max_iterations):
    print(i)
    batch_ind = np.random.choice(N,batch_size,replace=False)
#    result = sess.run([size1,size2,size3,size4],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5})
#    print(result[0])
#    print(result[1])
#    print(result[2])
#    print(result[3])
    if i%100 == 1:
      #Measure train performance
      result = sess.run([cost,accuracy,train_step],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5})
      perf_collect[0,step] = result[0]
      perf_collect[2,step] = result[1]
        
        
      #Measure test performance
      test_ind = np.random.choice(Ntest,test_size,replace=False)
      result = sess.run([cost,accuracy,merged], feed_dict={ x: X_test[test_ind], y_: y_test[test_ind], keep_prob: 1.0})
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
      sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5})
 
 
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
