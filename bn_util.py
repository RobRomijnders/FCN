# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:44:17 2016

@author: rob
"""

def train_norm(self,x):
      mean, variance = tf.nn.moments(x, [0,1,2])
      assign_mean = self.mean.assign(mean)
      assign_variance = self.variance.assign(variance)
      with tf.control_dependencies([assign_mean, assign_variance]):
        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, self.beta, self.gamma,
            self.epsilon, self.scale_after_norm)
            
  def test_norm(self,x):
      mean = self.ewma_trainer.average(self.mean)
      variance = self.ewma_trainer.average(self.variance)
      local_beta = tf.identity(self.beta)
      local_gamma = tf.identity(self.gamma)
      return tf.nn.batch_norm_with_global_normalization(
          x, mean, variance, local_beta, local_gamma,
          self.epsilon, self.scale_after_norm)