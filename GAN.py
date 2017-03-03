import tensorflow as tf
import os
import utils

batch_size = 100
iterations = 10000
im_size = 64
layer = 3 
z_size = 100
z_in = tf.placeholder(shape=[batch_size,z_size], dtype=tf.float32)
real_in = tf.placeholder(shape=[batch_size,im_size,im_size,layers], 

g = utils.generator(z_in,im_size, layers,batch_size)
d = utils.discriminator(real_in,batch_size

d_fake = utils.discriminator(g, batch_size, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d, tf.ones_like(d)))

d_loss = d_loss_fake 
