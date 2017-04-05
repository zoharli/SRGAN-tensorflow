import tensorflow as tf
from utils import *
class srResNet:
    def __init__(self,inputs,train_mode=1,name='srResNet'):
        with tf.variable_scope(name):
            self.train_mode=train_mode
            conv1=conv_layer(inputs,64,3,activation=tf.nn.relu,name='conv1')
            block=conv1
            for i in xrange(16):
                block=self.residual_block(block,name='block_'+str(i))
            conv2=conv_layer(block,64,3,name='conv2')
            bn1=batch_norm(conv2)
            sum1=tf.add(bn1,conv1)
            conv3=conv_layer(sum1,256,3,name='conv3')
            ps1=tf.depth_to_space(conv3,2) #pixel-shuffle
            relu2=tf.nn.relu(ps1)
            conv4=conv_layer(relu2,256,3,name='conv4')
            ps2=tf.depth_to_space(conv4,2)
            relu3=tf.nn.relu(ps2)
            self.conv5=conv_layer(relu3,3,3,name='conv5')
    def residual_block(self,inputs,name='block'):
        with tf.variable_scope(name):
            conv1=conv_layer(inputs,64,3,name='conv1')
            bn1=batch_norm(conv1)
            relu=tf.nn.relu(bn1)
            conv2=conv_layer(relu,64,3,name='conv2')
            bn2=batch_norm(conv2)
            return tf.add(inputs,bn2)
